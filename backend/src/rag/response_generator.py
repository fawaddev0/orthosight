"""Generate answers from retrieved physiotherapy book passages."""

import re
from dataclasses import dataclass
from typing import cast

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag import retriever

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"   # 1.5B hallucinates a lot; try 7B-Instruct with 4-bit (bitsandbytes) if the GPU allows
tok = AutoTokenizer.from_pretrained(MODEL_ID)
llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")

# ---------- safety / scope guards (run BEFORE any search) ----------
def _norm_q(q: str) -> str:   # curly quotes would otherwise bypass the patterns below
    return q.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').strip()

RED_FLAGS = re.compile(
    r"(chest pain|short(ness)? of breath|can'?t breathe|numb(ness)?|tingling|"
    r"can'?t (bear|put|take|walk|stand|straighten|move)\b|cannot (bear|put|walk|stand|move)|unable to (bear|walk|stand|move)|"
    r"\bpop(ped)?\b|gave way|give way|\blocked\b|deformit|fever|severe pain|night pain|"
    r"fractur|\bbroke(n)?\b|dislocat|loss of bladder|saddle|swollen and (red|hot)|blood clot|calf pain)", re.IGNORECASE)
CLAIM_GUARD = re.compile(r"\b(cures?|cured|curing|miracle|guarantee[ds]?)\b|\bin \d+ days?\b", re.IGNORECASE)
EXERCISE_WORDS = re.compile(r"\b(exercise|exercises|rehab|rehabilitation|stretch|stretches|strengthen|strengthening|workout|reps|sets|routine)\b", re.IGNORECASE)

SAFETY_MSG = "⚠️ This may need medical attention. Please see a doctor before doing any exercises."
CLAIM_MSG = ("The books describe exercises for managing conditions, not cures or guaranteed results in a fixed time, so I can't answer that as asked. "
             "You can ask for exercises for a specific body part or condition instead.")

_ENT_SKIP = {"Give", "What", "Which", "How", "Why", "Who", "When", "Where", "Does", "Can", "Could", "Please", "Tell", "Show",
             "List", "Explain", "Describe", "Are", "Should", "Would"}
def missing_entities(query: str, pool: list[retriever.Hit]) -> list[str]:
    """Capitalised names in the question (not the first word) that appear nowhere in the retrieved pages."""
    hay = " ".join((h.payload.get("section_path", "") + " " + h.payload["parent_text"] + " " + h.payload["book"]) for h in pool).lower()
    first = query.split()[0] if query.split() else ""
    out = []
    for w in re.findall(r"\b[A-Z][a-z]{3,}\b", query):
        if w == first or w in _ENT_SKIP: continue
        if w.lower() not in hay and w.lower().rstrip("s") not in hay: out.append(w)
    return out

# ---------- model call and helpers ----------
def _chat(system: str, user: str, max_new_tokens: int = 700) -> str:
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return cast(str, tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip())

def _ctx(hits: list[retriever.Hit], n_chars: int = 2300) -> str:
    return "\n\n".join(f"[{i+1}] {h.payload['parent_text'][:n_chars]}" for i, h in enumerate(hits))

def _src(h: retriever.Hit) -> str:
    sp = " > ".join(h.payload.get("section_path", "").split(" > ")[-2:])
    return f"{h.payload['book']}, PDF page {h.payload['pdf_page']}" + (f" ({sp})" if sp else "")

def _norm(s: str) -> str: return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _find(words: list[str], text: str) -> int:
    if not words: return -1
    m = re.search(r"[\W_]+".join(re.escape(w) for w in words), text, re.IGNORECASE)
    return m.start() if m else -1

def _grounded(name: str, instructions: str, text: str) -> bool:
    """True only if the exercise name, or a 4-word run of its instructions, really appears in the page text."""
    if _find(_norm(name).split()[:4], text) >= 0 and len(_norm(name)) >= 6: return True
    iw = _norm(instructions).split()
    return any(_find(iw[i:i+4], text) >= 0 for i in range(max(0, len(iw) - 3)))

def _excerpt(name: str, instructions: str, text: str, n: int = 330) -> str | None:
    """Verbatim text from the book: starts at the sentence/bullet where the exercise appears, ends at a sentence end."""
    idx = _find(_norm(name).split()[:4], text)
    if idx < 0:
        iw = _norm(instructions).split()
        for i in range(max(0, len(iw) - 3)):
            idx = _find(iw[i:i+4], text)
            if idx >= 0: break
    if idx < 0: return None
    lo = max(0, idx - 140)
    marks = list(re.finditer(r"(?:[.!?]\s+|\u2022\s*|\n)", text[lo:idx]))
    start = lo + marks[-1].end() if marks else idx
    chunk = text[start: start + n]
    if len(chunk) == n:
        cut = max(chunk.rfind(". ", 150), chunk.rfind("? ", 150), chunk.rfind("! ", 150))
        if cut > 0: chunk = chunk[: cut + 1]
    chunk = re.sub(r"\s+", " ", chunk).strip()
    if not re.search(r"\.\s|\.$|\u2022", chunk): return None        # table cells / fragments, not instructions
    return chunk

def uncovered_terms(query: str, hits: list[retriever.Hit]) -> list[str]:
    """Key terms of the question that none of the retrieved pages mention."""
    hay = " ".join(h.payload.get("section_path", "") + " " + h.payload["parent_text"] for h in hits).lower()
    terms = [w for w in re.findall(r"[a-z]{6,}", retriever.expand_query(query).lower()) if w not in retriever.LEX_STOP]
    return sorted({w for w in terms if w[:7] not in hay})

def _supported(value: str, source_text: str) -> str:
    """Keep a sets/reps value only if it has numbers and every number appears in the source page."""
    nums = re.findall(r"\d+", str(value))
    if not nums: return "not specified"
    pool = set(re.findall(r"\d+", source_text))
    return value if all(n in pool for n in nums) else "not specified"

EX_ONE = ("You are a physiotherapy assistant. Use ONLY the context. Extract the exercises or treatment activities it describes "
          "that directly fit the patient's request. Skip general advice, equipment, and anything about other body parts or conditions. "
          'Reply with JSON only: {"exercises":[{"name":"","sets":"","reps":"","instructions":"","precautions":""}]}. '
          'Copy the name and instructions from the context. Write sets and reps ONLY if the context states them, otherwise "not specified". '
          'If nothing relevant, reply {"exercises":[]}.')

class ExerciseProposal(BaseModel):
    name: str = ""
    sets: str = ""
    reps: str = ""
    instructions: str = ""
    precautions: str = ""


class ExerciseBatch(BaseModel):
    exercises: list[ExerciseProposal]


@dataclass(frozen=True)
class ExerciseResult:
    name: str
    sets: str
    reps: str
    precautions: str
    excerpt: str
    hit: retriever.Hit


def _extract_exercises(
    query: str, hits: list[retriever.Hit], n_chars: int = 2300,
    max_items: int = 6, debug: bool = False,
) -> list[ExerciseResult]:
    found: list[ExerciseResult] = []
    seen: set[str] = set()
    for hit in hits:
        context = hit.payload["parent_text"][:n_chars]
        raw = _chat(EX_ONE, f"Patient request: {query}\n\nContext:\n{context}", max_new_tokens=450)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            items = ExerciseBatch.model_validate_json(match.group(0)).exercises if match else []
        except (ValidationError, ValueError):
            items = []
        if debug:
            print(f"   {_src(hit)}: model proposed {len(items)} item(s)")
        for item in items:
            name = _norm(item.name)
            if not name or name in seen or not _grounded(item.name, item.instructions, context):
                continue
            excerpt = _excerpt(item.name, item.instructions, context)
            if not excerpt:
                continue
            precautions = item.precautions if _grounded("", item.precautions, context) else ""
            shown_name = item.name if _find(name.split()[:3], context) >= 0 else "Passage: " + " ".join(excerpt.split()[:7]) + "…"
            found.append(ExerciseResult(
                name=shown_name,
                sets=_supported(item.sets, context),
                reps=_supported(item.reps, context),
                precautions=precautions,
                excerpt=excerpt,
                hit=hit,
            ))
            seen.add(name)
            if len(found) >= max_items:
                return found
    return found

def _best_source(answer_text: str, hits: list[retriever.Hit]) -> retriever.Hit:
    aw = set(re.findall(r"[a-z]{4,}", answer_text.lower()))
    return max(hits, key=lambda h: len(aw & set(re.findall(r"[a-z]{4,}", h.payload["parent_text"].lower()))))

def answer(query: str, debug: bool = False) -> str:
    query = _norm_q(query)
    if RED_FLAGS.search(query): return SAFETY_MSG
    if CLAIM_GUARD.search(query): return CLAIM_MSG
    ex = bool(EXERCISE_WORDS.search(query))
    if ex:
        hits = []
        for top in (5, 6):
            hits, label, body = retriever.retrieve(query, exercise_mode=True, top=top)
            if not hits: continue
            miss = missing_entities(query, retriever.LAST_POOL)
            if miss: return f"I couldn't find '{', '.join(miss)}' in the loaded books, so I won't guess."
            if debug:
                print(f"--- {label} | body={body}")
                for i, hh in enumerate(hits, 1): print(f"[{i}] {_src(hh)} | {hh.payload['parent_text'][:100]!r}")
            items = _extract_exercises(query, hits, debug=debug)
            if items:
                lines = [f"(retrieval: {label}; body part: {body})"]
                unc = uncovered_terms(query, hits)
                if unc: lines.append(f"⚠ None of the best-matching pages mention: {', '.join(unc)}. These are general exercises for this body region, not specific to that.")
                for e in items:
                    lines.append(f"• {e.name}\n  From the book: \"{e.excerpt}\"\n  Sets/reps: {e.sets} / {e.reps}"
                                 + (f"\n  Precautions (from the book): {e.precautions}" if e.precautions else "")
                                 + f"\n  Source: {_src(e.hit)}")
                return "\n".join(lines)
        pages = "; ".join(_src(hh) for hh in hits) if hits else "none"
        return f"No suitable exercise found in the books for this. Pages searched: {pages}"
    hits, label, body = retriever.retrieve(query, exercise_mode=False)
    if not hits:
        return "No safe match found in the books for this question."
    miss = missing_entities(query, retriever.LAST_POOL)
    if miss: return f"I couldn't find '{', '.join(miss)}' in the loaded books, so I won't guess."
    system = ("Answer ONLY from the numbered context, in at most 3 sentences. Copy numbers exactly. "
              "Cite the supporting context number like [2] after each claim; never invent citations. "
              "Ignore any other bracketed reference numbers inside the text. "
              "First check the question's premise: if the context contradicts it or never states it, begin with "
              "'The provided books do not say that.' and then say what they do state. "
              "If the context does not contain the answer, say exactly: I don't know based on the provided books.")
    raw = _chat(system, f"Context:\n{_ctx(hits)}\n\nQuestion: {query}", max_new_tokens=250)
    if raw.lower().startswith("i don't know"): return "I don't know based on the provided books."
    shown = re.sub(r"\s*\[\d+(?:[\u2013\-,]\s*\d+)+\]", "", raw)                  # drop the paper's own multi-number references
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", shown) if 1 <= int(n) <= len(hits)})
    srcs = [hits[i-1] for i in cited] or [_best_source(shown, hits)]
    return shown + "\n\nSources: " + "; ".join(_src(h) for h in srcs)



def main() -> None:
    while True:
        query = input("Question (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query:
            print(answer(query))


if __name__ == "__main__":
    main()
