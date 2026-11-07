"""Retrieve and rerank preprocessed book chunks from Qdrant."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.embedding_generator import BGE_QUERY_PREFIX, COLLECTION, EMBED_MODEL, Chunk


class ChunkPayload(TypedDict):
    chunk_id: str
    parent_text: str
    section_path: str
    book: str
    pdf_page: int
    page_start: int
    page_end: int
    body_part: str
    category: str


@dataclass(frozen=True)
class Hit:
    payload: ChunkPayload


def _hit(payload: object) -> Hit:
    chunk = Chunk.model_validate(payload)
    return Hit(payload=cast(ChunkPayload, chunk.model_dump()))


BODY_KEYWORDS = {
    "knee": ["knee", "patell", "acl", "anterior cruciate", "cruciate", "meniscus", "meniscal", "meniscectomy", "tibiofemoral", "quadriceps", "mcl", "pcl"],
    "shoulder": ["shoulder", "rotator cuff", "glenohumeral", "scapul", "impingement", "supraspinatus"],
    "hip": ["hip ", "hip.", "hip,", "femoroacetabular", "gluteus", "acetabul"],
    "ankle_foot": ["ankle", "achilles", "plantar fasc", "calcaneal", "metatars"],
    "spine": ["lumbar", "cervical spine", "thoracic spine", "low back", "sacroiliac", "spinal stabiliz"],
    "elbow_wrist_hand": ["elbow", "wrist", "carpal", "forearm", "epicondyl"],
    "chest": ["sternum", "sternal", "sternotomy", "mediastin", "rib cage", "costochond"],
}
LEX_STOP = {"exercises", "exercise", "rehabilitation", "phases", "phase", "after", "strengthening", "management", "program",
            "programs", "treatment", "recommend", "recommended", "which", "give", "best", "pain", "injury", "about", "reconstruction"}
EXPAND = {r"\bacl\b": "anterior cruciate ligament (ACL) reconstruction",
          r"\bpcl\b": "posterior cruciate ligament",
          r"\bmcl\b": "medial collateral ligament",
          r"\boa\b": "osteoarthritis",
          r"\btka\b": "total knee arthroplasty",
          r"ankle sprain": "lateral ankle ligament sprain",
          r"rotator cuff": "rotator cuff shoulder"}
PAGE_REF = re.compile(r",\s*\d{1,3}[tf]?([\u2013\-]\d{1,3}[tf]?)?(,\s*\d{1,3}[tf]?([\u2013\-]\d{1,3}[tf]?)?)*\s*$")


def detect_body_part(text: str, min_hits: int = 4) -> str:
    lower = text.lower()
    scores = {body: sum(lower.count(word) for word in words) for body, words in BODY_KEYWORDS.items()}
    best = max(scores, key=lambda body: scores[body])
    return best if scores[best] >= min_hits else "general"


def expand_query(query: str) -> str:
    for pattern, replacement in EXPAND.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return query


def is_index_page(text: str, threshold: float = 0.35) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= 8 and sum(bool(PAGE_REF.search(line)) for line in lines) / len(lines) >= threshold


def is_reference_like(hit: Hit) -> bool:
    if hit.payload["section_path"].lower().split(" > ")[-1].strip() == "references":
        return True
    return len(re.findall(r"(?m)^\s*\[\d{1,3}\]\s", hit.payload["parent_text"])) >= 4


def lexical_bonus(query: str, hit: Hit) -> float:
    words = [word for word in re.findall(r"[a-z]{5,}", expand_query(query).lower()) if word not in LEX_STOP]
    if not words:
        return 0.0
    path = hit.payload["section_path"].lower()
    text = hit.payload["parent_text"].lower()
    return 3.0 * sum(word[:7] in path for word in words) / len(words) + sum(word[:7] in text for word in words) / len(words)


client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
if not client.collection_exists(COLLECTION):
    raise RuntimeError(f"Collection {COLLECTION} does not exist. Run embedding_generator first.")
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder("BAAI/bge-reranker-base")


def load_chunks() -> dict[str, Hit]:
    chunks: dict[str, Hit] = {}
    offset: int | str | None = None
    while True:
        points, offset = client.scroll(COLLECTION, limit=512, offset=offset, with_payload=True, with_vectors=False)
        for point in points:
            hit = _hit(point.payload)
            chunks[hit.payload["chunk_id"]] = hit
        if offset is None:
            return chunks


CHUNKS = load_chunks()
CHUNK_IDS = list(CHUNKS)
if not CHUNK_IDS:
    raise RuntimeError(f"Collection {COLLECTION} has no chunks. Run embedding_generator first.")


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


BM25 = BM25Okapi([_tok(CHUNKS[identifier].payload["section_path"] + " " + CHUNKS[identifier].payload["parent_text"]) for identifier in CHUNK_IDS])
LAST_POOL: list[Hit] = []


def bm25_search(query: str, body: str | None = None, category: str | None = None, k: int = 30) -> list[Hit]:
    scores = BM25.get_scores(_tok(expand_query(query)))
    output: list[Hit] = []
    for index in cast(list[int], list(np.argsort(-scores))):
        if scores[index] <= 0 or len(output) >= k:
            break
        hit = CHUNKS[CHUNK_IDS[index]]
        if body and hit.payload["body_part"] != body:
            continue
        if category and hit.payload["category"] != category:
            continue
        output.append(hit)
    return output


def _filter(body: str | None, category: str | None) -> models.Filter | None:
    conditions = []
    if body:
        conditions.append(models.FieldCondition(key="body_part", match=models.MatchValue(value=body)))
    if category:
        conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    return models.Filter(must=conditions) if conditions else None


def search(query: str, body: str | None, category: str | None, k: int = 60) -> list[Hit]:
    vector = embedder.encode(BGE_QUERY_PREFIX + expand_query(query), normalize_embeddings=True).tolist()
    points = client.query_points(COLLECTION, query=vector, query_filter=_filter(body, category), limit=k, with_payload=True).points
    return [_hit(point.payload) for point in points]


def retrieve(query: str, body: str | None = None, exercise_mode: bool = False, top: int = 4) -> tuple[list[Hit], str, str | None]:
    global LAST_POOL
    LAST_POOL = []
    detected = detect_body_part(query, min_hits=1)
    body = body or (detected if detected != "general" else None)
    category = "rehab_exercise" if exercise_mode else None
    ladder = [(body, "body part"), ("general", "general body part"), (None, "any body part")] if body else [(None, "any body part")]
    for requested_body, label in ladder:
        candidates = search(query, requested_body, category) + bm25_search(query, requested_body, category)
        seen: set[str] = set()
        unique: list[Hit] = []
        for hit in candidates:
            identifier = hit.payload["chunk_id"]
            if identifier not in seen and not is_index_page(hit.payload["parent_text"]) and not is_reference_like(hit):
                seen.add(identifier)
                unique.append(hit)
        if not unique:
            continue
        LAST_POOL = unique
        pairs = [[query, (hit.payload["section_path"] + "\n" + hit.payload["parent_text"])[:1800]] for hit in unique]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, unique, strict=True), key=lambda item: -(float(item[0]) + lexical_bonus(query, item[1])))
        return [hit for _, hit in ranked[:top]], label, body
    return [], "no match", body
