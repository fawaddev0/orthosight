#!/usr/bin/env python3
"""
book_preprocessor.py - generic book -> clean text pipeline for RAG.

Layout-aware book preprocessing. Heuristic results require review, especially
for OCR, unconventional layouts, formulas, and inferred heading hierarchies.

Usage
-----
    pip install pymupdf
    python book_preprocessor.py books/ -o processed/            # a folder of books
    python book_preprocessor.py books/anatomy.pdf -o processed/ --debug
    python book_preprocessor.py books/ -o processed/ --config my_config.json

Output (per book, in <out>/<book_name>/):
    sections.jsonl   one record per heading-section: text + heading_path + pages  (feed this to your chunker)
    clean.md         human-readable cleaned text with headings
    tables.jsonl     tables extracted SEPARATELY (not mixed into the prose)
    report.json      what was removed and why (use it to tune the config)
    removed.jsonl    (only with --debug) every dropped block with its reason

Pipeline stages
---------------
 1. Open + triage pages (digital vs scanned -> OCR fallback)
 2. Layout extraction (blocks with bbox, font size, bold)
 3. Region exclusion (images, vector figures, tables, rotated text, diagram labels)
 4. Header / footer / page-number removal (cross-page repetition)
 5. Font profiling (body size) -> heading detection + hierarchy (uses embedded PDF outline if present)
 6. Block-level noise removal (captions, question lists, copyright, TOC leaders)
 7. Section-level removal (summary, review questions, references, index, ...)
 8. Text reconstruction (dehyphenate, merge lines/paragraphs, across pages)
 9. Normalisation (unicode, citations, whitespace) + junk-paragraph filter
10. Export + QC report
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from statistics import median
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

if __package__:
    from . import text_boundaries
else:
    import text_boundaries

join_lines = text_boundaries.join_lines

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:  # older installs
    import fitz  # type: ignore

log = logging.getLogger("bookprep")
PIPELINE_VERSION = '2026.10.28.2'


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- triage / OCR
    ocr: bool = True
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    min_chars_per_page: int = 80          # fewer chars + has images -> treat as scanned

    # --- region exclusion
    drop_figure_labels: bool = True
    figure_overlap: float = 0.6           # block must overlap a figure this much
    figure_label_max_words: int = 12      # raster images: labels up to this many words
    vector_label_max_words: int = 12
    figure_min_paths: int = 15            # a vector "figure" needs >= N drawn paths (text boxes have 1-3)
    figure_max_inner_words: int = 60      # a "figure" holding more text than this is a text box -> keep it
    table_max_block_words: int = 40       # long paragraphs inside a detected table are kept (false positives)
    detect_tables: bool = True
    table_overlap: float = 0.6
    keep_tables_separately: bool = False
    drop_rotated_text: bool = True

    # --- header / footer
    margin_zone: float = 0.09             # top/bottom 9% of the page
    min_repeat_pages: int = 4             # same margin text on >= N pages -> running header/footer
    margin_max_lines: int = 2

    # --- headings
    heading_size_ratio: float = 1.12      # font >= body * ratio
    max_heading_words: int = 18
    max_heading_lines: int = 3
    use_pdf_outline: bool = True

    # --- noise
    drop_captions: bool = True
    drop_question_lists: bool = True
    drop_copyright_lines: bool = True
    drop_chapter_summaries: bool = True
    drop_footnotes: bool = True
    drop_student_case_studies: bool = True
    skip_heading_patterns: list = field(default_factory=lambda: [
        # Keep clinical summaries, precautions, goals, and treatment outcomes.
        # Only explicitly pedagogical objectives are excluded by default.
        r"^(learning |chapter |performance )(objectives|outcomes|goals)\b",
        # practice material
        r"^(review|study|practice|self[- ]assessment|discussion|revision|chapter|multiple[- ]choice|short[- ]answer|essay|viva)\s+(questions?|quiz|quizzes|problems|exercises)\b",
        r"^(questions?|quiz|self[- ]?test|test yourself|mcqs?|check your (understanding|knowledge|progress))$",
        r"^(self[- ]assessment|multiple[- ]choice questions?)$",
        r"^answers?( key| to .*)?$",
        r"^(independent learning activities|critical thinking and discussion|laboratory practice)$",
        # front / back matter
        r"^(references?|bibliography|further reading|suggested readings?|recommended reading|works cited|"
        r"glossary|index|subject index|author index|abbreviations|list of (figures|tables|abbreviations|illustrations|contributors)|"
        r"acknowledg(e)?ments?|about the authors?|contributors|dedication|foreword|preface|"
        r"table of contents|contents|copyright|title page|half[- ]title)$",
    ])
    caption_regex: str = (
        r"^(fig(?:ure)?s?|table|plate|photo(?:graph)?|illustration|image|diagram|chart|graph)\.?\s*"
        r"([\dIVXivx]+(?:[.\-\u2013:]\d+)?[a-zA-Z]?)"
    )
    noise_block_patterns: list = field(default_factory=lambda: [
        r"all rights reserved", r"\bisbn\b", r"printed in ", r"\bcopyright\b|\u00a9",
        r"^(source|courtesy|reproduced|adapted|modified)( from| by| with| of)?\b",
        r"^https?://\S+$", r"^www\.\S+$", r"^\(?continued\)?$",
    ])

    # --- text cleanup
    strip_superscript_citations: bool = True  # raised reference numbers: "...trunk.30,37"
    strip_figure_refs: bool = True         # "(see Fig. 5.22)", "VIDEO 17.2"
    callout_labels: list = field(default_factory=lambda: [
        r"^(clinical tips?|focus on evidence|evidence in focus|precautions?|contraindications?|cautions?|"
        r"warnings?|notes?|clinical pearls?|red flags?|clinical insights?)$",
    ])
    strip_numeric_citations: bool = True   # [12], [3-5], [1,2]
    strip_author_year_citations: bool = True  # (Smith et al., 2010)
    strip_footnote_markers: bool = True
    min_paragraph_words: int = 4
    min_alpha_ratio: float = 0.5
    fail_on_review: bool = False
    page_order_overrides: dict = field(default_factory=dict)
    compound_prefixes: list = field(default_factory=lambda:['self','well','ill'])
    flag_repeated_sentences: bool = True
    flag_figure_dependencies: bool = True

    @classmethod
    def load(cls, path: str | None) -> "Config":
        cfg = cls()
        if path:
            for k, v in json.loads(Path(path).read_text()).items():
                if not hasattr(cfg, k):
                    raise KeyError(f"Unknown config key: {k}")
                setattr(cfg, k, v)
        return cfg


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    page: int                  # 0-based
    bbox: tuple
    y0n: float                 # normalised vertical position (0 top .. 1 bottom)
    y1n: float
    lines: list
    size: float
    bold: bool
    first_bold: bool
    n_chars: int
    bold_frac: float = 1.0
    kind: str = "text"         # text | heading | dropped
    level: int = 0
    reason: str = ""
    source_id: str = ""
    region: str = "main"
    column: str = ""
    font: str = ""
    color: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.lines)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


FLAGS = fitz.TEXT_PRESERVE_IMAGES | fitz.TEXT_MEDIABOX_CLIP
BULLET_RE = re.compile(r"^\s*([\u2022\u25cf\u25cb\u25aa\u25a0\u25ba\u25e6\u27a2\u2013\u2014\u00b7*\-])\s*(?=\S)")
NUMBERED_RE = re.compile(r"^\s*\d{1,2}[.)]\s+[A-Z]")
PAGE_NUM_RE = re.compile(r"^[\W_]*(?:page\s*)?(\d{1,4}|[ivxlcdm]{1,7})[\W_]*$", re.I)
TOC_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$|(\s\.){4,}")


def _overlap(a, b) -> float:
    """Fraction of rect a covered by rect b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    area = (a[2] - a[0]) * (a[3] - a[1])
    if x1 <= x0 or y1 <= y0 or area <= 0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / area


def _is_bold(span) -> bool:
    f = span["font"].lower()
    return bool(span["flags"] & 16) or "bold" in f or "black" in f or "heavy" in f


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# --------------------------------------------------------------------------- #
# Stage 1-3: extraction with region exclusion
# --------------------------------------------------------------------------- #
SUP_RE = re.compile(r"^[\d,\s\u2013\-]+$")
UNIT_PREV = re.compile(r"(?:^|\s)(?:m|cm|mm|km|ft|in)$")


def _line_spans(l, strip_sup: bool):
    """Spans of a line, minus raised reference numbers (citation superscripts)."""
    spans = [s for s in l["spans"] if s["text"].strip()]
    if not spans or not strip_sup:
        return l["spans"]
    # Numeric citations can be longer than all the prose on this line. They
    # must never establish the reference baseline or reference font size.
    prose = [s for s in spans if re.search(r"[A-Za-z]", s["text"]) and not (s["flags"] & 1)]
    dom = max(prose or spans, key=lambda s: s["size"])
    keep, prev = [], ""
    for s in l["spans"]:
        t = s["text"].strip()
        if t and SUP_RE.fullmatch(t):
            small = s["size"] <= 0.85 * dom["size"]
            raised = s["origin"][1] < dom["origin"][1] - 0.12 * dom["size"]
            if (s["flags"] & 1) or (small and raised):
                if not (t in {"2", "3"} and UNIT_PREV.search(prev.rstrip())):
                    continue
        keep.append(s)
        prev += s["text"]
    return keep


def numeric_table_regions(data, previous, page_height):
    """Conservative borderless tables: repeated aligned label/value cells.

    Unlike page-wide text-table detection, this does not interpret ordinary
    prose columns as cells. A value column must start with a numeric value.
    A single continuation row needs a matching table at the previous page edge.
    """
    lines=[]
    for b in data['blocks']:
        if b['type']!=0:
            continue
        for ln in b['lines']:
            if abs(ln['dir'][0])<.9:
                continue
            text=''.join(s['text'] for s in ln['spans']).strip()
            if text:
                lines.append({'text':text,'bbox':ln['bbox'],
                              'size':max(s['size'] for s in ln['spans'])})
    rows=[]
    for ln in sorted(lines,key=lambda x:(x['bbox'][1],x['bbox'][0])):
        row=next((r for r in reversed(rows) if abs(r[0]['bbox'][1]-ln['bbox'][1])<2.5),None)
        if row is None:
            rows.append([ln])
        else:
            row.append(ln)
    candidates=[]
    for row in rows:
        cells=sorted(row,key=lambda x:x['bbox'][0])
        if len(cells)!=2:
            continue
        label,value=cells
        if (len(label['text'].split())<=8 and len(value['text'].split())<=5
                and re.search(r'[A-Za-z]',label['text'])
                and not re.search(r'[.!?;:]$',label['text'])
                and re.match(r'^[+\-]?\d+(?:[.,]\d+)?(?:\s|\u00b0|%|$)',value['text'])
                and value['bbox'][0]-label['bbox'][2]>24
                and abs(label['size']-value['size'])<1):
            candidates.append((label,value))
    groups=[]
    for pair in candidates:
        label,value=pair
        match=next((g for g in groups if abs(g[-1][0]['bbox'][0]-label['bbox'][0])<5
                    and abs(g[-1][1]['bbox'][0]-value['bbox'][0])<5
                    and 0<label['bbox'][1]-g[-1][0]['bbox'][1]<max(label['size']*3,20)),None)
        if match is None:
            groups.append([pair])
        else:
            match.append(pair)
    regions,next_edge=[],[]
    for group in groups:
        first,last=group[0],group[-1]
        continued=first[0]['bbox'][1]<page_height*.25 and any(
            abs(first[0]['bbox'][0]-p[0])<5 and abs(first[1]['bbox'][0]-p[1])<5
            and abs(first[0]['size']-p[2])<1 for p in previous)
        if len(group)<3 and not continued:
            continue
        rect=(min(c['bbox'][0] for row in group for c in row)-1,
              min(c['bbox'][1] for row in group for c in row)-1,
              max(c['bbox'][2] for row in group for c in row)+1,
              max(c['bbox'][3] for row in group for c in row)+1)
        regions.append((rect,[[c['text'] for c in row] for row in group]))
        if last[0]['bbox'][3]>page_height*.8:
            next_edge.append((last[0]['bbox'][0],last[1]['bbox'][0],last[0]['size']))
    return regions,next_edge


def extract_pages(doc, cfg: Config, report: dict):
    blocks: list[Block] = []
    tables: list[dict] = []
    ocr_pages = 0
    previous_numeric_tables=[]

    for pno in range(len(doc)):
        page = doc[pno]
        W, H = page.rect.width, page.rect.height
        d = page.get_text("dict", flags=FLAGS, sort=False)

        # --- Stage 1: scanned-page triage -> OCR fallback
        n_chars = sum(len(s["text"]) for b in d["blocks"] if b["type"] == 0
                      for l in b["lines"] for s in l["spans"])
        ocr_used = False
        if cfg.ocr and n_chars < cfg.min_chars_per_page and page.get_images():
            try:
                tp = page.get_textpage_ocr(language=cfg.ocr_lang, dpi=cfg.ocr_dpi, full=True)
                d = page.get_text("dict", flags=FLAGS, textpage=tp, sort=False)
                ocr_used = True
                ocr_pages += 1
            except Exception as e:  # tesseract missing etc.
                log.warning("OCR failed on page %d: %s", pno + 1, e)
                report.setdefault("extraction_warnings", []).append({"page": pno+1, "reason": "ocr_failed", "error": str(e)})
        if n_chars < cfg.min_chars_per_page and not ocr_used:
            report.setdefault("extraction_warnings", []).append({"page": pno+1, "reason": "sparse_page_without_ocr"})

        # Enclosing single rectangular borders represent independent callouts.
        # Filled title bars alone are not independent regions.
        sidebars = []
        drawings = page.get_drawings() if not ocr_used else []
        for dr in drawings:
            r = dr["rect"]
            if (dr['type'] == 's' and len(dr['items']) == 1 and dr['items'][0][0] == 're'
                    and 100 < r.width < W * .95 and 80 < r.height < H * .9
                    and r.x0 >= 0 and r.y0 >= 0):
                sidebars.append(tuple(r + (-3,-3,3,3)))

        # --- Stage 3: regions to exclude
        fig_rects = []                                   # (rect, "img" | "vec")
        if cfg.drop_figure_labels and not ocr_used:
            for b in d["blocks"]:
                if b["type"] == 1:
                    r = b["bbox"]
                    a = (r[2] - r[0]) * (r[3] - r[1])
                    if (2500 < a < 0.7 * W * H
                            and not (r[2]-r[0]>4*(r[3]-r[1]) and r[3]-r[1]<65)):
                        fig_rects.append((tuple(r), "img"))
            try:
                paths = [(tuple(dr["rect"]),len(dr['items'])) for dr in drawings]
                for r in page.cluster_drawings():
                    if r.width > 25 and r.height > 60 and r.width*r.height>2000 and r.width * r.height < 0.7 * W * H:
                        # Straight segments may have zero-area rectangles. Area
                        # overlap cannot count them; rectangle containment can.
                        envelope=fitz.Rect(r)+(-1,-1,1,1)
                        n_paths = sum(n for q,n in paths if envelope.contains(fitz.Rect(q)))
                        if n_paths >= cfg.figure_min_paths:
                            fig_rects.append((tuple(r), "vec"))
            except Exception as exc:
                report.setdefault("extraction_warnings", []).append({"page": pno+1, "reason": "figure_detection_failed", "error": str(exc)})

        table_rects = []
        if cfg.detect_tables and not ocr_used:
            try:
                for t in page.find_tables().tables:
                    rows=t.extract()
                    graphic_caption=any(b['type']==0 and re.match(r'^Fig(?:ure)?\.?\s*\d',
                        ' '.join(''.join(s['text'] for s in ln['spans']) for ln in b['lines']),re.I)
                        and -3 <= b['bbox'][1]-t.bbox[3] <= 45
                        and min(b['bbox'][2],t.bbox[2])-max(b['bbox'][0],t.bbox[0])>20 for b in d['blocks'])
                    populated=sum(bool(str(cell or '').strip()) for row in rows for cell in row)
                    if t.row_count >= 1 and t.col_count >= 2 and populated>=2 and not graphic_caption:
                        table_rects.append(tuple(t.bbox))
                        tables.append({"page": pno + 1, "bbox":list(t.bbox), "rows": rows})
            except Exception as exc:
                report.setdefault("extraction_warnings", []).append({"page": pno+1, "reason": "table_detection_failed", "error": str(exc)})
            numeric_regions,previous_numeric_tables=numeric_table_regions(d,previous_numeric_tables,H)
            for rect,rows in numeric_regions:
                if any(_overlap(rect,t)>.8 for t in table_rects):
                    continue
                table_rects.append(rect)
                tables.append({'page':pno+1,'bbox':list(rect),'rows':rows,'method':'aligned_numeric_cells'})
        else:
            previous_numeric_tables=[]

        # --- Stage 2: build blocks. PyMuPDF sometimes glues a caption/heading to the
        #     paragraph below it, so we split a block whenever font size or boldness changes.
        page_blocks: list[Block] = []
        for bno, b in enumerate(d["blocks"]):
            if b["type"] != 0:
                continue
            infos = []                                   # one entry per non-empty line
            for l in b["lines"]:
                spans = _line_spans(l, cfg.strip_superscript_citations)
                txt = "".join(s["text"] for s in spans).strip()
                n = sum(len(s["text"].strip()) for s in spans)
                if not txt or n == 0:
                    continue
                prose_spans = [s for s in spans if len(re.findall(r'[A-Za-z]',s['text']))>=2]
                sz = max(s['size'] for s in (prose_spans or spans))
                bd = sum(len(s["text"].strip()) for s in spans if _is_bold(s)) / n
                fb = next((_is_bold(s) for s in spans if s["text"].strip()), False)
                dominant = max(spans, key=lambda s: len(s['text']))
                infos.append({"t": txt, "sz": sz, "bold": bd > 0.6, "bd": bd, "fb": fb, "n": n,
                              "font": dominant['font'], "color": dominant['color'],
                              "bb": l["bbox"], "rot": abs(l["dir"][0]) < 0.9})
            groups, cur_g = [], []
            for li in infos:
                # Split only actual typographic transitions (headings/captions),
                # not italic lead-ins, isolated bold words, or numeric citations.
                if cur_g and (abs(li["sz"] - cur_g[-1]["sz"]) > 1.0
                              or li['color'] != cur_g[-1]['color']
                              or (li['bd'] >= .9) != (cur_g[-1]['bd'] >= .9)):
                    groups.append(cur_g)
                    cur_g = []
                cur_g.append(li)
            if cur_g:
                groups.append(cur_g)

            for gno, g in enumerate(groups):
                total = sum(x["n"] for x in g)
                bb = (min(x["bb"][0] for x in g), min(x["bb"][1] for x in g),
                      max(x["bb"][2] for x in g), max(x["bb"][3] for x in g))
                bfrac = sum(x["bd"] * x["n"] for x in g) / total
                blk = Block(pno, bb, bb[1] / H, bb[3] / H, [x["t"] for x in g],
                            sum(x["sz"] * x["n"] for x in g) / total,
                            bfrac > 0.6, g[0]["fb"], total, bold_frac=bfrac)
                blk.source_id = f"p{pno+1}-b{bno}-g{gno}"
                blk.font, blk.color = g[0]['font'], g[0]['color']
                for ri, r in enumerate(sidebars):
                    if _overlap(bb, r) > .9:
                        blk.region = f"box-p{pno+1}-{ri}"
                        break
                if all(x["rot"] for x in g) and cfg.drop_rotated_text:
                    blk.kind, blk.reason = "dropped", "rotated_text"
                elif any(_overlap(bb, t) >= cfg.table_overlap for t in table_rects):
                    blk.kind, blk.reason = "dropped", "table"
                page_blocks.append(blk)

        # --- figure labels: only short, unpunctuated text inside a *real* figure.
        #     A "figure" that contains lots of text is a shaded text box -> keep its text.
        # Figure labels often sit outside the actual drawing or tiled image.
        # Associate the nearby caption with the graphic to recover its panel area.
        page_fonts=Counter()
        for x in page_blocks:
            if x.region=='main' and x.n_words>25:
                page_fonts[round(x.size*2)/2]+=x.n_chars
        local_body=page_fonts.most_common(1)[0][0] if page_fonts else 10
        for caption in page_blocks:
            if not re.match(r'^FIG(?:URE)?\.?\s*\d',caption.text.strip(),re.I):
                continue
            nearby=[r for r,_ in fig_rects if -15<=caption.bbox[1]-r[3]<100
                    and min(r[2],caption.bbox[2])-max(r[0],caption.bbox[0])>20]
            if nearby:
                top=min(r[1] for r in nearby)-26
                panel=(caption.bbox[0]-3,top,caption.bbox[2]+3,caption.bbox[1])
                for x in page_blocks:
                    if (x.region=='main' and x.kind=='text' and x.size<=local_body*.95
                            and x.n_words<=15 and not re.search(r'[.!?:;]$',x.text.strip())
                            and _overlap(x.bbox,panel)>.65):
                        x.kind,x.reason='dropped','figure_panel_label'
        for fr, fkind in fig_rects:
            inside = [x for x in page_blocks if _overlap(x.bbox, fr) >= cfg.figure_overlap]
            if (sum(x.n_words for x in inside) > cfg.figure_max_inner_words
                    and any(x.n_words>25 for x in inside if x.region=='main')):
                continue
            caption_support=any(re.match(r'^FIG(?:URE)?\.?\s*\d',c.text.strip(),re.I)
                and -15<=c.bbox[1]-fr[3]<=80
                and min(c.bbox[2],fr[2])-max(c.bbox[0],fr[0])>20 for c in page_blocks)
            for x in inside:
                if x.kind != "text" or (x.region != 'main' and not caption_support):
                    continue
                if (fkind == "img" and x.n_words <= cfg.figure_label_max_words and x.size<=local_body
                        and not re.search(r'[.!?:;]$',x.text.strip())):
                    x.kind, x.reason = "dropped", "figure_label"
                elif (fkind == "vec" and x.n_words <= cfg.vector_label_max_words
                      and x.size<=local_body and not re.search(r"[.!?:;,]$", x.text.strip())):
                    x.kind, x.reason = "dropped", "figure_label"
        # Panel letters are often just outside the raster boundary.
        for x in page_blocks:
            if x.kind == 'text' and x.region == 'main' and re.fullmatch(r'[A-Z]',x.text.strip()):
                if any(_overlap(x.bbox,(r[0]-16,r[1]-16,r[2]+16,r[3]+16))>.6 for r,_ in fig_rects):
                    x.kind,x.reason = 'dropped','figure_panel_label'
        obstacles=[list(r) for r,_ in fig_rects]+[list(r) for r in table_rects]+[list(r) for r in sidebars]
        report.setdefault('page_obstacles',{})[str(pno+1)]=obstacles
        bars=[b['bbox'] for b in d['blocks'] if b['type']==1 and b['bbox'][2]-b['bbox'][0]>W*.65 and 12<b['bbox'][3]-b['bbox'][1]<65]
        bars += [tuple(dr['rect']) for dr in drawings if dr['rect'].width>W*.65 and 12<dr['rect'].height<65]
        for r in bars:
            if any(x.region=='main' and x.size>=12 and _overlap(x.bbox,r)>.5 for x in page_blocks):
                report.setdefault('layout_separators',{}).setdefault(str(pno+1),[]).append(list(r))
        # A title identifies a whole table, including continued tables. Removing
        # only its caption leaves a misleading stream of unrelated cells.
        for region in {x.region for x in page_blocks if x.region!='main'}:
            members=[x for x in page_blocks if x.region==region]
            tabular=any(re.match(r'^TABLE\s+\d',x.text.strip(),re.I) for x in members)
            # Borderless cells inside a boxed summary: repeated aligned rows
            # with wide gaps between cells. Ordinary wrapped prose has no such gaps.
            words=page.get_text('words',clip=fitz.Rect(min(x.bbox[0] for x in members),min(x.bbox[1] for x in members),max(x.bbox[2] for x in members),max(x.bbox[3] for x in members)))
            rows=[]
            for w in sorted(words,key=lambda w:(round(w[1]/3),w[0])):
                row=next((r for r in rows if abs(r[0][1]-w[1])<3),None)
                if row is None:
                    rows.append([w])
                else:
                    row.append(w)
            multi=sum(any(b[0]-a[2]>25 for a,b in zip(sorted(r,key=lambda w:w[0]),sorted(r,key=lambda w:w[0])[1:])) for r in rows)
            tabular=tabular or (multi>=4 and multi>=len(rows)*.35)
            if tabular:
                for x in members:
                    x.kind,x.reason='dropped','table_region'
        # Chapter mini-contents often lack dot leaders. Detect dense page-number
        # endings in a smaller font, retaining the narrative below the contents.
        font_counts=Counter()
        for x in page_blocks:
            if x.region=='main' and x.n_words>30:
                font_counts[round(x.size*2)/2]+=x.n_chars
        page_body=font_counts.most_common(1)[0][0] if font_counts else 10
        toc=[x for x in page_blocks if x.region=='main' and x.size<page_body*.95
             and any(re.search(r'\s\d{2,4}\s*$',ln) for ln in x.lines)]
        if sum(sum(bool(re.search(r'\s\d{2,4}\s*$',ln)) for ln in x.lines) for x in toc)>=12:
            lo,hi=min(x.bbox[1] for x in toc),max(x.bbox[3] for x in toc)
            for x in page_blocks:
                if x.region=='main' and x.size<page_body*.95 and lo-10<=x.bbox[1]<=hi:
                    x.kind,x.reason='dropped','contents_entry'
        for x in page_blocks:
            if x.kind=='text' and x.size<local_body*.95 and x.lines and all(
                    re.fullmatch(r'\d+(?:\s*[,\u2013-]\s*\d+){2,}\.?',
                                 re.sub(r'^[-*\u2020\u2021\u25a0\s]+','',ln.strip())) for ln in x.lines):
                x.kind,x.reason='dropped','citation_note'
            if x.kind=='text' and x.region=='main' and x.y0n>.87 and x.size<local_body*.95:
                t=re.sub(r'^[*\u2020\u2021\u25a0\s]+','',x.text.strip())
                if re.fullmatch(r'\d+(?:\s*[,\u2013-]\s*\d+){2,}\.?',t):
                    x.kind,x.reason='dropped','citation_note'
                elif not re.match(cfg.caption_regex,x.text.strip(),re.I):
                    x.region=f'footnotes-p{pno+1}'
        blocks.extend(page_blocks)

    report["pages"] = len(doc)
    report["ocr_pages"] = ocr_pages
    return blocks, tables


# --------------------------------------------------------------------------- #
# Stage 4: running headers / footers / page numbers
# --------------------------------------------------------------------------- #
def remove_headers_footers(blocks: list[Block], cfg: Config, n_pages: int):
    def in_margin(b: Block):
        return b.y1n <= cfg.margin_zone or b.y0n >= 1 - cfg.margin_zone

    def norm(t: str):
        t = re.sub(r"\d+", "#", t.lower())
        return re.sub(r"[^\w#]+", " ", t).strip()

    cand = [b for b in blocks if b.kind == "text" and not b.region.startswith('box-')
            and in_margin(b) and len(b.lines) <= cfg.margin_max_lines]
    pages_by_text: dict[str, set] = {}
    for b in cand:
        pages_by_text.setdefault(norm(b.text), set()).add(b.page)

    for b in cand:
        if PAGE_NUM_RE.match(b.text):
            b.kind, b.reason = "dropped", "page_number"
        elif n_pages >= cfg.min_repeat_pages and len(pages_by_text[norm(b.text)]) >= cfg.min_repeat_pages:
            b.kind, b.reason = "dropped", "running_header_footer"


# --------------------------------------------------------------------------- #
# Stage 5: font profile + headings
# --------------------------------------------------------------------------- #
def body_font_size(blocks: list[Block]) -> float:
    c = Counter()
    for b in blocks:
        if b.kind == "text":
            c[round(b.size * 2) / 2] += b.n_chars
    return c.most_common(1)[0][0] if c else 10.0


def remove_contents_and_margin_fragments(blocks,body):
    for b in blocks:
        if (b.kind=='text' and b.region=='main' and b.y1n<.065
                and b.size<body*.95 and len(b.lines)<=2):
            b.kind,b.reason='dropped','margin_furniture'
    for page in {b.page for b in blocks}:
        items=[b for b in blocks if b.page==page and b.region=='main' and b.kind=='text']
        toc=[b for b in items if b.size<body*.95 and any(re.search(r'\s\d{2,4}\s*$',ln) for ln in b.lines)]
        if sum(sum(bool(re.search(r'\s\d{2,4}\s*$',ln)) for ln in b.lines) for b in toc)>=12:
            lo,hi=min(b.bbox[1] for b in toc),max(b.bbox[3] for b in toc)
            for b in items:
                if b.size<body*.95 and lo-10<=b.bbox[1]<=hi:
                    b.kind,b.reason='dropped','contents_entry'


def order_layout(blocks: list[Block], report: dict) -> list[Block]:
    """Recursive whitespace cuts: columns before rows, spanning titles as bands.

    Work on paragraph rectangles, never globally interleave individual lines.
    Independent bordered callouts are processed in separate flows by assemble().
    """
    def gaps(items, axis):
        intervals = sorted((b.bbox[axis], b.bbox[axis+2]) for b in items)
        result, end = [], intervals[0][1]
        for start, stop in intervals[1:]:
            if start-end > (9 if axis == 0 else 2):
                result.append((end, start))
            end = max(end, stop)
        return result

    def walk(items, path='0'):
        if not items:
            return []
        if len(items) == 1:
            items[0].column = path
            return items
        xs = gaps(items, 0)
        if xs:
            a,z = max(xs, key=lambda g: g[1]-g[0])
            cut = (a+z)/2
            return walk([b for b in items if b.bbox[2] <= cut], path+'L') + walk([b for b in items if b.bbox[0] >= cut], path+'R')
        ys = gaps(items, 1)
        if ys:
            a,z = max(ys, key=lambda g: g[1]-g[0])
            cut = (a+z)/2
            # Horizontal bands share their column path, permitting normal joins.
            return walk([b for b in items if b.bbox[3] <= cut], path) + walk([b for b in items if b.bbox[1] >= cut], path)
        ordered = sorted(items, key=lambda b: (b.bbox[1], b.bbox[0]))
        for b in ordered:
            b.column = path
        # Overlaps are observable, not silently labelled as reliable ordering.
        for a,b in zip(ordered,ordered[1:]):
            inline_label=(a.source_id and a.source_id.rsplit('-g',1)[0]==b.source_id.rsplit('-g',1)[0]
                          and abs(a.bbox[1]-b.bbox[1])<2 and a.text.rstrip().endswith(':'))
            if a.kind!='placeholder' and b.kind!='placeholder' and not inline_label and _overlap(a.bbox,b.bbox) > .15:
                report.setdefault('layout_warnings', []).append({'page':b.page+1,'reason':'overlapping_text','source_ids':[a.source_id,b.source_id]})
        return ordered

    result = []
    for page in sorted({b.page for b in blocks}):
        pageblocks = [b for b in blocks if b.page == page]
        for region in sorted({b.region for b in pageblocks}, key=lambda r:r != 'main'):
            items = [b for b in pageblocks if b.region == region and b.kind != 'dropped']
            if region == 'main':
                for r in report.get('layout_separators',{}).get(str(page+1),[]):
                    items.append(Block(page,tuple(r),0,0,[],10,False,False,0,kind='placeholder'))
                # A callout occupies space even though its text is a separate flow.
                # Preserve that space during column segmentation.
                for box in {b.region for b in pageblocks if b.region!='main'}:
                    contents=[b for b in pageblocks if b.region==box and b.kind!='dropped']
                    if not contents:
                        continue
                    rect=(min(b.bbox[0] for b in contents),min(b.bbox[1] for b in contents),
                          max(b.bbox[2] for b in contents),max(b.bbox[3] for b in contents))
                    items.append(Block(page,rect,0,0,[],10,False,False,0,kind='placeholder'))
            ordered=[b for b in walk(items) if b.kind!='placeholder']
            if region == 'main':
                # A figure can force a paragraph into the next column before
                # later text resumes below the figure in the first column.
                # Put that continuation beside its source before assembling
                # sections, or it may attach to a later heading instead.
                obstacles = report.get('page_obstacles', {}).get(str(page + 1), [])
                for continuation in list(ordered):
                    if (continuation.kind != 'text' or continuation.y0n >= .4
                            or not text_boundaries.begins_with_continuation(continuation.text)):
                        continue
                    index = ordered.index(continuation)
                    for source_index in range(index - 2, -1, -1):
                        source = ordered[source_index]
                        if (source.kind != 'text' or source.n_words < 12
                                or text_boundaries.is_boundary_label(source.text)
                                or source.column == continuation.column
                                or source.bbox[2] + 15 >= continuation.bbox[0]
                                or re.search(r'[.!?][)\u201d\u2019"]*$', source.text)):
                            continue
                        figure_below = any(
                            0 <= rect[1] - source.bbox[3] <= 30
                            and rect[3] - rect[1] > 100
                            and min(rect[2], source.bbox[2]) - max(rect[0], source.bbox[0]) > 30
                            for rect in obstacles
                        )
                        if figure_below:
                            ordered.pop(index)
                            ordered.insert(source_index + 1, continuation)
                            break
            override = report.get('_page_order_overrides',{}).get(str(page+1)) if region=='main' else None
            if override is not None:
                by_id={b.source_id:b for b in ordered}
                if len(override)!=len(by_id) or set(override)!=set(by_id):
                    raise ValueError(f'Page {page+1} order override must list every retained main source_id exactly once')
                ordered=[by_id[i] for i in override]
                report.setdefault('manual_order_pages',[]).append(page+1)
            result.extend(ordered)
        result.extend(b for b in pageblocks if b.kind == 'dropped')
    return result


def _outline_index(doc):
    idx: dict[int, list] = {}
    try:
        for lvl, title, pg in doc.get_toc():
            if pg >= 1:
                idx.setdefault(pg - 1, []).append((_alnum(title), min(lvl, 4)))
    except Exception:
        pass
    return idx


def detect_headings(blocks: list[Block], body: float, cfg: Config, doc):
    outline = _outline_index(doc) if cfg.use_pdf_outline else {}
    cap_re = re.compile(cfg.caption_regex, re.I)

    def outline_level(b: Block):
        key = _alnum(b.text)
        if len(key) < 3:
            return 0
        for pg in (b.page, b.page - 1):
            for t, lvl in outline.get(pg, []):
                short_title=(b.n_words<=cfg.max_heading_words and len(b.lines)<=cfg.max_heading_lines
                             and .75<=len(key)/max(len(t),1)<=1.25)
                if t and (key == t or (short_title and len(key)>=6 and (t.startswith(key) or key.startswith(t)))):
                    return lvl
        return 0

    def is_candidate(b: Block) -> bool:
        t = b.text.strip()
        if any(re.fullmatch(p,t.rstrip(':'),re.I) for p in cfg.callout_labels):
            return True
        if BULLET_RE.match(t) or t.endswith('-') or t.count('(')>t.count(')'):
            return False
        if re.match(r'^BOX\s+\d',t,re.I) and b.region != 'main':
            return True
        big = b.size >= body * cfg.heading_size_ratio
        if (len(b.lines) > max(cfg.max_heading_lines,6 if big else 0)
                or b.n_words > max(cfg.max_heading_words,30 if big else 0)
                or len(t) > 300 or not re.search(r"[A-Za-z]", t) or len(t) < 3):
            return False
        if cap_re.match(t) or t.endswith((",", ";")):
            return False
        # bold-only headings must be (almost) fully bold and a single phrase; a bold lead-in
        # followed by regular text ("Median nerve. The most common ...") is a run-in, not a heading
        bold_short = (b.bold_frac >= 0.9 and b.size >= body * 0.95 and b.n_words <= 12
                      and not t.endswith(".") and not re.search(r"[.!?]\s+[A-Z]", t))
        chapter_word = re.match(r"^(chapter|part|unit|section)\s+(\d+|[ivxlc]+)\b", t, re.I) \
            and b.size >= body
        return big or bold_short or bool(chapter_word)

    cands = []
    prev_block = None
    for b in blocks:
        if b.kind != "text":
            continue
        ol = outline_level(b)
        inline_continuation = (prev_block is not None and b.source_id and b.page==prev_block.page
            and b.region==prev_block.region and b.source_id.rsplit('-g',1)[0]==prev_block.source_id.rsplit('-g',1)[0]
            and not re.search(r'[.!?:;][)\u201d\u2019"]*$',prev_block.text.strip())
            and -1<=b.bbox[1]-prev_block.bbox[3]<=body*1.8)
        if ol or (is_candidate(b) and not inline_continuation):
            cands.append((b, ol))
        prev_block = b

    callout_res = [re.compile(p,re.I) for p in cfg.callout_labels]
    size_levels = sorted({round(b.size*2)/2 for b,_ in cands
                          if b.region=='main' and not any(r.fullmatch(b.text.strip()) for r in callout_res)},reverse=True)
    for b, ol in cands:
        if ol:
            lvl = ol
        elif round(b.size*2)/2 in size_levels:
            lvl = size_levels.index(round(b.size*2)/2) + 1
        else:
            lvl = len(size_levels)+1
        b.kind, b.level = "heading", lvl

    # merge consecutive same-level headings on the same page ("CHAPTER 3" + "Shoulder complex")
    prev = None
    for b in blocks:
        if b.kind == "heading":
            if (prev and prev.page == b.page and prev.level == b.level
                    and prev.region == b.region and prev.column == b.column
                    and re.match(r'^(CHAPTER|PART|UNIT|SECTION)\s+[\dIVX]+$', prev.text.strip(), re.I)
                    and 0 <= b.bbox[1]-prev.bbox[3] < body*3):
                prev.lines += b.lines
                b.kind, b.reason = "dropped", "merged_heading"
                continue
            prev = b
        elif b.kind == "text":
            prev = None


# --------------------------------------------------------------------------- #
# Stage 6: block-level noise
# --------------------------------------------------------------------------- #
def drop_block_noise(blocks: list[Block], body: float, cfg: Config):
    cap_re = re.compile(cfg.caption_regex, re.I)
    noise_res = [re.compile(p, re.I) for p in cfg.noise_block_patterns]

    for b in blocks:
        if b.kind != "text":
            continue
        t = b.text.strip()
        if cfg.drop_footnotes and b.region.startswith('footnotes-'):
            b.kind,b.reason='dropped','footnote'
            continue
        if re.match(r'^(?:students?|learners?)\s+(?:should|will|must)\s+(?:now\s+)?be able to\b',t,re.I):
            b.kind,b.reason='dropped','student_learning_objective'
            continue

        if re.fullmatch(r'C\s*H\s*A\s*P\s*T\s*E\s*R',t,re.I):
            b.kind,b.reason='dropped','chapter_decoration'
            continue
        if re.fullmatch(r'\d{1,3}',t) and b.size>body*2 and b.y0n<.3:
            b.kind,b.reason='dropped','chapter_decoration'
            continue
        if (b.y0n<.3 and ',' in t and b.n_words<30
                and re.search(r'\b(?:PT|DPT|PhD|MD|MS)\b',t)):
            b.kind,b.reason='dropped','author_byline'
            continue

        # captions: "Figure 3.2 ..." - but NOT body sentences like "Figure 3 shows that ..."
        if cfg.drop_captions and cap_re.match(t):
            if b.size <= body * 0.97 or b.first_bold:
                b.kind, b.reason = "dropped", "caption"
                continue

        # copyright / credit lines (only short blocks)
        if cfg.drop_copyright_lines and len(b.lines) <= 3 and any(r.search(t) for r in noise_res):
            b.kind, b.reason = "dropped", "copyright_or_credit"
            continue

        # question lists: most lines end with '?', or MCQ options (a) b) c) d)
        if cfg.drop_question_lists and len(b.lines) >= 2:
            q = sum(1 for l in b.lines if l.rstrip().endswith("?")) / len(b.lines)
            mcq = sum(1 for l in b.lines if re.match(r"^\(?[A-Da-d][.)]\s", l)) / len(b.lines)
            # Clinical interview questions are useful content. Remove practice
            # questions by their section context, not by question-mark density.
            if len(b.lines)>=3 and mcq >= 0.6:
                b.kind, b.reason = "dropped", "question_list"


# --------------------------------------------------------------------------- #
# Stage 8-9: text reconstruction + normalisation
# --------------------------------------------------------------------------- #
CIT_NUM = re.compile(r"\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]")
CIT_AY = re.compile(
    r"\((?:see |e\.g\.,? |cf\. )?[A-Z][A-Za-z\u00c0-\u017f'\-]+(?: et al\.?)?(?:,? (?:and|&) [A-Z][A-Za-z\u00c0-\u017f'\-]+)?"
    r",? \d{4}[a-z]?(?:\s*[;,]\s*[A-Z][^()]*?\d{4}[a-z]?)*\)")


FIG_REF = re.compile(
    r"\(\s*(?:see |refer to |also see |compare )?(?:Figs?\.?|Figures?|Tables?|Boxes|Box|Chapters?|Videos?)(?=[\s\d])[^()]{0,90}\)", re.I)
VIDEO_REF = re.compile(r"\bVIDEOS?(?:\s*\d+(?:\.\d+)?(?:\s*(?:and|,|\u2013|-)\s*\d+(?:\.\d+)?)*)?\b")


def normalise(text: str, cfg: Config) -> str:
    text = unicodedata.normalize("NFKC", text)           # ligatures; keeps the degree sign
    text = re.sub(r"[\u00ad\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    if cfg.strip_figure_refs:
        text = FIG_REF.sub("", text)
        text = VIDEO_REF.sub("", text)
    if cfg.strip_numeric_citations:
        text = CIT_NUM.sub("", text)
    if cfg.strip_author_year_citations:
        text = CIT_AY.sub("", text)
    if cfg.strip_footnote_markers:
        text = re.sub(r'(?<=[.!?])[*\u2020\u2021]+(?=\s|$)', '', text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def block_to_units(b: Block, cfg=None) -> list[str]:
    """Split a block into paragraph units (bullets / numbered steps start new units)."""
    units, cur = [], []
    for ln in b.lines:
        if TOC_LEADER_RE.search(ln):
            continue
        starts_item = bool(BULLET_RE.match(ln) or NUMBERED_RE.match(ln))
        if starts_item and cur:
            units.append(join_lines(cur, cfg.compound_prefixes if cfg else None))
            cur = []
        cur.append(BULLET_RE.sub("- ", ln) if BULLET_RE.match(ln) else ln)
    if cur:
        units.append(join_lines(cur, cfg.compound_prefixes if cfg else None))
    return units


def is_junk(p: str, cfg: Config) -> bool:
    letters = sum(c.isalpha() for c in p)
    if letters / max(len(p), 1) < cfg.min_alpha_ratio:
        return True
    is_item = p.startswith("- ") or bool(NUMBERED_RE.match(p))
    words = len(p.split())
    if not is_item and words < cfg.min_paragraph_words and not re.search(r"[.:!?]$", p):
        return True
    return False


# --------------------------------------------------------------------------- #
# Stage 7 + assembly
# --------------------------------------------------------------------------- #
def assemble(blocks: list[Block], cfg: Config, report: dict, book: str):
    """Build heading nodes and auditable paragraphs without language-only joins."""
    skip_res = [re.compile(p,re.I) for p in cfg.skip_heading_patterns]
    if cfg.drop_chapter_summaries:
        skip_res.append(re.compile(r'^(?:(?:chapter|section)\s+)?summary$',re.I))
    callouts = [re.compile(p,re.I) for p in cfg.callout_labels]
    nodes, records, stack, anchor = [], [], [], {}
    skipped = []
    skip_level = None
    current = None
    previous = None
    pending_label = None

    def key(t):
        # Only strip actual numeric prefixes, never leading Roman letters in words.
        return re.sub(r'^(?:(?:chapter|unit|part|section)\s+)?(?:\d+(?:\.\d+)*|[IVX]+)[.):\-\s]+', '', t, flags=re.I).strip()

    def node(title, level, b, parent=None, kind='heading'):
        nid = f'{book}:h{len(nodes):05d}'
        ancestry = next((n['heading_path'] for n in nodes if n['node_id']==parent), [])
        n = {'node_id':nid,'parent_id':parent,'child_ids':[], 'title':title,
             'level':len(ancestry)+1,'detected_level':level,'kind':kind,
             'heading_path':ancestry+[title], 'page_start':b.page+1,'page_end':b.page+1,
             'source_id':b.source_id}
        nodes.append(n)
        if parent:
            next(n for n in nodes if n['node_id']==parent)['child_ids'].append(nid)
        return n

    def record(b):
        parent = stack[-1][1] if stack else None
        path = next((n['heading_path'] for n in nodes if n['node_id']==parent), [])
        r = {'book':book,'section_id':f'{book}:s{len(records):05d}',
             'heading_id':parent,'parent_id':next((n['parent_id'] for n in nodes if n['node_id']==parent),None),
             'heading_path':path,'heading':path[-1] if path else '',
             'region':b.region,'paragraphs':[], 'page_start':b.page+1,'page_end':b.page+1}
        records.append(r)
        return r

    def provenance(b):
        return {'source_id':b.source_id,'page':b.page+1,'bbox':list(b.bbox),'column':b.column,'region':b.region}

    def can_continue(a,b,prev,p):
        if a is None or a.region != b.region or abs(a.size-b.size)>1.0:
            return False
        if text_boundaries.is_boundary_label(p):
            return False
        if a.page != b.page and prev.lstrip().startswith('- '):
            return False
        if (re.search(r'[.!?][)\u201d\u2019"]*$',prev) or BULLET_RE.match(p) or NUMBERED_RE.match(p)):
            return False
        # A continuation can begin with a number, e.g. "30 to 60 seconds".
        same_source = (a.page==b.page and a.source_id and
                       a.source_id.rsplit('-g',1)[0]==b.source_id.rsplit('-g',1)[0])
        function_word = re.search(r'\b(the|a|an|and|or|of|to|with|for|in|by|as|at|from|than|that|is|are)$',prev,re.I)
        open_parenthesis_at_column_edge = (
            a.page == b.page and a.column != b.column
            and a.y1n > .8 and b.y0n < .5
            and prev.count('(') > prev.count(')')
        )
        if not (p.lstrip('(')[:1].islower() or p[:1].isdigit() or prev.endswith('-')
                or same_source or function_word or open_parenthesis_at_column_edge):
            return False
        if same_source and abs(a.bbox[1]-b.bbox[1])<2 and prev.rstrip().endswith(':'):
            return True
        if a.page == b.page and a.column == b.column:
            gap = b.bbox[1]-a.bbox[3]
            overlap = min(a.bbox[2],b.bbox[2])-max(a.bbox[0],b.bbox[0])
            aligned=overlap > min(a.bbox[2]-a.bbox[0],b.bbox[2]-b.bbox[0])*.4
            if -1 <= gap <= max(a.size,b.size)*1.8 and aligned:
                return True
            if gap>0 and aligned:
                gaprect=(max(a.bbox[0],b.bbox[0]),a.bbox[3],min(a.bbox[2],b.bbox[2]),b.bbox[1])
                return any(_overlap(gaprect,r)>.25 for r in report.get('page_obstacles',{}).get(str(b.page+1),[]))
            return False
        # Only adjacent main-flow columns/pages, at a verified column/page edge.
        if b.page-a.page not in (0,1):
            return False
        if a.page == b.page:
            return a.column != b.column and b.bbox[0]>a.bbox[0]+20
        # Here a is the last retained narrative block on the preceding page
        # and b is the first on the next. A figure may occupy either page edge.
        return b.page == a.page+1

    def consume(flow, initial_stack=None):
        nonlocal stack,current,previous,pending_label,skip_level
        stack = list(initial_stack or [])
        floor = len(stack)
        current,previous,pending_label,skip_level = None,None,None,None
        for bi,b in enumerate(flow):
            if b.kind == 'dropped':
                continue
            if b.region == 'main':
                anchor[b.source_id] = list(stack)
            if b.kind == 'heading':
                title = normalise(join_lines(b.lines),cfg)
                if any(r.fullmatch(key(title)) for r in callouts):
                    pending_label = title.upper()
                    continue
                current,previous,pending_label = None,None,None
                if skip_level is not None and b.level <= skip_level:
                    skip_level = None
                if skip_level is not None:
                    b.kind,b.reason = 'dropped','skipped_section'
                    continue
                practice_case=False
                if cfg.drop_student_case_studies and re.fullmatch(r'case stud(?:y|ies)',key(title),re.I):
                    upcoming=[]
                    for candidate in flow[bi+1:]:
                        if candidate.kind=='heading':
                            break
                        if candidate.kind!='dropped':
                            upcoming.append(candidate.text)
                        if len(upcoming)>=12:
                            break
                    practice_case=bool(re.search(
                        r'(?:^|[.!?]\s+|[\u25a0\u2022]\s*|\b\d{1,2}\.\s+)'
                        r'(?:describe|outline|explain|think about|design|develop|how would you|what would you)\b',
                        ' '.join(upcoming),re.I))
                if practice_case or any(r.match(key(title)) for r in skip_res):
                    skip_level = b.level
                    skipped.append({'heading':title,'page':b.page+1})
                    b.kind,b.reason = 'dropped','skipped_section'
                    continue
                while len(stack)>floor and stack[-1][0] >= b.level:
                    stack.pop()
                n = node(title,b.level,b,stack[-1][1] if stack else None)
                stack.append((b.level,n['node_id']))
                if b.region=='main':
                    anchor[b.source_id]=list(stack)
                continue
            if skip_level is not None:
                b.kind,b.reason = 'dropped','skipped_section'
                continue
            for u in block_to_units(b,cfg):
                p = normalise(u,cfg)
                if not p:
                    continue
                if text_boundaries.is_boundary_label(p):
                    b.kind, b.reason = 'dropped', 'box_number_label'
                    continue
                if current is None:
                    current = record(b)
                paras = current['paragraphs']
                if paras and pending_label is None and can_continue(previous,b,paras[-1]['text'],p):
                    paras[-1]['text'] = normalise(join_lines([paras[-1]['text'],p],cfg.compound_prefixes),cfg)
                    paras[-1]['sources'].append(provenance(b))
                    paras[-1]['join_evidence'].append('adjacent_geometry')
                else:
                    if pending_label:
                        p = pending_label+': '+p
                        pending_label = None
                    # Retain short fragments for review instead of deleting dose values.
                    paras.append({'paragraph_id':f"{current['section_id']}:p{len(paras):04d}",
                                  'text':p,'sources':[provenance(b)],'join_evidence':[]})
                current['page_end'] = b.page+1
                previous = b

    # Sidebars never interrupt an unfinished sentence in the main narrative.
    consume([b for b in blocks if b.region=='main'])
    for region in dict.fromkeys(b.region for b in blocks if b.region!='main'):
        flow = [b for b in blocks if b.region==region and b.kind!='dropped']
        if not flow:
            continue
        first = flow[0]
        preceding = [b for b in blocks if b.region=='main' and b.page==first.page
                     and b.bbox[1] <= first.bbox[1] and b.source_id in anchor]
        near = min(preceding,key=lambda b:abs(b.bbox[0]-first.bbox[0])+abs(b.bbox[1]-first.bbox[1])*.1) if preceding else None
        base = anchor.get(near.source_id,[]) if near else []
        titleblock = next((b for b in flow if b.kind=='heading'),None)
        if titleblock is None:
            label='Footnotes' if region.startswith('footnotes-') else 'Callout'
            n = node(label,99,first,base[-1][1] if base else None,'callout')
            base = base+[(99,n['node_id'])]
        consume(flow,base)

    # Page-spanning prose can be separated by figure or table headings. Move the
    # first narrative paragraph on the next page back to its source section when
    # both text boundaries show a continuation. Never invent absent PDF text.
    for r in records:
        if r['region'] != 'main' or not r['paragraphs']:
            continue
        for p in list(r['paragraphs']):
            if not text_boundaries.has_dangling_prose_end(p['text']):
                continue
            later = r['paragraphs'][r['paragraphs'].index(p) + 1:]
            if any(not text_boundaries.is_boundary_label(item['text']) for item in later):
                continue
            end_page = p['sources'][-1]['page']
            candidates = [
                (other, candidate) for other in records if other['region'] == 'main'
                for candidate in other['paragraphs']
                if candidate is not p and candidate['sources'][0]['page'] == end_page + 1
                and text_boundaries.begins_with_continuation(candidate['text'])
            ]
            # A word split at the page edge resumes in the first prose column.
            # Record order can instead place unrelated right-column prose first.
            if p['text'].endswith('-'):
                candidates.sort(key=lambda item: (
                    item[1]['sources'][0]['bbox'][0], item[1]['sources'][0]['bbox'][1]
                ))
            following = candidates[0] if candidates else None
            if following:
                other, continuation = following
                p['text'] = normalise(join_lines(
                    [p['text'], continuation['text']], cfg.compound_prefixes), cfg)
                p['sources'].extend(continuation['sources'])
                p['join_evidence'].extend(continuation['join_evidence'])
                p['join_evidence'].append('cross_page_continuation')
                other['paragraphs'].remove(continuation)
            elif end_page == report['pages']:
                complete = text_boundaries.completed_sentence_prefix(p['text'])
                report.setdefault('incomplete_fragments_dropped', []).append({
                    'paragraph_id': p['paragraph_id'],
                    'source_ids': [source['source_id'] for source in p['sources']],
                    'text': p['text'][len(complete):].strip(),
                })
                if complete:
                    p['text'] = complete
                else:
                    r['paragraphs'].remove(p)

    # Sources, rather than old section boundaries, determine page spans after repair.
    for r in records:
        pages = [source['page'] for p in r['paragraphs'] for source in p['sources']]
        if pages:
            r['page_start'], r['page_end'] = min(pages), max(pages)

    # Repeated factual instructions may be intentional across phases. Retain
    # them, but do not label their source quality as automatically verified.
    def sentences(text):
        return [re.sub(r'\s+',' ',s).strip() for s in re.split(r'(?<=[.!?])\s+',text)]
    repeats=Counter(s.casefold() for r in records for p in r['paragraphs']
                    for s in sentences(p['text']) if len(s.split())>=12)
    issues=[]
    for r in records:
        for p in r['paragraphs']:
            t = p['text']
            p['quality_flags'] = []
            if t[:1].islower():
                p['quality_flags'].append('lowercase_start')
            if not re.search(r'[.!?:;)\u201d\u2019"\u00b0%]$',t):
                p['quality_flags'].append('open_ending')
            if (re.search(r'(?<=[A-Za-z.])\d+(?:,\d+){2,}',t)
                    or re.fullmatch(r'[-\s]*\d+(?:\s*[,\u2013-]\s*\d+){2,}',t)):
                p['quality_flags'].append('possible_residual_citation')
            if cfg.flag_repeated_sentences and any(len(s.split())>=12 and repeats[s.casefold()]>1 for s in sentences(t)):
                p['quality_flags'].append('repeated_sentence')
            if cfg.flag_figure_dependencies and cfg.drop_figure_labels and re.search(
                    r'\bfig(?:ure)?s?\.?\s+\d[\d.\-]*\s+(?:shows?|illustrates?|depicts?|demonstrates?)\b',t,re.I):
                p['quality_flags'].append('figure_dependency')
            if p['quality_flags']:
                issues.append({'paragraph_id':p['paragraph_id'],'flags':p['quality_flags'],
                               'text':t,'sources':p['sources']})
        r['text']='\n\n'.join(p['text'] for p in r['paragraphs'])
        r['n_chars']=len(r['text'])
        r['ready_for_embedding'] = all(not p['quality_flags'] for p in r['paragraphs'])
        for n in nodes:
            if n['node_id']==r['heading_id']:
                n['page_end']=max(n['page_end'],r['page_end'])
    for n in reversed(nodes):
        if n['parent_id']:
            parent=next(x for x in nodes if x['node_id']==n['parent_id'])
            parent['page_end']=max(parent['page_end'],n['page_end'])
    # Keep content-bearing nodes and all their ancestors. Discard empty generated
    # callout/footnote nodes, without removing useful heading-only parents.
    by_id={n['node_id']:n for n in nodes}
    live=set()
    for r in records:
        nid=r['heading_id'] if r['paragraphs'] else None
        while nid and nid not in live:
            live.add(nid)
            nid=by_id[nid]['parent_id']
    nodes=[n for n in nodes if n['node_id'] in live]
    for n in nodes:
        n['child_ids']=[cid for cid in n['child_ids'] if cid in live]
    report['skipped_sections']=skipped
    report['_hierarchy']=nodes
    report['_issues']=issues
    report['quality_flag_counts']=dict(Counter(flag for issue in issues for flag in issue['flags']))
    return [r for r in records if r['paragraphs']]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_book(path: Path, out_root: Path, cfg: Config, debug: bool = False) -> dict:
    book = path.stem
    out = out_root / re.sub(r"[^\w\-]+", "_", book)
    out.mkdir(parents=True, exist_ok=True)
    existing=out/'report.json'
    if existing.exists():
        previous=json.loads(existing.read_text(encoding='utf-8'))
        if Path(previous['source']).resolve()!=path.resolve():
            raise ValueError(f'Output name collision for {path.name}; use a separate output root or unique filenames')
    report: dict = {"book": book, "source": str(path), "output_dir":str(out.resolve()), "_removed": Counter()}
    report['pipeline_version']=PIPELINE_VERSION

    doc = fitz.open(path)
    if doc.needs_pass:
        raise RuntimeError(f"{path.name} is password protected")

    blocks, tables = extract_pages(doc, cfg, report)                 # stages 1-3
    if debug:
        with open(out/'extraction.json','w',encoding='utf-8') as f:
            json.dump({'blocks':[asdict(b) for b in blocks], 'tables':tables,
                       'report':{k:v for k,v in report.items() if k!='_removed'}},f,ensure_ascii=False)
    remove_headers_footers(blocks, cfg, len(doc))                    # stage 4
    body = body_font_size(blocks)                                    # stage 5
    remove_contents_and_margin_fragments(blocks,body)
    drop_block_noise(blocks, body, cfg)
    report['_page_order_overrides'] = cfg.page_order_overrides
    blocks = order_layout(blocks, report)
    report.pop('_page_order_overrides',None)
    detect_headings(blocks, body, cfg, doc)
    records = assemble(blocks, cfg, report, book)                    # stages 7-9
    # Only join literal line-end hyphens in join_lines(). Vocabulary guessing can
    # silently corrupt medical compounds and is deliberately not used.
    report["hyphen_breaks_repaired"] = 0
    paras = [p for r in records for p in r["text"].split("\n\n")]
    report["paragraphs_total"] = len(paras)
    report["paragraphs_ending_mid_sentence"] = sum(
        1 for p in paras if not re.search(r'[.!?:;)"\u201d\u2019\u00b0%]$', p.strip()))

    # ---- export
    with open(out / "sections.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nodes = report.pop('_hierarchy')
    issues = report.pop('_issues')
    with open(out / 'hierarchy.json', 'w', encoding='utf-8') as f:
        json.dump({'book':book,'roots':[n['node_id'] for n in nodes if n['parent_id'] is None],
                   'nodes':nodes},f,ensure_ascii=False,indent=2)
    with open(out / 'review.jsonl','w',encoding='utf-8') as f:
        for issue in issues:
            f.write(json.dumps(issue,ensure_ascii=False)+'\n')
    with open(out / 'layout.jsonl','w',encoding='utf-8') as f:
        for b in blocks:
            f.write(json.dumps(asdict(b),ensure_ascii=False)+'\n')
    report['review_items'] = len(issues)
    report['quality_status'] = (
        'needs_review'
        if issues or report.get('layout_warnings') or report.get('extraction_warnings')
        or report.get('incomplete_fragments_dropped')
        else 'checks_passed'
    )
    report['hierarchy_method'] = 'PDF outline plus typography heuristics; inferred relationships require review'
    report['lowercase_paragraph_starts'] = sum('lowercase_start' in i['flags'] for i in issues)
    report['possible_residual_citations'] = sum('possible_residual_citation' in i['flags'] for i in issues)
    report['paragraphs_with_open_endings'] = report['paragraphs_ending_mid_sentence']
    report['metric_note'] = 'Open endings are punctuation-based review signals, not a measured sentence-error rate. Lists and labels may be intentional.'

    with open(out / "clean.md", "w", encoding="utf-8") as f:
        last = []
        for r in records:
            for depth, h in enumerate(r["heading_path"]):
                if depth >= len(last) or last[depth] != h:
                    f.write("#" * (depth + 1) + " " + h + "\n\n")
            last = r["heading_path"]
            f.write(r["text"] + "\n\n")

    if cfg.keep_tables_separately:
        with open(out / "tables.jsonl", "w", encoding="utf-8") as f:
            for t in tables:
                f.write(json.dumps({"book": book, **t}, ensure_ascii=False) + "\n")
    elif (out/'tables.jsonl').exists():
        (out/'tables.jsonl').unlink()

    for b in blocks:
        if b.kind == "dropped":
            report["_removed"][b.reason] += b.n_chars
    removed = report.pop("_removed")
    report.update({
        "body_font_size": body,
        "headings_detected": sum(1 for b in blocks if b.kind == "heading"),
        "sections_kept": len(records),
        "chars_kept": sum(r["n_chars"] for r in records),
        "tables_extracted": len(tables),
        "chars_removed_by_reason": dict(removed.most_common()),
    })
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),encoding='utf-8')

    if debug:
        with open(out / "removed.jsonl", "w", encoding="utf-8") as f:
            for b in blocks:
                if b.kind == "dropped":
                    f.write(json.dumps({"page": b.page + 1, "reason": b.reason,
                                        "text": b.text[:300]}, ensure_ascii=False) + "\n")
    else:
        for name in ('removed.jsonl','extraction.json'):
            if (out/name).exists():
                (out/name).unlink()
    log.info("%s: %d pages -> %d sections, %d chars kept", book, report["pages"],
             len(records), report["chars_kept"])
    doc.close()
    if cfg.fail_on_review and report['quality_status'] != 'checks_passed':
        raise RuntimeError(f"{book}: extraction saved but requires review; see review.jsonl and report.json")
    return report


def main():
    ap = argparse.ArgumentParser(description="Generic book preprocessing for RAG")
    ap.add_argument("input", help="a book file or a folder of books (.pdf/.epub)")
    ap.add_argument("-o", "--out", default="processed")
    ap.add_argument("--config", help="JSON file overriding Config fields")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--debug", action="store_true", help="write removed.jsonl for tuning")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = Config.load(args.config)
    if args.no_ocr:
        cfg.ocr = False

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*")
                                               if p.suffix.lower() in {".pdf", ".epub"})
    failed = 0
    if not files:
        ap.error('No PDF or EPUB files found')
    for p in files:
        try:
            process_book(p, Path(args.out), cfg, args.debug)
        except Exception as e:
            log.error("FAILED %s: %s", p.name, e)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
