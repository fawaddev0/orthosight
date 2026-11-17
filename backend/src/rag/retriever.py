"""Retrieve complete book sections for supported conservative diagnoses."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from rag.cloudflare_ai import RERANKER_MODEL, rerank as cloudflare_rerank
from rag.embedding_generator import (
    ACTIVE_ALIAS,
    ACTIVE_COLLECTION,
    BGE_QUERY_PREFIX,
    EMBED_MODEL,
    CloudflareEmbeddingModel,
    IndexedSection,
)

logger = logging.getLogger(__name__)
RERANK_MODEL = RERANKER_MODEL
CANDIDATE_LIMIT = 12
QDRANT_SEARCH_LIMIT = 48
RESULT_LIMIT = 4
ALLOWED_BOOKS = (
    "Kisner",
    "Kisner Ch16-17",
    "shoulder_joint_pathologies",
    "shoulder_rotatory_cuff",
)
CONDITION_QUERIES = {
    "4810": (
        "Frozen shoulder adhesive capsulitis conservative home exercises stretching range of "
        "motion pendulum wall climbing self mobilization"
    ),
    "4820": (
        "Conservative management painful shoulder syndromes without dislocation: control pain; "
        "maintain joint integrity; pendulum exercises without weights; passive, active-assistive "
        "or self-assisted shoulder range of motion; multiple-angle muscle setting; protected "
        "stabilization exercises; rotator cuff, biceps, and scapular muscles."
    ),
}


class QueryEmbedder(Protocol):
    def encode(self, text: str, *, normalize_embeddings: bool) -> NDArray[np.float32]: ...


class Reranker(Protocol):
    def predict(self, pairs: list[list[str]]) -> NDArray[np.float32]: ...


class CloudflareReranker:
    def predict(self, pairs: list[list[str]]) -> NDArray[np.float32]:
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Each reranker input must contain a query and passage")
        if not pairs:
            return np.empty(0, dtype=np.float32)
        query = pairs[0][0]
        if any(pair[0] != query for pair in pairs):
            raise ValueError("Cloudflare reranker batch must use one query")
        return np.asarray(cloudflare_rerank(query, [pair[1] for pair in pairs]), dtype=np.float32)


@dataclass(frozen=True)
class RetrievedSection:
    section: IndexedSection
    score: float


@lru_cache(maxsize=1)
def _load_embedder() -> QueryEmbedder:
    return CloudflareEmbeddingModel()


@lru_cache(maxsize=1)
def _load_reranker() -> Reranker:
    return CloudflareReranker()


def _section_from_payload(payload: dict[str, object]) -> IndexedSection:
    """Normalize chunk payloads and retain their complete parent section as evidence."""
    book = payload.get("book")
    text = payload.get("text")
    parent_text = payload.get("parent_text")
    section_id = payload.get("section_id")
    if not isinstance(section_id, str):
        parent_id = payload.get("parent_id")
        child_idx = payload.get("child_idx")
        if not isinstance(parent_id, str) or not isinstance(child_idx, int):
            raise TypeError("Qdrant payload has no stable section identifier")
        section_id = parent_id

    heading_path = payload.get("heading_path", payload.get("section_path", []))
    if isinstance(heading_path, str):
        heading_path = [part.strip() for part in heading_path.split(">") if part.strip()]
    if not isinstance(heading_path, list) or not all(isinstance(part, str) for part in heading_path):
        raise ValueError(f"Qdrant payload has invalid section path: {section_id}")

    page_start = payload.get("page_start", payload.get("pdf_page"))
    page_end = payload.get("page_end", payload.get("pdf_page"))
    if not isinstance(book, str) or not isinstance(text, str):
        raise TypeError(f"Qdrant payload is missing book or text: {section_id}")
    if not isinstance(page_start, int) or not isinstance(page_end, int):
        raise TypeError(f"Qdrant payload is missing page range: {section_id}")
    evidence_text = parent_text if isinstance(parent_text, str) and parent_text.strip() else text

    return IndexedSection(
        book=book,
        section_id=section_id,
        heading_path=heading_path,
        text=evidence_text,
        page_start=page_start,
        page_end=page_end,
    )


def retrieve(
    condition_id: str,
    *,
    client: QdrantClient | None = None,
    embedder: QueryEmbedder | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedSection]:
    """Search by final diagnosis and return reranked candidates for safety filtering."""
    if condition_id not in CONDITION_QUERIES:
        raise ValueError(f"Unsupported diagnosis: {condition_id}")
    if client is None:
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    if embedder is None:
        embedder = _load_embedder()
    if reranker is None:
        reranker = _load_reranker()

    query = CONDITION_QUERIES[condition_id]
    vector = embedder.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    book_filter = models.Filter(must=[
        models.FieldCondition(key="book", match=models.MatchAny(any=list(ALLOWED_BOOKS)))
    ])
    collection = ACTIVE_ALIAS if client.collection_exists(ACTIVE_ALIAS) else ACTIVE_COLLECTION
    logger.info("%s", json.dumps({
        "event": "rag_retrieval_started",
        "condition_id": condition_id,
        "collection": collection,
        "query": query,
        "embedding_model": EMBED_MODEL,
        "query_embedding_count": 1,
        "query_vector_dimensions": len(vector),
        "qdrant_vector_payload_requested": False,
        "qdrant_limit": QDRANT_SEARCH_LIMIT,
    }, separators=(",", ":")))
    points = client.query_points(
        collection,
        query=vector,
        query_filter=book_filter,
        limit=QDRANT_SEARCH_LIMIT,
        with_payload=True,
    ).points
    logger.info("%s", json.dumps({
        "event": "rag_qdrant_candidates_fetched",
        "condition_id": condition_id,
        "collection": collection,
        "point_count": len(points),
    }, separators=(",", ":")))
    sections: list[IndexedSection] = []
    seen_sections: set[str] = set()
    for point in points:
        section = _section_from_payload(dict(point.payload or {}))
        if section.section_id not in seen_sections:
            seen_sections.add(section.section_id)
            sections.append(section)
        if len(sections) == CANDIDATE_LIMIT:
            break
    if not sections:
        return []
    pairs = [[query, section.embedding_text] for section in sections]
    scores = reranker.predict(pairs)
    ranked = sorted(
        (RetrievedSection(section=section, score=float(score)) for section, score in zip(sections, scores, strict=True)),
        key=lambda hit: hit.score,
        reverse=True,
    )
    logger.info("%s", json.dumps({
        "event": "rag_sections_reranked",
        "condition_id": condition_id,
        "candidate_count": len(sections),
        "reranker_model": RERANK_MODEL,
        "sections": [
            {
                "section_id": hit.section.section_id,
                "book": hit.section.book,
                "page_start": hit.section.page_start,
                "page_end": hit.section.page_end,
                "score": round(hit.score, 4),
            }
            for hit in ranked[:CANDIDATE_LIMIT]
        ],
    }, separators=(",", ":")))
    # Return reranked candidates before the condition safety filter runs.
    # Otherwise an irrelevant top-four hit can hide a useful candidate below it.
    return ranked[:CANDIDATE_LIMIT]
