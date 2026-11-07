"""Retrieve complete book sections for supported conservative diagnoses."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from rag.embedding_generator import ACTIVE_ALIAS, EMBED_MODEL, IndexedSection

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATE_LIMIT = 12
RESULT_LIMIT = 4
ALLOWED_BOOKS = ("shoulder_joint_pathologies", "shoulder_rotatory_cuff")
CONDITION_QUERIES = {
    "4810": "Primary adhesive capsulitis frozen shoulder conservative home exercises range of motion self stretching",
    "4820": "Rotator cuff tendinopathy subacromial impingement conservative progressive exercises home program",
}


class QueryEmbedder(Protocol):
    def encode(self, text: str, *, normalize_embeddings: bool) -> NDArray[np.float32]: ...


class PairTokenizer(Protocol):
    def encode(
        self, query: str, passage: str, *, add_special_tokens: bool, truncation: bool
    ) -> list[int]: ...


class Reranker(Protocol):
    tokenizer: PairTokenizer
    max_length: int

    def predict(self, pairs: list[list[str]]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class RetrievedSection:
    section: IndexedSection
    score: float


def retrieve(
    condition_id: str,
    *,
    client: QdrantClient | None = None,
    embedder: QueryEmbedder | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedSection]:
    """Search by final diagnosis only; rerank 12 candidates and return up to 4."""
    if condition_id not in CONDITION_QUERIES:
        raise ValueError(f"Unsupported diagnosis: {condition_id}")
    if client is None:
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = cast(QueryEmbedder, SentenceTransformer(EMBED_MODEL))
    if reranker is None:
        from sentence_transformers import CrossEncoder

        reranker = cast(Reranker, CrossEncoder(RERANK_MODEL, max_length=8192))

    query = CONDITION_QUERIES[condition_id]
    vector = embedder.encode(query, normalize_embeddings=True).tolist()
    book_filter = models.Filter(must=[
        models.FieldCondition(key="book", match=models.MatchAny(any=list(ALLOWED_BOOKS)))
    ])
    points = client.query_points(
        ACTIVE_ALIAS,
        query=vector,
        query_filter=book_filter,
        limit=CANDIDATE_LIMIT,
        with_payload=True,
    ).points
    sections = [IndexedSection.model_validate(point.payload) for point in points]
    if not sections:
        return []
    pairs = [[query, section.embedding_text] for section in sections]
    for pair, section in zip(pairs, sections, strict=True):
        if len(reranker.tokenizer.encode(pair[0], pair[1], add_special_tokens=True, truncation=False)) > reranker.max_length:
            raise ValueError(f"Reranker input would truncate section: {section.section_id}")
    scores = reranker.predict(pairs)
    ranked = sorted(
        (RetrievedSection(section=section, score=float(score)) for section, score in zip(sections, scores, strict=True)),
        key=lambda hit: hit.score,
        reverse=True,
    )
    return ranked[:RESULT_LIMIT]
