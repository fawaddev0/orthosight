"""Typed REST client for Cloudflare Workers AI models used by the RAG pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

CLOUDFLARE_API_ROOT = "https://api.cloudflare.com/client/v4/accounts"
EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5"
RERANKER_MODEL = "@cf/baai/bge-reranker-base"

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class EmbeddingResult(BaseModel):
    data: list[list[float]]


class RerankResultItem(BaseModel):
    id: int
    score: float


class RerankResult(BaseModel):
    response: list[RerankResultItem]


class CloudflareEnvelope(BaseModel):
    result: object
    success: bool


ResultModel = TypeVar("ResultModel", bound=BaseModel)


def _post(model_id: str, payload: dict[str, object], response_model: type[ResultModel]) -> ResultModel:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_WORKER_AI_TOKEN")
    if not account_id or not token:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_WORKER_AI_TOKEN are required")

    request = Request(
        f"{CLOUDFLARE_API_ROOT}/{account_id}/ai/run/{model_id}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Cloudflare Workers AI request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("Cloudflare Workers AI request failed") from exc

    try:
        envelope = CloudflareEnvelope.model_validate_json(body)
        result = response_model.model_validate(envelope.result)
    except ValidationError as exc:
        raise RuntimeError("Cloudflare Workers AI returned an invalid response") from exc
    if not envelope.success:
        raise RuntimeError("Cloudflare Workers AI reported request failure")
    return result


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Return Cloudflare BGE Small embeddings for input texts."""
    if not texts:
        return []
    result = _post(EMBEDDING_MODEL, {"text": texts}, EmbeddingResult)
    return result.data


def rerank(query: str, contexts: list[str]) -> list[float]:
    """Return relevance scores in the same order as input contexts."""
    if not contexts:
        return []
    result = _post(
        RERANKER_MODEL,
        {
            "query": query,
            "contexts": [{"text": context} for context in contexts],
            "top_k": len(contexts),
        },
        RerankResult,
    )
    scores = [0.0] * len(contexts)
    seen_ids: set[int] = set()
    for item in result.response:
        if item.id < 0 or item.id >= len(contexts) or item.id in seen_ids:
            raise RuntimeError("Cloudflare Workers AI returned an invalid reranker context index")
        seen_ids.add(item.id)
        scores[item.id] = item.score
    if len(seen_ids) != len(contexts):
        raise RuntimeError("Cloudflare Workers AI returned incomplete reranker results")
    return scores
