"""Focused RAG pipeline tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag.embedding_generator import (
    ACTIVE_ALIAS,
    VECTOR_SIZE,
    activate_collection,
    collection_name,
    index_sections,
    prepare_sections,
)


def _write_sections(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _section(section_id: str, text: str, ready: bool = True) -> dict[str, object]:
    return {
        "book": "shoulder_book",
        "section_id": section_id,
        "heading_path": ["Shoulder", "Exercises"],
        "page_start": 3,
        "page_end": 4,
        "text": text,
        "ready_for_embedding": ready,
        "paragraphs": [],
    }


def test_prepare_sections_preserves_ready_section_without_splitting(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    text = "First complete sentence.\n\nSecond complete sentence."
    _write_sections(path, [_section("one", text), _section("two", "Flagged.", False)])

    records = prepare_sections(path, lambda value: len(value.split()), max_tokens=20)

    assert len(records) == 1
    assert records[0].section_id == "one"
    assert records[0].text == text
    assert records[0].heading_path == ["Shoulder", "Exercises"]
    assert records[0].page_start == 3
    assert records[0].page_end == 4


def test_prepare_sections_rejects_oversized_section_before_indexing(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    _write_sections(path, [_section("too-long", "one two three four five six")])

    with pytest.raises(ValueError, match="too-long"):
        prepare_sections(path, lambda value: len(value.split()), max_tokens=4)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(range(len(text.split())))


class _EmbeddingModel:
    tokenizer = _Tokenizer()
    max_seq_length = 8192

    def encode(self, texts: list[str], *, normalize_embeddings: bool) -> np.ndarray:
        vectors = np.zeros((len(texts), VECTOR_SIZE), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def test_index_and_activate_complete_section(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    original = "First sentence.\n\nSecond sentence."
    _write_sections(path, [_section("one", original)])
    client = QdrantClient(":memory:")

    assert index_sections(path, "v1", model=_EmbeddingModel(), client=client) == 1
    points, _ = client.scroll(collection_name("v1"), with_payload=True)
    assert len(points) == 1
    assert points[0].payload is not None
    assert points[0].payload["text"] == original

    with pytest.raises(ValueError, match="unexpected section count"):
        activate_collection("v1", 2, client=client)
    assert client.get_aliases().aliases == []

    activate_collection("v1", 1, client=client)
    assert client.get_aliases().aliases[0].alias_name == ACTIVE_ALIAS


def test_index_rejects_model_that_would_truncate_sections(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    _write_sections(path, [_section("one", "one two three four five")])
    client = QdrantClient(":memory:")
    model = _EmbeddingModel()
    model.max_seq_length = 4

    with pytest.raises(ValueError, match="4 embedding tokens"):
        index_sections(path, "v1", model=model, client=client)
    assert not client.collection_exists(collection_name("v1"))
