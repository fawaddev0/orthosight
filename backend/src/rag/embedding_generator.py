"""Embed complete, reviewed book sections in a versioned Qdrant collection."""

from __future__ import annotations

import argparse
import os
import re
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

EMBED_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024
MAX_INPUT_TOKENS = 8192
COLLECTION_PREFIX = "physio_sections_"
ACTIVE_ALIAS = "physio_sections_active"


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class EmbeddingModel(Protocol):
    tokenizer: Tokenizer
    max_seq_length: int

    def encode(self, texts: list[str], *, normalize_embeddings: bool) -> NDArray[np.float32]: ...


class Section(BaseModel):
    book: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    heading_path: list[str]
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    ready_for_embedding: bool


class IndexedSection(BaseModel):
    book: str
    section_id: str
    heading_path: list[str]
    text: str
    page_start: int
    page_end: int

    @property
    def embedding_text(self) -> str:
        heading = " > ".join(self.heading_path)
        return f"{heading}\n{self.text}" if heading else self.text


def collection_name(version: str) -> str:
    if re.fullmatch(r"v[1-9][0-9]*", version) is None:
        raise ValueError("Collection version must look like v1 or v2")
    return f"{COLLECTION_PREFIX}{version}"


def read_sections(path: Path) -> Iterator[Section]:
    files = sorted(path.rglob("sections.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No sections.jsonl under {path}")
    for file in files:
        with file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield Section.model_validate_json(line)
                    except ValueError as exc:
                        raise ValueError(f"{file}:{line_number}: invalid section") from exc


def prepare_sections(
    path: Path,
    count_tokens: Callable[[str], int],
    max_tokens: int = MAX_INPUT_TOKENS,
) -> list[IndexedSection]:
    """Validate all input before any vector database writes."""
    records: list[IndexedSection] = []
    seen: set[str] = set()
    for section in read_sections(path):
        if not section.ready_for_embedding:
            continue
        if section.section_id in seen:
            raise ValueError(f"Duplicate section ID: {section.section_id}")
        seen.add(section.section_id)
        record = IndexedSection.model_validate(section.model_dump(exclude={"ready_for_embedding"}))
        if not record.text.strip():
            raise ValueError(f"Empty ready section: {record.section_id}")
        if count_tokens(record.embedding_text) > max_tokens:
            raise ValueError(f"Section exceeds {max_tokens} embedding tokens: {record.section_id}")
        records.append(record)
    if not records:
        raise ValueError("No sections marked ready_for_embedding")
    return records


def index_sections(
    path: Path,
    version: str,
    *,
    model: EmbeddingModel | None = None,
    client: QdrantClient | None = None,
) -> int:
    """Create new collection only after every ready section passes preflight."""
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = cast(EmbeddingModel, SentenceTransformer(EMBED_MODEL))

    name = collection_name(version)
    records = prepare_sections(
        path,
        lambda value: len(model.tokenizer.encode(value, add_special_tokens=True)),
        max_tokens=min(MAX_INPUT_TOKENS, model.max_seq_length),
    )
    if client is None:
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    if client.collection_exists(name):
        raise ValueError(f"Collection already exists: {name}; use a new version")
    client.create_collection(name, vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE))
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        vectors = model.encode([record.embedding_text for record in batch], normalize_embeddings=True)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, record.section_id)),
                vector=vector.tolist(),
                payload=record.model_dump(),
            )
            for record, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(name, points=points, wait=True)
    count = client.count(name, exact=True).count
    if count != len(records):
        raise RuntimeError(f"Indexed {count} sections, expected {len(records)}")
    return count


def activate_collection(
    version: str,
    expected_count: int,
    *,
    client: QdrantClient | None = None,
) -> None:
    """Switch active alias after checking collection completeness and dimensions."""
    if expected_count < 1:
        raise ValueError("Expected section count must be positive")
    name = collection_name(version)
    if client is None:
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    info = client.get_collection(name)
    if client.count(name, exact=True).count != expected_count:
        raise ValueError(f"Collection {name} has unexpected section count")
    params = info.config.params.vectors
    if not isinstance(params, models.VectorParams) or params.size != VECTOR_SIZE:
        raise ValueError(f"Collection {name} has incompatible vector size")
    aliases = client.get_aliases().aliases
    actions: list[models.AliasOperations] = []
    if any(alias.alias_name == ACTIVE_ALIAS for alias in aliases):
        actions.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ACTIVE_ALIAS)))
    actions.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=name, alias_name=ACTIVE_ALIAS)))
    client.update_collection_aliases(change_aliases_operations=actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index complete book sections")
    parser.add_argument("path", type=Path, nargs="?", help="sections.jsonl file or processed books directory")
    parser.add_argument("--version", required=True, help="Collection version, such as v1")
    parser.add_argument("--activate", action="store_true", help="Point active alias to this verified collection")
    parser.add_argument("--expected-count", type=int, help="Required section count when activating an existing collection")
    args = parser.parse_args()
    if args.path is None:
        if not args.activate or args.expected_count is None:
            parser.error("Use a path to index, or --activate --expected-count to promote an existing collection")
        activate_collection(args.version, args.expected_count)
        print(f"Activated {collection_name(args.version)}")
        return
    count = index_sections(args.path, args.version)
    if args.activate:
        activate_collection(args.version, count)
    print(f"Indexed {count} sections in {collection_name(args.version)}")


if __name__ == "__main__":
    main()
