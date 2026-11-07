"""Index book_preprocessor.py sections.jsonl output in Qdrant.

Run: python -m rag.embedding_generator processed/ --category rehab_exercise
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import BaseModel, Field

COLLECTION = "physio_sections_v1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_CHUNK_TOKENS = 350


class Source(BaseModel):
    page: int


class Paragraph(BaseModel):
    text: str
    sources: list[Source] = Field(default_factory=list)


class Section(BaseModel):
    book: str
    section_id: str
    heading_path: list[str]
    paragraphs: list[Paragraph]
    page_start: int
    page_end: int


class Chunk(BaseModel):
    chunk_id: str
    parent_text: str
    section_path: str
    book: str
    pdf_page: int
    page_start: int
    page_end: int
    body_part: str
    category: str


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


def chunks_for_section(
    section: Section,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    category: str,
    body_part: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    path = " > ".join(section.heading_path)
    budget = max_tokens - len(encode(path)) - 2
    if budget < 1:
        raise ValueError(f"Heading path exceeds token budget: {section.section_id}")
    pieces: list[tuple[str, int, int]] = []
    for paragraph in section.paragraphs:
        token_ids = encode(paragraph.text)
        if not token_ids:
            continue
        pages = [source.page for source in paragraph.sources]
        first = min(pages, default=section.page_start)
        last = max(pages, default=section.page_end)
        for offset in range(0, len(token_ids), budget):
            text = decode(token_ids[offset:offset + budget]).strip()
            if text:
                pieces.append((text, first, last))

    chunks: list[Chunk] = []
    group: list[str] = []
    group_pages: list[tuple[int, int]] = []
    size = 0

    def flush() -> None:
        nonlocal size
        if not group:
            return
        index = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{section.section_id}:c{index:04d}",
            parent_text="\n\n".join(group),
            section_path=path,
            book=section.book,
            pdf_page=group_pages[0][0],
            page_start=min(start for start, _ in group_pages),
            page_end=max(end for _, end in group_pages),
            body_part=body_part,
            category=category,
        ))
        group.clear()
        group_pages.clear()
        size = 0

    for text, first, last in pieces:
        cost = len(encode(text))
        if group and size + cost + 2 > budget:
            flush()
        group.append(text)
        group_pages.append((first, last))
        size += cost + (2 if len(group) > 1 else 0)
    flush()
    return chunks


def index_sections(path: Path, category: str, body_part: str) -> int:
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    tokenizer = model.tokenizer
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise RuntimeError("Embedding model did not report vector dimension")
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE))
        for field in ("body_part", "category", "book"):
            client.create_payload_index(COLLECTION, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)

    batch: list[Chunk] = []
    total = 0

    def upload() -> None:
        nonlocal total
        if not batch:
            return
        texts = [(chunk.section_path + "\n" + chunk.parent_text).strip() for chunk in batch]
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=64)
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
            vector=vector.tolist(),
            payload=chunk.model_dump(),
        ) for chunk, vector in zip(batch, vectors, strict=True)]
        client.upsert(COLLECTION, points=points, wait=True)
        total += len(batch)
        batch.clear()

    for section in read_sections(path):
        for chunk in chunks_for_section(
            section,
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            category,
            body_part,
        ):
            batch.append(chunk)
            if len(batch) >= 128:
                upload()
    upload()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Index preprocessed book sections")
    parser.add_argument("path", type=Path, help="sections.jsonl file or processed books directory")
    parser.add_argument("--category", choices=("rehab_exercise", "reference"), default="reference")
    parser.add_argument("--body-part", default="general", help="Metadata for all sections in this run")
    args = parser.parse_args()
    print(f"Indexed {index_sections(args.path, args.category, args.body_part)} chunks in {COLLECTION}")


if __name__ == "__main__":
    main()
