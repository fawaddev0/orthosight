# Run the RAG pipeline with Docker

Run commands from `backend`. Docker Compose starts Qdrant, preprocesses
the four included digital PDF excerpts, indexes ready sections, and activates
the collection. The first build and model downloads can take time.

```bash
docker compose run --build --rm rag
```

The command exits after indexing. Qdrant stays running. Each version can be
indexed once. To index changed books, use a new version:

```bash
RAG_COLLECTION_VERSION=v2 docker compose run --build --rm rag
```

To use your own PDF or EPUB books, point `RAG_BOOKS_DIR` at their directory:

```bash
RAG_BOOKS_DIR=/absolute/path/to/books docker compose run --build --rm rag
```

Set a new `RAG_COLLECTION_VERSION` for each indexing run. Processing output
and review files remain under that version in the `processed_data` Docker volume. Only sections
marked `ready_for_embedding` are indexed. The container disables OCR, so use
digital PDFs or EPUBs with extractable text. The retriever searches only books
named `shoulder_joint_pathologies` and `shoulder_rotatory_cuff` and supports
diagnosis IDs `4810` and `4820`.

The existing plan generator is a Python function, `generate_plan`, with no CLI.
This Docker command prepares and indexes the books. You can run Python inside
the same image with `docker compose run --rm rag python ...` to call it.

Stop Qdrant with `docker compose down`. Docker volumes retain the index, model
cache, and processed files.
