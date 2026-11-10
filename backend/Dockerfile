FROM ghcr.io/astral-sh/uv:python3.15-trixie-slim

WORKDIR /app
ENV UV_NO_DEV=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY src/rag ./src/rag
COPY src/decision_tree ./src/decision_tree

CMD set -e; \
    attempt=0; \
    until python -c 'import os; from qdrant_client import QdrantClient; QdrantClient(url=os.environ["QDRANT_URL"]).get_collections()' >/dev/null 2>&1; do \
      attempt=$((attempt + 1)); \
      if [ "$attempt" -ge 30 ]; then echo "Qdrant unavailable" >&2; exit 1; fi; \
      sleep 2; \
    done; \
    python src/rag/book_preprocessor.py /app/books -o "/app/processed/$RAG_COLLECTION_VERSION" --no-ocr; \
    python src/rag/embedding_generator.py "/app/processed/$RAG_COLLECTION_VERSION" --version "$RAG_COLLECTION_VERSION" --activate
