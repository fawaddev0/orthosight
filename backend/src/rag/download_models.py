"""Download local embedding and reranking models before serving requests."""

from huggingface_hub import snapshot_download

from rag.embedding_generator import EMBED_MODEL
from rag.retriever import RERANK_MODEL


def main() -> None:
    for model_id in (EMBED_MODEL, RERANK_MODEL):
        model_path = snapshot_download(repo_id=model_id)
        print(f"Cached {model_id} at {model_path}")


if __name__ == "__main__":
    main()
