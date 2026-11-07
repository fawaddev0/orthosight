# OrthoSight backend

This project uses Python 3.15.2 and [uv](https://docs.astral.sh/uv/getting-started/installation/). Run these commands from `backend/`:

```bash
uv sync --group dev
uv run pytest tests/test_book_preprocessing.py -v
uv run python src/rag/book_preprocessor.py src/rag/test_books -o /tmp/orthosight-processed --no-ocr
```

`uv.lock` records resolved package versions. The local `.venv` is ignored by Git.

The PDF preprocessing tests use four digital shoulder-book excerpts in `src/rag/test_books`. OCR on scanned PDFs requires Tesseract and language data, which this project does not install. Processing reports may still require manual review.

Strict mypy and Ruff checks for `src/rag` currently report existing issues. Run them with:

```bash
uv run mypy --strict src/rag
uv run ruff check src/rag
```
