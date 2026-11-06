# Backend Subsystem

## 1. Stack & Tools
* **Language:** Python 3.15.2+
* **Frameworks:** FastAPI, Pydantic v2
* **Static Analysis & Formatting:** `mypy --strict`, `ruff`
* **Testing:** `pytest`, `pytest-asyncio`, `httpx`

---

## 2. Non-Negotiable Core Directives

### A. Read-Only Layer JSON files
* `data/layer1.json` through `data/layer4.json` are **immutable**.
* Do not write file-saving, file-updating, or file-writing logic targeting these files under any circumstances.

### B. Strict Static Typing (Zero `Any`)
* `typing.Any` is **strictly forbidden** in all Python scripts under `backend/`.
* Code must pass `mypy --strict src` with zero errors.
* Dynamic or incoming JSON data must be parsed directly into validated Pydantic models.

### C. Decoupled Core Engine
* `src/decision_tree/core.py` must maintain zero web or FastAPI dependencies.
* It must exist as a standalone, standard Python library class (`DecisionTree`) so that both the RAG pipeline and FastAPI endpoints can import and instantiate it directly.

---

## 3. SCOPED TESTING & TYPE-CHECKING

Run type checking and tests ONLY for module being edited. Do not run full backend suite during targeted work.

### A. Core Engine (`src/decision_tree/`)
- Type check: `mypy --strict src/decision_tree`
- Unit tests: `pytest tests/test_core.py -v`

### B. API Routes (`src/api/`)
- Type check: `mypy --strict src/api`
- API tests: `pytest tests/test_api.py -v`

### C. RAG Pipeline (`src/rag/`)
- Type check: `mypy --strict src/rag`
- RAG tests: `pytest tests/test_rag.py -v`

---


## 3. Architecture Overview

```text
backend/
├── src/
│   ├── decision_tree/
│   │   ├── __init__.py       # Exports DecisionTree class
│   │   ├── core.py           # Core evaluation logic & traversal engine
│   │   ├── models.py         # Pydantic models (LayerNode, DiagnosticInput, etc.)
│   │   └── exceptions.py     # Custom error hierarchy
│   ├── api/
│   └── rag/
└── tests/
    ├── test_core.py          # DecisionTree class unit tests
    ├── test_api.py           # FastAPI endpoint tests
    └── test_immutability.py  # Test enforcing read-only state of layer files
```

---

## 4. Testing & Verification Checklist

Before finalizing any backend task, agents must execute and verify:

1. **Type Checking:**
   ```bash
   mypy --strict src
   ```
2. **Linting:**
   ```bash
   ruff check src
   ```
3. **Automated Tests:**
   ```bash
   pytest tests/ -v
   ```

### Required Test Scenarios
* **File Immutability:** Verify layer JSON SHA-256 hashes do not change during test execution.
* **Core Traversal:** Verify complete traversal from Layer 1 through Layer 4 given target input parameters.
* **Error Handling:** Verify invalid layer IDs or non-existent node requests raise domain exceptions (e.g., `NodeNotFoundError`, `InvalidLayerError`).
* **API Endpoints:** Test `/api/v1/web` and `/api/v1/mobile` routes using FastAPI's `TestClient`.
