# Multi-Source Production RAG

A production-grade Retrieval-Augmented Generation system, built incrementally
and documented at each step. Ingests documents from Azure Blob Storage, local
filesystem, and a REST API, and answers questions grounded in that content.

This repo is built in small, deliberate steps — each step is a working
commit, not a dump of finished code. The goal is to show *how* a production
AI system is designed, not just the end result.

## Progress

- [x] **Step 1 — Config & Logging spine**
      Typed settings (`app/config.py`) and structured JSON logging
      (`app/logger.py`). No AI logic yet — this is the foundation every
      later step builds on.
- [ ] Step 2 — Local file ingestion (parse + chunk, no embeddings yet)
- [ ] Step 3 — Embeddings + vector store for local source
- [ ] Step 4 — Basic retrieval + LLM answer (first working RAG)
- [ ] Step 5 — Azure Blob source added behind the same interface
- [ ] Step 6 — API source added
- [ ] Step 7 — Reranker
- [ ] Step 8 — Fallback handling
- [ ] Step 9 — FastAPI wrapper
- [ ] Step 10 — Eval harness
- [ ] Step 11 — Docker
- [ ] Step 12 — CI

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt
python main.py
```

## Design principles followed in this repo

1. **Contract-first** — every component (data source, embedder, vector
   store) is built behind an interface before a second implementation
   is added.
2. **One concern per step** — each commit adds exactly one capability.
3. **No `print()`** — structured logs from the very first line of code.
4. **Typed config** — no scattered constants or hardcoded values.
