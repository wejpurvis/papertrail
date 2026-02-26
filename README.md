# Papertrail

A toy FastAPI application for ingesting and querying scientific papers, built to learn FastAPI, Tortoise ORM, async Python, and LLM integration.

## Stack

- FastAPI
- Tortoise ORM + SQLite
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- Claude (Haiku) or Ollama for LLM responses
- pytest + pytest-asyncio

## Setup

```bash
conda env create -f environment.yml
conda activate papertrail
```

## Configuration

Copy the example env file and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_PROVIDER=claude          # or "ollama" to run locally for free
ANTHROPIC_API_KEY=sk-ant-... # only required if LLM_PROVIDER=claude
```

If using Ollama, make sure it's running locally with a model pulled:

```bash
ollama pull llama3.2
ollama serve
```

## Run

```bash
uvicorn app.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

> **Note:** Always run `uvicorn` from the project root. The SQLite database is created as `papertrail.db` relative to your working directory. To fully reset the database, delete all three SQLite files: `papertrail.db`, `papertrail.db-shm`, and `papertrail.db-wal`.

## How to use

1. **Ingest papers** — Enter a search topic (e.g. "Carbon Capture") and hit **Ingest**. Papers are fetched from arXiv, chunked, and embedded automatically.
2. **Upload a PDF** — Use the upload button to ingest a local PDF instead.
3. **Ask a question** — Type a question and hit **Ask**. The app retrieves the most relevant chunks via semantic search and sends them as context to the LLM, which returns a cited answer.

## API

|Method|Endpoint|Description|
|------|--------|-----------|
|`POST`|`/ingest`|Fetch and ingest papers from arXiv|
|`POST`|`/upload-pdf`|Upload and ingest a PDF|
|`POST`|`/ask`|Ask a question over ingested papers|
|`POST`|`/search`|Semantic search over chunks|
|`GET`|`/papers/{id}`|Retrieve a paper (optionally with chunks)|

## Test

```bash
pytest tests/ -v -m "not integration"  # skip arXiv network tests
pytest tests/ -v                        # run everything including network tests
```

## Status

### Day 1

Estbalished working API that can store and retrieve scientific papers in a SQLite database. No external data sources yet: manually sending data to local API. Everything is async & tested.

### Day 2

Added a `Chunk` model related to `Paper` via a foreign key, and a text chunking utility that splits abstracts into overlapping word windows. New endpoints trigger chunking and retrieve chunks per paper, with an optional `include_chunks` query parameter on the existing paper endpoint. BackgroundTasks used for async chunking.

### Day 3

Connected to the arXiv API to ingest real papers automatically. Added a sentence-transformers embedding model (`all-MiniLM-L6-v2`) and an `Embedding` model (one-to-one with `Chunk`) to store 384-dimensional vectors. Implemented semantic search via cosine similarity, returning the most relevant chunks for a natural language query. Duplicate ingestion is handled by deduplicating on `arxiv_id`.

### Day 4

Added a full RAG pipeline with a `/ask` endpoint: embeds the question, retrieves the top-k most relevant chunks via cosine similarity, and sends them as context to an LLM to generate a cited answer. LLM backend is swappable via `LLM_PROVIDER` env var — supports Claude (Haiku) and Ollama for local/free usage. Added PDF ingestion via `/upload-pdf` (text extracted with pypdf, then chunked and embedded like arXiv papers). Wired everything into a simple browser UI for end-to-end use without touching the API directly.

## License

This project is licensed under the MIT license - see the [LICENSE](license.txt) file for details.
