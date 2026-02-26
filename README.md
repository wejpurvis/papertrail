# Papertrail

A toy FastAPI application for ingesting and retrieving scientific papers, built to learn FastAPI, Tortoise ORM, and async Python patterns.

## Stack

- FastAPI
- Tortoise ORM + SQLite
- pytest + pytest-asyncio

## Setup

```bash
conda env create -f environment.yml
conda activate papertrail
```

## Run

```bash
uvicorn app.main:app --reload
```

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

## License

This project is licensed under the MIT license - see the [LICENSE](license.txt) file for details.
