# Papertrail

A toy FastAPI application for ingesting and retrieving scientific papers, 
built to learn FastAPI, Tortoise ORM, and async Python patterns.

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
pytest tests/
```

## Status

WIP

## License

This project is licensed under the MIT license - see the [LICENSE](license.txt) file for details.
