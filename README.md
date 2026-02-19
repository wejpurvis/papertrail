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
pytest tests/
```

## Status

### Day 1

Estbalished working API that can store and retrieve scientific papers in a SQLite database. No external data sources yet: manually sending data to local API. Everything is async & tested.

### Day 2

WIP

## License

This project is licensed under the MIT license - see the [LICENSE](license.txt) file for details.
