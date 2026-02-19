# Day 1 — Foundation: FastAPI + Tortoise ORM + SQLite

## Goal

By the end of today you have a working API that can store and retrieve scientific
papers in a SQLite database. No external data sources yet — you are manually
sending data to your own API. Everything is async, tested, and the project
structure is clean enough to build on for the next three days.

---

## Context: Where do the papers come from today?

Nowhere external. You are the data source. You will send paper data directly to
your API using either:

- The FastAPI auto-generated docs UI at `http://localhost:8000/docs` (easiest
  for manual testing)
- Your pytest test suite (the real test)

On Day 3 you will connect to the arXiv API (free, no key required) to ingest
real papers automatically. Today is purely about getting the plumbing right.

---

## Stack for today

| Tool                    | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| FastAPI                 | Web framework, handles HTTP routing                        |
| Pydantic                | Request/response validation (built into FastAPI)           |
| Tortoise ORM            | Async ORM for talking to the database                      |
| SQLite                  | Database — stored as a local `.db` file, no setup required |
| aiosqlite               | Async SQLite driver that Tortoise uses under the hood      |
| pytest + pytest-asyncio | Test framework with async support                          |
| httpx                   | Async HTTP client used by FastAPI's test client            |

---

## Project structure to create

```txt
papertrail/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app, routes, lifespan
│   ├── models.py        ← Tortoise ORM database models
│   ├── schemas.py       ← Pydantic request/response schemas
│   └── database.py      ← Tortoise configuration
├── tests/
│   ├── __init__.py
│   └── test_papers.py   ← Your test suite
├── days/
│   └── day1.md          ← This file
├── .env                 ← Never committed to git
├── .gitignore
├── environment.yml
└── README.md
```

Create all empty files first so you can see the whole structure before writing
any code:

```bash
mkdir -p app tests days
touch app/__init__.py app/main.py app/models.py app/schemas.py app/database.py
touch tests/__init__.py tests/test_papers.py
```

---

## Build order

Follow this order exactly — each file depends on the one before it.

### 1. `app/models.py`

Define a `Paper` Tortoise model with these fields:

| Field      | Type          | Notes                       |
| ---------- | ------------- | --------------------------- |
| id         | IntField      | primary key, auto-generated |
| title      | CharField     | max 500 characters          |
| abstract   | TextField     | no length limit             |
| authors    | JSONField     | store as a list of strings  |
| year       | IntField      | publication year            |
| created_at | DatetimeField | set auto_now_add=True       |

Key concept: a Tortoise model is a Python class that maps to a database table.
Each field is a class attribute. Tortoise handles the SQL underneath.

```python
from tortoise import fields
from tortoise.models import Model

class Paper(Model):
    # your fields here

    class Meta:
        table = "papers"
```

### 2. `app/schemas.py`

Define two Pydantic models:

**`PaperCreate`** — used to validate incoming POST request body:

- title: str
- abstract: str
- authors: list[str]
- year: int

**`PaperResponse`** — used to shape the API response:

- id: int
- title: str
- abstract: str
- authors: list[str]
- year: int
- created_at: datetime

Key concept: keeping ORM models and Pydantic schemas separate is deliberate.
Your ORM model is about database structure. Your Pydantic schema is about what
the API accepts and returns. They often look similar but serve different purposes
and diverge over time.

### 3. `app/database.py`

Configure Tortoise to use SQLite. You need to tell Tortoise:

- Which database to connect to (a local `.db` file)
- Where your models are (so it can generate the schema)

```python
TORTOISE_ORM = {
    "connections": {
        "default": "sqlite://./papertrail.db"
    },
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        }
    }
}
```

### 4. `app/main.py`

This is the most important file today. Three things to wire up:

**Lifespan** — Tortoise needs to initialise when the app starts and close when
it stops. FastAPI handles this via a lifespan context manager:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from tortoise import Tortoise
from app.database import TORTOISE_ORM

@asynccontextmanager
async def lifespan(app: FastAPI):
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()

app = FastAPI(lifespan=lifespan)
```

**POST /papers** — accepts a `PaperCreate` body, creates a `Paper` record,
returns a `PaperResponse`.

**GET /papers/{paper_id}** — retrieves a paper by ID, returns a `PaperResponse`,
raises a 404 if not found.

### 5. `tests/test_papers.py`

Write tests BEFORE finishing each endpoint. This is the most important habit
to build — the interview is literally a failing test suite you have to fix.

You need a test client that works with async FastAPI. The pattern looks like:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from tortoise import Tortoise
from app.main import app

@pytest.fixture(scope="function", autouse=True)
async def setup_db():
    await Tortoise.init(
        db_url="sqlite://:memory:",
        modules={"models": ["app.models"]}
    )
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
```

Write these two tests:

**`test_create_paper`** — POST a paper with valid data, assert 200 status, assert
the response contains the correct title and authors.

**`test_get_paper`** — POST a paper, then GET it by the returned ID, assert the
retrieved paper matches what was posted.

### 6. `pytest.ini`

pytest-asyncio requires explicit configuration or it will silently skip your
async tests. Create `pytest.ini` in the project root:

```ini
[pytest]
asyncio_mode = auto
```

---

## Running everything

Start the server:

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` — you should see your two endpoints. Try
POSTing a paper manually through the UI.

Run tests:

```bash
pytest tests/ -v
```

You want to see this:

```bash
tests/test_papers.py::test_create_paper PASSED
tests/test_papers.py::test_get_paper PASSED
```

---

## Success criteria

You are done with Day 1 when:

- [ ] `uvicorn app.main:app --reload` starts without errors
- [ ] `http://localhost:8000/docs` shows two endpoints
- [ ] You can POST a paper through the docs UI and get a valid response
- [ ] `pytest tests/ -v` shows 2 passing tests
- [ ] `papertrail.db` exists in your project root (the real database)
- [ ] No secrets or `.db` files committed to git

---

## Where to use Claude Code today

- Ask it to explain the lifespan pattern if it's not clicking
- Ask it to explain why pytest-asyncio needs `asyncio_mode = auto`
- Ask it to help debug if your tests are being skipped silently (common)
- **Don't** ask it to write your models or routes — write those yourself

The goal is that by tonight these patterns are in your fingers, not just working
on your screen.

---

## What's coming on Day 2

You'll add a `Chunk` model (foreign key relationship to `Paper`), a chunking
utility that splits abstracts into overlapping text segments, and GET endpoints
to retrieve chunks. This builds the retrieval foundation for the RAG layer on
Day 3.
