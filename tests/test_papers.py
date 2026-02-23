r"""
Integration tests for the /papers/ API endpoints.

Uses an in-memory SQLite database (via Tortoise ORM) spun up fresh for each
test function, and an httpx AsyncClient pointed at the FastAPI ASGI app.

Covered cases:
- POST /papers/    — create a paper and verify the returned fields
- GET  /papers/{id} — retrieve an existing paper by ID
- GET  /papers/{id} — 404 response for a non-existent paper ID
"""

import pytest
from httpx import AsyncClient, ASGITransport
from tortoise import Tortoise
from app.main import app
from app.chunker import chunk_text


@pytest.fixture(scope="function", autouse=True)
async def setup_db():
    await Tortoise.init(db_url="sqlite://:memory:", modules={"models": ["app.models"]})
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# Test creating a paper
@pytest.mark.asyncio
async def test_create_paper(client):
    response = await client.post(
        "/papers/",
        json={
            "title": "Test Paper",
            "abstract": "This is a test abstract.",
            "authors": ["Alice", "Bob"],
            "year": 2024,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Paper"
    assert data["abstract"] == "This is a test abstract."
    assert data["authors"] == ["Alice", "Bob"]
    assert data["year"] == 2024
    assert "id" in data
    assert "created_at" in data


# Test retrieving a paper
@pytest.mark.asyncio
async def test_get_paper(client):
    # First create a paper to retrieve
    create_response = await client.post(
        "/papers/",
        json={
            "title": "Another Test Paper",
            "abstract": "This is another test abstract.",
            "authors": ["Charlie", "Dana"],
            "year": 2023,
        },
    )
    paper_id = create_response.json()["id"]

    # Now retrieve it
    get_response = await client.get(f"/papers/{paper_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["id"] == paper_id
    assert data["title"] == "Another Test Paper"
    assert data["abstract"] == "This is another test abstract."
    assert data["authors"] == ["Charlie", "Dana"]
    assert data["year"] == 2023


# Test retrieving a non-existent paper
@pytest.mark.asyncio
async def test_get_nonexistent_paper(client):
    response = await client.get("/papers/9999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Paper not found"}


# Test chunk text function
@pytest.mark.asyncio
async def test_chunk_text():
    text = "This is a test string to be chunked into smaller pieces for testing."
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == [
        "This is a test string",
        "test string to be chunked",
        "be chunked into smaller pieces",
        "smaller pieces for testing.",
    ]


# Test chunk text with short text
@pytest.mark.asyncio
async def test_chunk_text_short():
    text = "the cat sat on the mat today"  # 7 words
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == ["the cat sat on the", "on the mat today"]
