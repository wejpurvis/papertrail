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


# Test chunk text function (pure functional test, not API-related)
def test_chunk_text():
    text = "This is a test string to be chunked into smaller pieces for testing."
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == [
        "This is a test string",
        "test string to be chunked",
        "be chunked into smaller pieces",
        "smaller pieces for testing.",
    ]


# Test chunk text with short text (pure functional test, not API-related)
def test_chunk_text_short():
    text = "the cat sat on the mat today"  # 7 words
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == ["the cat sat on the", "on the mat today"]


LONG_ABSTRACT = (
    "This paper presents a novel approach to information retrieval using transformer "
    "based language models. We evaluate our method on several benchmark datasets and "
    "demonstrate improvements over existing baselines. Our approach leverages dense "
    "vector representations to enable semantic search across large document collections. "
    "Experiments show that our model achieves state of the art performance while "
    "maintaining reasonable computational efficiency across a variety of real world tasks."
)  # 70 words — enough to satisfy the chunker's overlap=50 threshold


@pytest.fixture
async def paper_with_chunks(client):
    """Creates a paper, triggers chunking, and returns the paper_id."""
    create_response = await client.post(
        "/papers/",
        json={
            "title": "Chunk Test Paper",
            "abstract": LONG_ABSTRACT,
            "authors": ["Alice"],
            "year": 2024,
        },
    )
    paper_id = create_response.json()["id"]
    await client.post(f"/papers/{paper_id}/chunks")
    return paper_id


# Test create chunks: check that chunk count is greater than 0
@pytest.mark.asyncio
async def test_create_chunks(client, paper_with_chunks):
    response = await client.get(f"/papers/{paper_with_chunks}/chunks")
    assert response.status_code == 200
    assert len(response.json()["chunks"]) > 0


# Test get chunks: list is non-empty and each chunk has expected fields
@pytest.mark.asyncio
async def test_get_chunks(client, paper_with_chunks):
    response = await client.get(f"/papers/{paper_with_chunks}/chunks")
    assert response.status_code == 200
    data = response.json()
    assert data["paper_id"] == paper_with_chunks
    assert len(data["chunks"]) > 0
    for chunk in data["chunks"]:
        assert "id" in chunk
        assert "text" in chunk
        assert "index" in chunk
        assert "paper_id" in chunk


# Test chunks are embedded in the paper response
@pytest.mark.asyncio
async def test_get_paper_with_chunks(client, paper_with_chunks):
    response = await client.get(
        f"/papers/{paper_with_chunks}", params={"include_chunks": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert len(data["chunks"]) > 0
    for chunk in data["chunks"]:
        assert "id" in chunk
        assert "text" in chunk
        assert "index" in chunk
