r"""
Tests for the paper ingestion, retrieval, chunking, embedding, and search pipeline.

Test setup:
- Each test function gets a fresh in-memory SQLite database via Tortoise ORM
- API calls go through an httpx AsyncClient pointed at the FastAPI ASGI app
- Integration tests (marked with @pytest.mark.integration) hit the live arXiv API
  and can be run with: pytest -m integration

Covered cases:
  Papers API:
  - POST /papers/             — create a paper and verify returned fields
  - GET  /papers/{id}         — retrieve an existing paper by ID
  - GET  /papers/{id}         — 404 for a non-existent paper ID
  - GET  /papers/{id}?include_chunks=true — retrieve a paper with its chunks

  Chunking:
  - POST /papers/{id}/chunks  — trigger chunking of a paper's abstract
  - GET  /papers/{id}/chunks  — list chunks with expected fields (id, text, index, paper_id)
  - chunk_text()              — unit tests for word-based chunking with overlap

  Embeddings:
  - embed_text()              — returns a 384-float vector
  - cosine_similarity()       — 1.0 for identical vectors, 0.0 for orthogonal

  Ingest & Search (integration):
  - POST /ingest              — fetch from arXiv, store papers/chunks/embeddings
  - POST /ingest (duplicate)  — skips already-ingested papers, reports skipped count
  - POST /search              — embeds query, returns ranked results with scores in [0, 1]
"""

import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from tortoise import Tortoise
from app.main import app
from app.chunker import chunk_text
from app.arxiv import fetch_arxiv_papers
from app.embedder import embed_text, cosine_similarity


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


@pytest.mark.integration
async def test_fetch_arxiv(query: str = "neural networks", max_results: int = 2):
    """
    Call `fetch_arxiv_papers("neural networks", max_results=2)`, assert you get 2 results each with required fields.

    Note: requires network access to arXiv API, so this is marked as an integration test. You can run it with `pytest -m integration`.
    """

    papers = await fetch_arxiv_papers(query, max_results=max_results)
    assert len(papers) == max_results
    for paper in papers:
        assert "arxiv_id" in paper
        assert "title" in paper
        assert "abstract" in paper
        assert "authors" in paper
        assert "year" in paper


@pytest.mark.asyncio
async def test_embed_text(text: str = "test string"):
    """
    Call `embed_text("test string")`, assert it returns a list of 384 floast
    """

    vector = embed_text(text)
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.asyncio
async def test_cosine_similarity():
    """
    Test `cosine_similarity` with two identical vectors (should be 1.0) and two orthogonal vectors (should be 0.0)
    """

    vec1 = [1.0] + [0.0] * 383
    vec2 = [1.0] + [0.0] * 383
    vec3 = [0.0] * 384

    sim_identical = cosine_similarity(vec1, vec2)
    sim_orthogonal = cosine_similarity(vec1, vec3)

    assert sim_identical == pytest.approx(1.0)
    assert sim_orthogonal == pytest.approx(0.0)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_endpoint(query: str = "quantum computing", max_results: int = 2):
    """
    POST to `/ingest` with a query, assert papers and chunks were created in the database
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        ingest_response = await client.post(
            "/ingest", params={"query": query, "max_results": max_results}
        )
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert "ingested_papers" in ingest_data
        assert "ingested_chunks" in ingest_data
        assert ingest_data["ingested_papers"] > 0
        assert ingest_data["ingested_chunks"] > 0


@pytest.mark.integration
async def test_search_endpoint(query: str = "machine learning", max_results: int = 2):
    """
    Ingest some papers, then POST to `/search` with a relevant query, assert you get back results with scores between 0 and 1.

    Note: this test relies on the ingest endpoint which calls the live arXiv API, so it's marked as an integration test. You can run it with `pytest -m integration`.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Ingest papers so there is embedded data to search against
        ingest_response = await client.post(
            "/ingest", params={"query": query, "max_results": max_results}
        )
        assert ingest_response.status_code == 200

        # Search with the same query — expect ranked results
        search_response = await client.post(
            "/search", params={"query": query, "top_k": max_results}
        )
        assert search_response.status_code == 200
        data = search_response.json()

        assert "query" in data
        assert data["query"] == query
        assert "results" in data
        assert len(data["results"]) > 0

        for result in data["results"]:
            assert "paper_id" in result
            assert "chunk_id" in result
            assert "chunk_text" in result
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_duplicate_ingest(query: str = "deep learning", max_results: int = 2):
    """
    Ingest the same query twice, assert that the second time it skips existing papers and does not create duplicates in the database.
    """

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # First ingest
        ingest_response_1 = await client.post(
            "/ingest", params={"query": query, "max_results": max_results}
        )
        assert ingest_response_1.status_code == 200
        data_1 = ingest_response_1.json()
        assert "ingested_papers" in data_1
        assert data_1["ingested_papers"] > 0

        # Second ingest with the same query
        ingest_response_2 = await client.post(
            "/ingest", params={"query": query, "max_results": max_results}
        )
        assert ingest_response_2.status_code == 200
        data_2 = ingest_response_2.json()
        assert "skipped_papers" in data_2
        assert data_2["skipped_papers"] > 0
