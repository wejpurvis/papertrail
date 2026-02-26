r"""
Wires everything together with two routes:

- `POST /papers/`: accepts a `PaperCreate` body, creates a `Paper` record, returns a `PaperResponse`
- `GET /papers/{paper_id}`: retrieves a paper by ID, returns a `

The `lifespan` function runs on startup/shutdown - it initialises the Tortoise connection using the config from `database.py`, auto-creates any missing tables (`generate_schemas()`), and then closes connections when the app stops.

Here is what `register_tortoise()` does under the hood:

Server starts
-> Tortoise.init() connects to SQLite and registers models
-> generate_schemas() creates the `papers` table if it doesn't exist
-> app is ready, routes can now use Paper.create() etc...

Server stops
-> Tortoise.close_connections() gracefully shuts down the DB connection

"""

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tortoise.contrib.fastapi import register_tortoise
from app.database import TORTOISE_ORM
from app.models import Chunk, Paper, Embedding
from app.schemas import (
    PaperCreate,
    PaperResponse,
    PaperWithChunks,
    AskRequest,
    AskResponse,
)
from typing import Union
from app.chunker import chunk_text
from app.arxiv import fetch_arxiv_papers
from app.embedder import embed_text, cosine_similarity
from app.llm import get_llm_provider
from app.pdf import extract_text_from_pdf


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

register_tortoise(app, config=TORTOISE_ORM, generate_schemas=True)


# UI
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


# POST /papers: accepts a `PaperCreate` body, creates a `Paper` record, returns a `PaperResponse`
@app.post("/papers/", response_model=PaperResponse)
async def create_paper(paper: PaperCreate):
    if paper.arxiv_id:
        existing = await Paper.get_or_none(arxiv_id=paper.arxiv_id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Paper with arxiv_id '{paper.arxiv_id}' already exists (id={existing.id})",
            )

    paper_obj = await Paper.create(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        year=paper.year,
    )
    return paper_obj


# GET /papers/{paper_id}: retrieves a paper by ID, returns a `PaperResponse`, raises a 404 if not found
@app.get("/papers/{paper_id}", response_model=Union[PaperWithChunks, PaperResponse])
async def get_paper(paper_id: int, include_chunks: bool = False):
    query = Paper.get_or_none(id=paper_id)
    if include_chunks:
        query = query.prefetch_related("chunks")
    paper_obj = await query
    if not paper_obj:
        raise HTTPException(status_code=404, detail="Paper not found")
    if include_chunks:
        return PaperWithChunks.model_validate(paper_obj)
    return PaperResponse.model_validate(paper_obj)


# POST /papers/{paper_id}/chunks: Triggers chunking of paper's abstract
@app.post("/papers/{paper_id}/chunks")
async def chunk_paper(paper_id: int, background_tasks: BackgroundTasks):
    paper_obj = await Paper.get_or_none(id=paper_id)
    if not paper_obj:
        raise HTTPException(status_code=404, detail="Paper not found")

    async def do_chunking():
        text_to_chunk = paper_obj.full_text or paper_obj.abstract
        chunks = chunk_text(text_to_chunk)
        for i, text in enumerate(chunks):
            await Chunk.create(paper=paper_obj, text=text, index=i)

    background_tasks.add_task(do_chunking)
    return {"message": "Chunking started", "paper_id": paper_id}


# GET /papers/{paper_id}/chunks: Returns all chunks for a paper as a list of `ChunkResponse` models
@app.get("/papers/{paper_id}/chunks")
async def get_chunks(paper_id: int):
    paper_obj = await Paper.get_or_none(id=paper_id).prefetch_related("chunks")
    if not paper_obj:
        raise HTTPException(status_code=404, detail="Paper not found")

    chunks = await paper_obj.chunks.all().values("id", "text", "index", "paper_id")
    return {"paper_id": paper_id, "chunks": chunks}


# POST /ingest
@app.post("/ingest")
async def ingest_arxiv(query: str = "CO2 Electroreduction", max_results: int = 5):
    """
    Accepts ArxivSearchRequest, fetches papers from arXiv API, stores each as a `Paper`, chunks the abstract, embeds each chunk, stores each `Embedding`. Returns count of papers and chunks ingested. Skip papers that already exist by `arxiv_id`

    """
    papers = await fetch_arxiv_papers(query=query, max_results=max_results)

    ingested_papers = 0
    ingested_chunks = 0
    skipped_papers = 0

    # Fetch papers from arXiv API, then for each paper:
    for paper in papers:
        existing = await Paper.get_or_none(arxiv_id=paper["arxiv_id"])
        if existing:
            print(
                f"Skipping existing paper with arxiv_id '{paper['arxiv_id']}' (id={existing.id})"
            )
            skipped_papers += 1
            continue

        paper_obj = await Paper.create(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            abstract=paper["abstract"],
            authors=paper["authors"],
            year=paper["year"],
        )
        ingested_papers += 1

        # Chunk the abstract and create `Chunk` records
        chunks = chunk_text(paper.get("full_text") or paper["abstract"])
        for i, text in enumerate(chunks):
            chunk_obj = await Chunk.create(paper=paper_obj, text=text, index=i)
            ingested_chunks += 1
            # Embed each chunk and create `Embedding` records (since ingest is write op)
            vector = embed_text(text)
            await Embedding.create(chunk=chunk_obj, vector=vector)

    return {
        "ingested_papers": ingested_papers,
        "ingested_chunks": ingested_chunks,
        "skipped_papers": skipped_papers,
    }


# POST /papers/{paper_id}/embed: Triggers embedding of all chunks for a paper
@app.post("/papers/{paper_id}/embed")
async def embed_paper_chunks(paper_id: int, background_tasks: BackgroundTasks):
    """
    Embeds all chunks for a specific paper (in case you want to embed a manually added paper). Returns count of embeddings created
    """
    paper_obj = await Paper.get_or_none(id=paper_id).prefetch_related("chunks")
    if not paper_obj:
        raise HTTPException(status_code=404, detail="Paper not found")

    async def do_embedding():
        chunks = await paper_obj.chunks.all()
        for chunk in chunks:
            vector = embed_text(chunk.text)
            await Embedding.create(chunk=chunk, vector=vector)

    background_tasks.add_task(do_embedding)
    return {"message": "Embedding started", "paper_id": paper_id}


# POST /search
@app.post("/search")
async def search_papers(query: str, top_k: int = 5):
    """
    Accepts a `SearchRequest`. Embeds the query, loads all chunks with their embeddings using `prefetch_related("embedding")`, computes cosine similarity betweeen query and each chunk, returns `top_k` resuls sorted by score as a list of `SearchResult`.
    """
    query_vector = embed_text(query)

    # Load all chunks with their embeddings
    chunks = await Chunk.all().prefetch_related("embedding")

    # Compute cosine similarity between query and each chunk
    results = []
    for chunk in chunks:
        if not chunk.embedding:
            continue  # skip chunks that haven't been embedded yet
        score = cosine_similarity(query_vector, chunk.embedding.vector)
        results.append(
            {
                "paper_id": chunk.paper_id,
                "chunk_id": chunk.id,
                "chunk_text": chunk.text,
                "score": score,
            }
        )

    # Return top_k results sorted by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "results": results[:top_k]}


# POST /ask
@app.post("/ask")
async def ask_question(question: AskRequest):
    """
    1. Embed the question (Using `embed_text()`)
    2. Retrieve top_k most relevant chunks using cosine similarity (similar to /search)
    3. Build a context string from the retrieved chunks::

    ```python
    context = "\n\n".join([
        f"[{i+1}] {result.text}"
        for i, result in enumerate(results)
    ])
    ```
    4. Build a system prompt::

    ```txt
    You are a scientific research assistant. Answer the user's question based
    only on the provided context. If the answer is not in the context, say so.
    Always cite which source [1], [2] etc. you are drawing from.
    ```

    5. Build the user prompt::

    ```txt
    Context:
    {context}

    Question: {question}
    ```
    6. Call `get_llm_provider().complete(system, user)` to get the answer from the LLM
    7. Return `AskResponse` with the answer and the source chunks
    """
    top_k = question.top_k
    question = question.question
    # 1. Embed question
    query_vector = embed_text(question)
    # 2. Retrieve top_k relevant chunks using cosine similarity
    chunks = await Chunk.all().prefetch_related("embedding", "paper")
    results = []
    for chunk in chunks:
        if not chunk.embedding:
            continue
        score = cosine_similarity(query_vector, chunk.embedding.vector)
        results.append(
            {
                "paper_id": chunk.paper_id,
                "chunk_id": chunk.id,
                "paper_title": chunk.paper.title,
                "test": chunk.text,
                "score": score,
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k]
    # 3. Build context string
    context = "\n\n".join(
        [f"[{i+1}] {result['test']}" for i, result in enumerate(top_results)]
    )
    # 4. Build system prompt
    system_prompt = """
        You are a scientific research assistant. Answer the user's question based only on the provided context. If the answer is not in the context, say so. Always cite which source [1], [2] etc. you are drawing from.
        """
    # 5. Build user prompt
    user_prompt = f"""Context: {context} \nQuestion: {question}"""
    # 6. Call LLM provider
    llm = await get_llm_provider().complete(system_prompt, user_prompt)
    # 7. Return response
    return AskResponse(answer=llm, source_chunks=top_results)


# POST /upload-pdf
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a file upload (`UploadFile` in FastAPI), extracts text using `extract_test_from_pdf()`, and creates a new `Paper` record (title from filename, abstract from first 1000 chars of extracted test, full text somwhere else), Chunks the full text (not just the abstract this time), embeds all chunks, and retuns `paper_id` and chunk count.
    """
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    title = file.filename
    abstract = text[:1000]  # first 1000 chars as abstract
    paper_obj = await Paper.create(
        title=title, abstract=abstract, authors=[], year=0, full_text=text
    )
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        chunk_obj = await Chunk.create(paper=paper_obj, text=chunk, index=i)
        vector = embed_text(chunk)
        await Embedding.create(chunk=chunk_obj, vector=vector)
    return {"paper_id": paper_obj.id, "chunk_count": len(chunks)}
