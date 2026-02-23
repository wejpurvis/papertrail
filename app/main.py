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

from fastapi import BackgroundTasks, FastAPI, HTTPException
from tortoise.contrib.fastapi import register_tortoise
from app.database import TORTOISE_ORM
from app.models import Chunk, Paper
from app.schemas import PaperCreate, PaperResponse, PaperWithChunks
from typing import Union
from app.chunker import chunk_text


app = FastAPI()

register_tortoise(app, config=TORTOISE_ORM, generate_schemas=True)


# POST /papers: accepts a `PaperCreate` body, creates a `Paper` record, returns a `PaperResponse`
@app.post("/papers/", response_model=PaperResponse)
async def create_paper(paper: PaperCreate):
    paper_obj = await Paper.create(
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
        chunks = chunk_text(paper_obj.abstract)
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
