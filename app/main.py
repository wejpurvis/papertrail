r"""
Wires everything together with two routes:

- `POST /papers/`: accepts a `PaperCreate` body, creates a `Paper` record, returns a `PaperResponse`
- `GET /papers/{paper_id}`: retrieves a paper by ID, returns a `

The `lifespan` function runs on startup/shutdown - it initialises the Tortoise connection using the config from `database.py`, auto-creates any missing tables (`generate_schemas()`), and then closes connections when the app stops.
"""

from fastapi import FastAPI, HTTPException
from tortoise.contrib.fastapi import register_tortoise
from app.database import TORTOISE_ORM
from app.models import Paper
from app.schemas import PaperCreate, PaperResponse


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
@app.get("/papers/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: int):
    paper_obj = await Paper.get_or_none(id=paper_id)
    if not paper_obj:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper_obj
