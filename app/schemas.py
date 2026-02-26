r"""
Defines two Pydantic models:

- `PaperCreate`: what the API accepts in a POST request body (`title`, `authors`,`year). No `id`, `abstract`, or `created_at` because those are generated server-side.
- `PaperResponse`: what the API *returns*. Includes the full set of fields. `from_attributes = True` lets Pydantic read directly from a Tortoise ORM object instead of requiring a plain dict.
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


# Validates incoming POST request body
class PaperCreate(BaseModel):
    arxiv_id: Optional[str] = None
    title: str
    abstract: Optional[str] = "This is a placeholder abstract."  # Optional with default
    authors: list[str]
    year: int


# Shapes API response
class PaperResponse(BaseModel):
    id: int
    arxiv_id: Optional[str] = None
    title: str
    abstract: str
    authors: list[str]
    year: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# For chunk-related endpoints
class ChunkResponse(BaseModel):
    id: int
    text: str
    index: int
    paper_id: int

    model_config = ConfigDict(from_attributes=True)


# For GET /papers/{id} with chunks
class PaperWithChunks(BaseModel):
    id: int
    arxiv_id: Optional[str] = None
    title: str
    abstract: str
    authors: list[str]
    year: int
    created_at: datetime
    chunks: list[ChunkResponse]

    model_config = ConfigDict(from_attributes=True)
