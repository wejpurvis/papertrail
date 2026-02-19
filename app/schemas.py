r"""
Defines two Pydantic models:

- `PaperCreate`: what the API accepts in a POST request body (`title`, `authors`,`year). No `id`, `abstract`, or `created_at` because those are generated server-side.
- `PaperResponse`: what the API *returns*. Includes the full set of fields. `orm_mode = True` lets Pydantic read directly from a Tortoise ORM object instead of requiring a plain dict.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# Validates incoming POST request body
class PaperCreate(BaseModel):
    title: str
    abstract: Optional[str] = "This is a placeholder abstract."  # Optional with default
    authors: list[str]
    year: int


# Shapes API response
class PaperResponse(BaseModel):
    id: int
    title: str
    abstract: str
    authors: list[str]
    year: int
    created_at: datetime

    class Config:
        orm_mode = True  # Allows Pydantic to read data from ORM objects
