# Deine two Pydantic models: validate incoming POST request body and shape API response


from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# Validates incoming POST request body
class PaperCreate(BaseModel):
    title: str
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
