"""Pydantic response models"""

from datetime import datetime
from pydantic import BaseModel

class Status(BaseModel):
    last_trained: datetime | None
    last_updated: datetime | None

class Summary(BaseModel):
    status: Status
    mean: float
    std: float

class Distribution(BaseModel):
    status: Status
    distribution: list[float]

class Headline(BaseModel):
    headline: str
    relevant: bool
    jitter: float | None