from datetime import datetime
from pydantic import BaseModel

class Status(BaseModel):
    last_trained: datetime
    last_updated: datetime

class Summary(BaseModel):
    status: Status
    mean: float
    std: float

class Distribution(BaseModel):
    status: Status
    distribution: list[float]

class Headline(BaseModel):
    status: Status
    id: int
    headline: str
    relevant: bool
    jitter: float | None