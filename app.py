from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from jitter import JitterEvaluator, get_headlines
import pandas as pd
import ast, datetime


class Status(BaseModel):
    last_trained: datetime.datetime
    last_updated: datetime.datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global status

    print("Loading model.")
    engine = JitterEvaluator()

    print("Training.")
    training = pd.read_csv("training.csv")
    training["embedding"] = training.embedding.apply(ast.literal_eval)

    engine.train(training)
    train_time = datetime.datetime.now()

    print("Begin predictions.")
    urls = pd.read_csv("sources.csv").url
    headlines = get_headlines(urls, date="today")
    engine.process_headlines(headlines)

    update_time = datetime.datetime.now()

    status = Status(last_trained=train_time, last_updated=update_time)

    print("Engine ready.")

    yield

    del engine


app = FastAPI(lifespan=lifespan)


class Summary(BaseModel):
    status: Status
    mean: float
    std: float


@app.get("/summary/")
async def summary() -> Summary:
    return Summary(status=status, mean=engine.mean, std=engine.std)


class Distribution(BaseModel):
    status: Status
    distribution: list[float]

@app.get("/distribution/")
async def distribution() -> Distribution:
    return Distribution(status=status, distribution=list(engine.distribution))
