from fastapi import FastAPI
from typing import Any
from contextlib import asynccontextmanager

import models
from session import Session


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session

    print("Loading model.")
    session = Session()

    print("Training.")
    await session.train()

    print("Begin predictions.")
    await session.fetch_and_process()

    print("Engine ready.")

    yield

    del session


app = FastAPI(lifespan=lifespan)


@app.get("/summary/", response_model=models.Summary)
async def summary() -> Any:
    return {"status": session.status, "mean": session.mean, "std": session.std}


@app.get("/distribution/", response_model=models.Distribution)
async def distribution() -> Any:
    return {"status": session.status, "distribution": list(session.distribution)}


@app.get("/random/", response_model=models.Headline)
async def random() -> Any:
    headline = session.random_headline()
    return {
        "headline": headline.concat.to_list()[0],
        "relevant": bool(headline.relevant.to_list()[0]),
        "jitter": headline.jitter.to_list()[0],
    }
