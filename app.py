from fastapi import FastAPI
from typing import Any
from contextlib import asynccontextmanager
import pandas as pd
import datetime

import models
from routines import train, predict
from jitter import JitterEvaluator, get_headlines


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global status

    print("Loading model.")
    engine = JitterEvaluator()

    print("Training.")
    await train(engine)
    train_time = datetime.datetime.now()

    print("Begin predictions.")
    await predict(engine)
    update_time = datetime.datetime.now()

    status = models.Status(last_trained=train_time, last_updated=update_time)

    print("Engine ready.")

    yield

    del engine


app = FastAPI(lifespan=lifespan)


@app.get("/summary/", response_model=models.Summary)
async def summary() -> Any:
    return {
        "status": status,
        "mean": engine.mean,
        "std": engine.std
    }


@app.get("/distribution/", response_model=models.Distribution)
async def distribution() -> Any:
    return {
        "status": status,
        "distribution": list(engine.distribution)
    }


@app.get("/random/", response_model=models.Headline)
async def random() -> Any:
    headline = engine.random_headline()
    return {
        "status": status,
        "id": headline.index.to_list()[0],
        "headline": headline.concat.to_list()[0],
        "relevant": bool(headline.relevant.to_list()[0]),
        "jitter": headline.jitter.to_list()[0]
    }