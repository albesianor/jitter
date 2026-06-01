from fastapi import FastAPI
from contextlib import asynccontextmanager
from jitter import JitterEvaluator, get_headlines
import pandas as pd
import ast

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    print("Loading model.")
    engine = JitterEvaluator()

    print("Training.")
    training = pd.read_csv("training.csv")
    training["embedding"] = training.embedding.apply(ast.literal_eval)

    engine.train(training)

    print("Begin predictions.")
    urls = pd.read_csv("sources.csv").url
    headlines = get_headlines(urls, date="today")
    engine.process_headlines(headlines)

    print("Engine ready.")

    yield

    del engine


app = FastAPI(lifespan=lifespan)


@app.get("/summary/")
async def summary():
    return {
        "mean": engine.mean,
        "std": engine.std,
    }

@app.get("/distribution/")
async def distribution():
    return list(engine.distribution)