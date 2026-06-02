import ast
import asyncio
import pandas as pd
from jitter import JitterEvaluator, get_headlines

async def train(engine: JitterEvaluator) -> None:
    """
    Trains the engine using `training.csv`

    Args:
        engine (JitterEvaluator): the scoring engine to train
    """
    training = pd.read_csv("training.csv")
    training["embedding"] = await asyncio.to_thread(training.embedding.apply, ast.literal_eval)

    await asyncio.to_thread(engine.train, training)

async def predict(engine: JitterEvaluator) -> None:
    """
    Load headlines from `sources.csv` into engine and compute scores

    Args:
        engine (JitterEvaluator): the scoring engine
    """
    urls = pd.read_csv("sources.csv").url
    headlines = get_headlines(urls, date="today")
    
    await asyncio.to_thread(engine.process_headlines, headlines)