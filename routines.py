import ast
import asyncio
import feedparser
import time
import ssl
import pandas as pd
from jitter import JitterEvaluator


async def train(engine: JitterEvaluator, filename: str | None = None) -> None:
    """
    Trains the engine using `training.csv`

    Args:
        engine (JitterEvaluator): the scoring engine to train
        filename (str): filename of training dataset
    """
    if filename is not None:
        training = pd.read_csv(filename)
    else:
        # placeholder for future implementation of training dataset fetcher
        training = pd.read_csv("training.csv")

    training["embedding"] = await asyncio.to_thread(
        training.embedding.apply, ast.literal_eval
    )

    await asyncio.to_thread(engine.train, training)


async def get_headlines(urls: list[str], date: str | None = None) -> pd.Series:
    if hasattr(ssl, "_create_unverified_context"):
        ssl._create_default_https_context = ssl._create_unverified_context

    parsed_feed = []

    for url in urls:
        try:
            feed = feedparser.parse(url)
        except:
            print("Unable to parse feed", url)

        for entry in feed.entries:
            try:
                parsed_entry = {
                    "concat": entry.title + " | " + entry.description,
                    "date": time.strftime("%Y-%m-%d", entry.published_parsed),
                }
                parsed_feed.append(parsed_entry)
            except:
                print("Skipping", entry)

    df = (
        pd.DataFrame(parsed_feed)
        .drop_duplicates(subset="concat", inplace=False)
        .dropna(inplace=False)
    )

    if date:
        if date == "today":
            date = time.strftime("%Y-%m-%d")

        df = df[df.date == date]

    df.reset_index(inplace=True)

    return df.concat


async def predict(engine: JitterEvaluator) -> None:
    """
    Load headlines from `sources.csv` into engine and compute scores

    Args:
        engine (JitterEvaluator): the scoring engine
    """
    urls = pd.read_csv("sources.csv").url
    headlines = await get_headlines(urls, date="today")

    await asyncio.to_thread(engine.process_headlines, headlines)
