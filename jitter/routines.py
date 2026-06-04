import feedparser
import time
import ssl
import pandas as pd


async def get_headlines(urls: list[str], trim: int) -> pd.Series:
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
                    "timestamp": time.strftime(
                        "%Y-%m-%d-%H-%M-%S", entry.published_parsed
                    ),
                }
                parsed_feed.append(parsed_entry)
            except:
                print("Skipping", entry)

    df = (
        pd.DataFrame(parsed_feed)
        .drop_duplicates(subset="concat", inplace=False)
        .dropna(inplace=False)
    )

    # sort by timestamp
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(inplace=True)

    # only keep the "trim"-most-recent
    to_remove = len(df) - trim
    if to_remove > 0:
        df = df.iloc[to_remove:]
        df.reset_index(inplace=True)

    return df.concat
