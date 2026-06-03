import feedparser
import time
import ssl
import pandas as pd


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
