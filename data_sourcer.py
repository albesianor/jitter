import feedparser
import pandas as pd

def from_urls(urls: list) -> list:
    parsed_feed = []

    for url in urls:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            parsed_entry = {
                "title": entry.title,
                "description": entry.description,
                "timestamp": entry.published_parsed,
                "source": feed.feed.title
            }

            parsed_feed.append(parsed_entry)

    return pd.DataFrame(parsed_feed)