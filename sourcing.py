import feedparser
import pandas as pd

def from_urls(urls: list) -> list:
    """
    Parses a list of RSS feeds provided as urls.

    Args:
        urls (list[str]): A list of RSS feed urls.

    Returns:
        pandas.DataFrame: A table of title, description, timestamp, source.
    """
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