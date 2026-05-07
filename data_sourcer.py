import feedparser
import pandas as pd

def from_urls(urls: list) -> list:
    """
    Takes a list of RSS feeds urls and returns a pandas dataframe
    of parsed feed entries. Each entry consists of:
      title, description, timestamp, source
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