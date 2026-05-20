import feedparser
import ssl
import pandas as pd

def from_urls(urls: list) -> list:
    """
    Parses a list of RSS feeds provided as urls.

    Args:
        urls (list[str]): A list of RSS feed urls.

    Returns:
        pandas.DataFrame: A table of title, description, timestamp, source.
    """
    # certificates error patch
    if hasattr(ssl, '_create_unverified_context'):
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
                    "title": entry.title,
                    "description": entry.description,
                    "timestamp": entry.published_parsed,
                    # "source": feed.feed.title
                }
                parsed_feed.append(parsed_entry)
            except:
                print("Skipping", entry)


    return pd.DataFrame(parsed_feed).drop_duplicates(subset="title", inplace=False).dropna(inplace=False)

def update_dataset(path: str, dataframe: pd.DataFrame) -> None:
    """
    Utility function to update an existing CSV dataframe by concatenating a new dataframe.

    Args:
        path (str): Path on disk of the existing dataframe.
        dataframe (pandas.DataFrame): The dataframe to concatenate.
    """
    old_df = pd.read_csv(path)
    new_df = pd.concat([old_df, dataframe], ignore_index=True)

    new_df.drop_duplicates(subset="title", inplace=True)
    new_df.dropna(inplace=True)
    new_df.to_csv(path, index=False)