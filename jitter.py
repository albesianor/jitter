import feedparser
import time
import ssl
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV


class JitterEvaluator:
    """The main engine for evaluating jitterness of headlines"""

    def __init__(self, embedding_model: str = "ProsusAI/finbert"):
        """
        Args:
            embedding_model (str): HuggingFace model identifier
        """
        self._embedder = SentenceTransformer(embedding_model)
        self._filter = LogisticRegressionCV(cv=5)
        self._scorer = LogisticRegressionCV(cv=5)
        self._current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jittery"]
        )
        self._total_jitter = None

    def train(self, df: pd.DataFrame):
        """
        Train the filter and scorer models.

        Args:
            df (pd.DataFrame): Pandas dataframe with (at least) columns `embeddings`, `relevant`, `jittery`
        """
        self._filter.fit(pd.DataFrame(df.embedding.to_list()), df.relevant)

        df = df[df.relevant == 1]
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        self._scorer.fit(pd.DataFrame(df.embedding.to_list()), df.jittery)

    def process_headlines(self, headlines: pd.Series):
        """
        Process the headlines and store the result in the `JitterEvaluator` object.

        Args:
            headlines (pd.Series): List of headlines as string.
        """
        self._current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jittery"]
        )
        self._current["concat"] = headlines
        self._current["embedding"] = self._embedder.encode(
            self._current.concat.to_list()
        ).tolist()

        self._current["relevant"] = self._filter.predict(
            pd.DataFrame(self._current.embedding.to_list())
        )
        self._current["jittery"] = self._current.apply(
            lambda x: (
                self._scorer.predict(np.array(x.embedding).reshape(1, -1))
                if x.relevant == 1
                else None
            ),
            axis=1,
        )

        self._total_jitter = self._current.jittery.dropna(inplace=False).mean()

    @property
    def current_prediction(self) -> pd.DataFrame:
        """
        Returns:
            The complete current prediction dataset.
        """
        return self._current

    @property
    def total_jitter(self) -> float:
        """
        Returns:
            The current total jitter (fraction of jittery headlines over total).
        """
        return self._total_jitter

    def random_headline(self) -> pd.DataFrame:
        """
        Returns:
            A random headline.
        """
        return self._current.sample()


def get_headlines(urls: list[str], date: str = None) -> pd.Series:
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
