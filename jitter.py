import os
os.environ["KERAS_BACKEND"] = "jax"

import feedparser
import time
import ssl
import keras
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
        # embedder
        self._embedder = SentenceTransformer(embedding_model)

        # models
        self._filter = LogisticRegressionCV(cv=5, class_weight="balanced")
        self._scorer = keras.Sequential(
            layers=[
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self._scorer.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mae"])

        # data
        self._current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jitter"]
        )
        self._mean = None
        self._std = None

    def train(self, df: pd.DataFrame):
        """
        Train the filter and scorer models.

        Args:
            df (pd.DataFrame): Pandas dataframe with (at least) columns `embedding`, `relevant`, `jitter`
        """
        assert len(df.embedding[0]) == self._embedder.get_embedding_dimension()

        # filter
        self._filter.fit(pd.DataFrame(df.embedding.to_list()), df.relevant)

        # scorer
        df = df[df.relevant == 1]
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        X = np.stack(df.embedding.values)
        y = df.jitter.values[:, np.newaxis]

        self._scorer.fit(X, y, batch_size=128, epochs=7, validation_split=0.2, verbose=0)

    def process_headlines(self, headlines: pd.Series):
        """
        Process the headlines and store the result in the `JitterEvaluator` object.

        Args:
            headlines (pd.Series): List of headlines as string.
        """
        self._current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jitter"]
        )
        self._current["concat"] = headlines
        self._current["embedding"] = self._embedder.encode(
            self._current.concat.to_list()
        ).tolist()

        self._current["relevant"] = self._filter.predict(
            pd.DataFrame(self._current.embedding.to_list())
        )
        self._current["jitter"] = self._current.apply(
            lambda x: (
                self._scorer.predict(np.array(x.embedding).reshape(1, -1), verbose=0)[0, 0]
                if x.relevant == 1
                else None
            ),
            axis=1,
        )

        self._mean = self._current.jitter.mean()
        self._std = self._current.jitter.std()

    @property
    def current_prediction(self) -> pd.DataFrame:
        """
        Returns:
            The complete current prediction dataset.
        """
        return self._current

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    @property
    def distribution(self) -> pd.Series:
        return self._current.jitter.dropna(inplace=False)

    def random_headline(self) -> pd.DataFrame:
        """
        Returns:
            A random headline.
        """
        return self._current.sample()


def get_headlines(urls: list[str], date: str | None = None) -> pd.Series:
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
