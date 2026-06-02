import os

os.environ["KERAS_BACKEND"] = "jax"

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
        self._scorer.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["mae"]
        )

        # data
        self.clear()

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

        self._scorer.fit(
            X, y, batch_size=128, epochs=7, validation_split=0.2, verbose=0
        )

    def process_headlines(self, headlines: pd.Series):
        """
        Process the headlines and store the result in the `JitterEvaluator` object.

        Args:
            headlines (pd.Series): List of headlines as string.
        """

        # only process new headlines
        headlines = headlines[~headlines.isin(self._current.concat)]

        if len(headlines) == 0:
            return

        new_current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jitter"]
        )
        new_current["concat"] = headlines
        new_current["embedding"] = self._embedder.encode(
            new_current.concat.to_list()
        ).tolist()

        new_current["relevant"] = self._filter.predict(
            pd.DataFrame(new_current.embedding.to_list())
        )
        new_current["jitter"] = new_current.apply(
            lambda x: (
                self._scorer.predict(np.array(x.embedding).reshape(1, -1), verbose=0)[
                    0, 0
                ]
                if x.relevant == 1
                else None
            ),
            axis=1,
        )

        # append to old current
        self._current = pd.concat([self._current, new_current], ignore_index=True)

        self._mean = self._current.jitter.mean()
        self._std = self._current.jitter.std()

    def clear(self):
        """
        Clear current headlines and predictions
        """
        self._current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jitter"]
        )
        self._mean = None
        self._std = None

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
