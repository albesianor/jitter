"""ML engine for filtering and scoring."""

import os

os.environ["KERAS_BACKEND"] = "torch"

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

    def embed(self, sequence: str | np.ndarray) -> np.ndarray:
        """
        Embed sequence(s).

        Args:
            sequence(s) (str | np.ndarray): the sequence to embed
        Returns:
            numpy.ndarray: the embedding as a list of floats
        """
        return self._embedder.encode(sequence)

    def filter(self, embedding: np.ndarray) -> np.ndarray:
        """
        Filter embedded sequence(s) by relevance.

        Args:
            embedding (numpy.ndarray): the embedding(s) of the sequence(s)
        Returns:
            numpy.ndarray: 0 if not relevant, 1 if relevant
        """
        return self._filter.predict(embedding)

    def score(self, embedding: np.ndarray) -> np.ndarray:
        """
        Score embedded sequence(s) by jitteriness.

        Args:
            embedding (numpy.ndarray): the embedding(s) of the sequence(s)
        Returns:
            numpy.ndarray: the score(s) between 0 (min) and 1 (max)
        """
        return self._scorer.predict(embedding)
