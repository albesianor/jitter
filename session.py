import asyncio, ast
from datetime import datetime
import pandas as pd
import numpy as np

from models import Status
from jitter import JitterEvaluator
from routines import get_headlines


class Session:
    """The current predictions"""

    def __init__(self, sources: str | None = None):
        """
        Args:
            sources (str): filename of sources list
        """
        self._engine = JitterEvaluator()
        self._sources = pd.read_csv("sources.csv").url
        self._status = Status(last_trained=None, last_updated=None)

        self.clear()

    def clear(self):
        """
        Clear current headlines and predictions
        """
        self._current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jitter"]
        )
        self._status.last_updated = None

        self._mean = None
        self._std = None

    async def train(self, filename: str | None = None) -> None:
        """
        Trains the engine using the training dataset

        Args:
            filename (str): filename of training dataset (optional)
        """
        if filename is not None:
            training = pd.read_csv(filename)
        else:
            # placeholder for future implementation of training dataset fetcher
            training = pd.read_csv("training.csv")

        training["embedding"] = await asyncio.to_thread(
            training.embedding.apply, ast.literal_eval
        )

        await asyncio.to_thread(self._engine.train, training)

        self._status.last_trained = datetime.now()

    async def fetch_and_process(self) -> None:
        """
        Fetch and process headlines.
        """
        if self._status.last_trained is None:
            return

        headlines = await get_headlines(self._sources, date="today")

        await asyncio.to_thread(self.process_headlines, headlines)

    def process_headlines(self, headlines: pd.Series) -> None:
        """
        Process the headlines and store the result in the `JitterEvaluator` object.

        Args:
            headlines (pd.Series): List of headlines as string.
        """
        if self._status.last_trained is None:
            return

        # only process new headlines
        headlines = headlines[~headlines.isin(self._current.concat)]

        if len(headlines) == 0:
            return

        new_current = pd.DataFrame(
            columns=["concat", "embedding", "relevant", "jitter"]
        )
        new_current["concat"] = headlines
        new_current["embedding"] = self._engine.embed(
            new_current.concat.to_list()
        ).tolist()

        new_current["relevant"] = self._engine.filter(
            pd.DataFrame(new_current.embedding.to_list())
        )
        new_current["jitter"] = new_current.apply(
            lambda x: (
                self._engine.score(np.array(x.embedding).reshape(1, -1))[
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

        self._status.last_updated = datetime.now()

    @property
    def status(self) -> Status:
        return self._status

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