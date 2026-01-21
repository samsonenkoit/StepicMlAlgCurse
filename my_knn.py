import random
import pandas as pd
import numpy as np


class MyKNNClf():
    def __init__(self, k: int = 3,
                 train_size=None,
                 metric: str = 'euclidean') -> None:
        self.k = k
        self.train_size = train_size
        self.metric = metric

    def __str__(self) -> str:
        return f'MyKNNClf class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = (X.shape[0], X.shape[1])

    def predict(self, X: pd.DataFrame):
        probabilities = self.predict_proba(X)
        return probabilities.apply(lambda x: 1 if x >= 0.5 else 0)

    def predict_proba(self, X: pd.DataFrame):
        probabilities = []
        for index, row in X.iterrows():
            neighbors = self._get_nearest_neighbors(row)
            classes = pd.Series([self.y[idx] for idx, _ in neighbors])
            probabilities.append((classes == 1).sum() / classes.shape[0])

        return pd.Series(probabilities)

    def _get_nearest_neighbors(self, item: pd.Series):
        distances = []

        for index, row in self.X.iterrows():
            distance = self._get_metric(row, item)
            distances.append((index, distance))

        distances.sort(key=lambda x: x[1])
        return distances[:self.k]

    def _get_metric(self, a: pd.Series, b: pd.Series):
        funcDict = {
            'euclidean': self._metric_euclidean,
            'manhattan': self._metric_manhattan,
            'chebyshev': self._metric_chebyshev,
            'cosine': self._metric_cosine
        }

        return funcDict[self.metric](a, b)

    @staticmethod
    def _metric_euclidean(a: pd.Series, b: pd.Series) -> float:
        return np.sqrt(np.sum((a - b) ** 2))

    @staticmethod
    def _metric_manhattan(a: pd.Series, b: pd.Series) -> float:
        return (a - b).abs().sum()

    @staticmethod
    def _metric_chebyshev(a: pd.Series, b: pd.Series) -> float:
        return (a - b).abs().max()

    @staticmethod
    def _metric_cosine(a: pd.Series, b: pd.Series) -> float:
        return 1 - (a.dot(b)) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
