import random
import pandas as pd
import numpy as np


class MyKNNClf():
    def __init__(self, k: int = 3,
                 train_size=None) -> None:
        self.k = k
        self.train_size = train_size

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
            distance = np.sqrt(np.sum((row - item) ** 2))
            distances.append((index, distance))

        distances.sort(key=lambda x: x[1])
        return distances[:self.k]
