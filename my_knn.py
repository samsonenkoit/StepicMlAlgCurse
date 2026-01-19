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
        distance = ((self.X - X) ** 2).sum() ** 0.5
        distance = distance.sort_values()

        y = self.y.reindex(distance.index)
        nearest = y[0:self.k]
        if (nearest == 0).sum() < (nearest == 1).sum():
            return 1
        else:
            return 0

    def predict_proba(self, X: pd.DataFrame):
        distance = ((self.X - X) ** 2).sum() ** 0.5
        distance = distance.sort_values()

        y = self.y.reindex(distance.index)
        nearest = y[0:self.k]

        positive_count = (nearest == 1).sum()
        negative_count = (nearest == 0).sum()
        if (nearest == 0).sum() < (nearest == 1).sum():
            return 1
        else:
            return 0
