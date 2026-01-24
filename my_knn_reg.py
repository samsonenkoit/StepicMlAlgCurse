import random
import pandas as pd
import numpy as np


class MyKNNReg():
    def __init__(self, train_size=None, k=3) -> None:
        self.k = k
        self.train_size = train_size

    def __str__(self) -> str:
        return f'MyKNNReg class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = (X.shape[0], X.shape[1])

    def predict(self, X: pd.DataFrame):
        predicts = []
        for _, row in X.iterrows():
            distances = ((self.X - row) ** 2).sum(axis=1) ** 0.5
            distances = distances.sort_values(ascending=True).iloc[:self.k]
            vals = self.y[distances.index]
            predicts.append(vals.mean())
        
        return pd.Series(predicts)

