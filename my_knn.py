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
