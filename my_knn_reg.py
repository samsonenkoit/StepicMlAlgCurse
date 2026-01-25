import random
import pandas as pd
import numpy as np


class MyKNNReg():
    def __init__(self, train_size=None, k=3, metric: str = 'euclidean', weight: str = 'uniform') -> None:
        self.k = k
        self.train_size = train_size
        self.metric = metric
        self.weight = weight

    def __str__(self) -> str:
        return f'MyKNNReg class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = (X.shape[0], X.shape[1])

    def predict(self, X: pd.DataFrame):
        predicts = []
        for _, row in X.iterrows():
            distances = []
            for index, source_row in self.X.iterrows():
                distances.append((index, self._get_metric(row, source_row)))
            
            distances.sort(key= lambda x: x[1], reverse=False)
            predicts.append(self._get_predict(distances))
        
        return pd.Series(predicts)
    
    def _get_predict(self, distances) -> float:
        distances = list(distances[:self.k])
        y_vals = [self.y[i[0]] for i in distances]

        koef = []
        if self.weight == 'uniform':
            koef = list(np.ones(len(y_vals)))
        elif self.weight == 'rank':
            denominator = sum([1/n for n in range(1, len(distances) + 1)])
            for index in range(1, len(distances) + 1):
                koef.append((1/ index) / denominator)
            
        else:
            denominator = sum([1/n[1] for n in distances])
            for i in distances:
                koef.append((1/i[1]) / denominator)
        
        vals = pd.Series(y_vals)
        koef = pd.Series(koef)

        if self.weight == 'uniform':
            return (vals * koef).mean()
        else:
            return (vals * koef).sum()
            

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

