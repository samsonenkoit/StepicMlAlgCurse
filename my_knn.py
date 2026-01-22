import random
import pandas as pd
import numpy as np


class MyKNNClf():
    def __init__(self, k: int = 3,
                 train_size=None,
                 metric: str = 'euclidean',
                 weight: str = 'uniform') -> None:
        self.k = k
        self.train_size = train_size
        self.metric = metric
        self.weight = weight

    def __str__(self) -> str:
        return f'MyKNNClf class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = (X.shape[0], X.shape[1])

    def predict(self, X: pd.DataFrame):
        probabilities = self._predict_proba_weights(X)
        predictions = [1 if prob[0] >= prob[1]
                       else 0 for prob in probabilities]

        return pd.Series(predictions)

    def predict_proba(self, X: pd.DataFrame):
        probabilities = self._predict_proba_weights(X)

        return pd.Series([prob[0] for prob in probabilities])

    def _predict_proba_weights(self, X: pd.DataFrame):
        probabilities = []
        for _, row in X.iterrows():
            neighbors = self._get_nearest_neighbors(row)
            weigts = self._get_weights(neighbors)
            probabilities.append((weigts[0],  weigts[1]))

        return probabilities

    def _get_weights(self, neighbors):
        funcDict = {
            'uniform': self._get_weights_uniform,
            'rank': self._get_weights_ranked,
            'distance': self._get_weights_distance
        }

        return funcDict[self.weight](neighbors)

    def _get_weights_uniform(self, neighbors):
        classes = pd.Series([self.y[idx] for idx, _ in neighbors])
        return ((classes == 1).sum() / classes.shape[0], (classes == 0).sum() / classes.shape[0])

    def _get_weights_ranked(self, neighbors):
        classes = [self.y[idx] for idx, _ in neighbors]

        step = 0
        positive_cls = 0
        negative_cls = 0
        denominator = 0
        for cl in classes:
            step += 1
            if cl == 1:
                positive_cls += 1 / step
            else:
                negative_cls += 1 / step
            denominator += 1 / step
        return (positive_cls / denominator, negative_cls / denominator)

    def _get_weights_distance(self, neighbors):
        step = 0
        positive_cls = 0
        negative_cls = 0
        denominator = 0

        for idx, dist in neighbors:
            cl = self.y[idx]
            step += 1
            if dist == 0:
                dist = 1e-5

            if cl == 1:
                positive_cls += 1 / dist
            else:
                negative_cls += 1 / dist
            denominator += 1 / dist
        return (positive_cls / denominator, negative_cls / denominator)

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
