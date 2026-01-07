import pandas as pd
import numpy as np
import random

class MyLogReg():

    _eps = 1e-15

    def __init__(self, n_iter: int = 10,
                 learning_rate: float = 0.1,
                 weight: pd.Series = None): # type: ignore
        self.n_inter = n_iter
        self.learning_rate = learning_rate
        self.weight = weight
    
    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_inter}, learning_rate={self.learning_rate}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose):
        X.insert(0, 'x0', 1)
        weights = pd.Series(np.ones(X.shape[1]), index=X.columns)

        for step in range(0, self.n_inter):
            pass
    
    @staticmethod
    def _loss(X: pd.DataFrame, y: pd.Series, weights: pd.Series) -> float:
        y_predicted = X.dot(weights)
    
    @staticmethod
    def _logloss(y: pd.Series, y_predicted: pd.Series):
        y_predicted.apply(MyLogReg._sigmoid)

    @staticmethod
    def _sigmoid(val: float) -> float:
        return 1 / (1 + np.exp(-val))