import numpy as np
import pandas as pd


class MyLineReg():
    def __init__(self, n_iter, learning_rate, weights: pd.Series = None):  # type: ignore
        self.n_inter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_inter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose):
        X.insert(0, 'x0', 1)

        weights = pd.Series(np.ones(X.shape[1]), index=X.columns)

        features_count = X.shape[0]

        if (verbose > 0):
            print(f'start | ${self._loss_function(y, X.dot(weights))}')

        for step in range(self.n_inter):
            predicted_values = X.dot(weights)
            mseGrad = (predicted_values - y).dot(X).mul(2).div(features_count)

            weights += mseGrad.mul(-1).mul(self.learning_rate)
            self.weights = weights

            if (verbose > 0 and step % verbose == 0):
                print(f'start | ${self._loss_function(y, X.dot(weights))}')

    def predict(self, X: pd.DataFrame) -> float:
        X.insert(0, 'x0', 1)
        predicted_values = X.dot(self.weights)
        return sum(predicted_values)

    def get_coef(self):
        return self.weights.values[1:]

    def _loss_function(self, y: pd.Series, predicted):
        mse = sum(((predicted - y) ** 2).div(y.shape[0]))
        return mse
