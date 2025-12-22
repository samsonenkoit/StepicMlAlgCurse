import numpy as np
import pandas as pd


class MyLineReg():

    def __init__(self, n_iter, learning_rate, weights: pd.Series = None, metric: str = None):  # type: ignore
        self.n_inter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_inter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose):
        X.insert(0, 'x0', 1)

        weights = pd.Series(np.ones(X.shape[1]), index=X.columns)

        features_count = X.shape[0]

        if (verbose > 0):
            self._print_education_score(
                y, X, weights, verbose, 'start', self.metric)

        for step in range(self.n_inter):
            predicted_values = X.dot(weights)
            mseGrad = (predicted_values - y).dot(X).mul(2).div(features_count)

            weights += mseGrad.mul(-1).mul(self.learning_rate)
            self.weights = weights

            if (verbose > 0 and step % verbose == 0):
                self._print_education_score(
                    y, X, weights, verbose, str(step), self.metric)

            self._best_score = MyLineReg._calculate_metric(
                self.metric, y, X.dot(self.weights))

    def predict(self, X: pd.DataFrame) -> float:
        X.insert(0, 'x0', 1)
        predicted_values = X.dot(self.weights)
        return sum(predicted_values)

    def get_best_score(self) -> float:
        return self._best_score

    def get_coef(self):
        return self.weights.values[1:]

    def _print_education_score(self, y: pd.Series,  X: pd.DataFrame, wg: pd.Series, verbose: bool, step: str = 'start', metric: str = None):  # type: ignore
        if not verbose:
            return

        predicted = X.dot(wg)
        prnt_str = f'{step}|loss {MyLineReg._mse(y, predicted)}'

        if metric:
            prnt_str += f'|{metric}: {MyLineReg._calculate_metric(metric, y, predicted)}'

        print(prnt_str)

    @staticmethod
    def _calculate_metric(metric: str, y: pd.Series, predicted: pd.Series) -> float:
        _metrics = {
            'mae': MyLineReg._mae,
            'mse': MyLineReg._mse,
            'rmse': MyLineReg._rmse,
            'mape': MyLineReg._mape,
            'r2': MyLineReg._r2
        }
        return _metrics[metric](y, predicted)

    @staticmethod
    def _rmse(y: pd.Series, predicted: pd.Series) -> float:
        return MyLineReg._mse(y, predicted) ** 0.5

    @staticmethod
    def _mae(y: pd.Series, predicted: pd.Series) -> float:
        return sum(abs(predicted - y)) / y.shape[0]

    @staticmethod
    def _mse(y: pd.Series, predicted: pd.Series) -> float:
        val = sum(((predicted - y) ** 2).div(y.shape[0]))
        return val

    @staticmethod
    def _r2(y: pd.Series, predicted: pd.Series) -> float:
        numerator = sum(((predicted - y) ** 2))
        average_y = sum(y) / len(y)
        denominator = sum((y - average_y) ** 2)
        return 1 - (numerator / denominator)

    @staticmethod
    def _mape(y: pd.Series, predicted: pd.Series) -> float:
        return sum(abs(y - predicted) / y) * (100/len(y))
