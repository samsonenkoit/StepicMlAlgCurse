import pandas as pd
import numpy as np


class MyLogReg():

    _first_feature_col_name = 'x0'
    _eps = 1e-15

    def __init__(self, n_iter: int = 10,
                 learning_rate: float = 0.1,
                 weights: pd.Series = None):  # type: ignore
        self.n_inter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_inter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose):
        X = X.copy()
        X.insert(0, MyLogReg._first_feature_col_name, 1)
        weights = pd.Series(np.ones(X.shape[1]), index=X.columns)

        self._print_education_score_if_need(
            y, X, weights, verbose, 'start')

        for step in range(0, self.n_inter):
            grad = self._grad(y, X, weights)
            weights += grad * -1 * self.learning_rate
            self.weights = weights

            self._print_education_score_if_need(y, X, weights, verbose, step)

    def predict_proba(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, MyLogReg._first_feature_col_name, 1)
        predicted = X.dot(self.weights)
        predicted = predicted.apply(MyLogReg._sigmoid)
        return predicted

    def predict(self, X: pd.DataFrame):
        predicted = self.predict_proba(X)
        predicted = predicted.apply(lambda x: 1 if x > 0.5 else 0)
        return predicted

    def get_coef(self):
        return self.weights.values[1:]

    def _print_education_score_if_need(self, y: pd.Series,  X: pd.DataFrame, wg: pd.Series, verbose, step, metric: str = None):
        if not verbose:
            return

        if type(step) is int and int(step) % verbose != 0:
            return

        prnt_str = f'{step}|loss {self._loss(X, y, wg)}'

        print(prnt_str)

    def _grad(self, y: pd.Series, X: pd.DataFrame, wg: pd.Series) -> pd.Series:
        y_predicted = X.dot(wg)
        grad = MyLogReg._logloss_derivative(y, y_predicted, X)
        return grad

    def _loss(self, X: pd.DataFrame, y: pd.Series, wg: pd.Series) -> float:
        y_predicted = X.dot(wg)
        return MyLogReg._logloss(y, y_predicted)

    def _calculate_metric(self, metric: str, y: pd.Series, y_predicted_score: pd.Series,) -> float:
        if not metric:
            raise ValueError('metric')

        metrics = {
            'accuracy': MyLogReg._metric_accuracy,
            'precision': MyLogReg._metric_precision,
            'recall': MyLogReg._metric_recall,
            'f1': MyLogReg._metric_f1
        }

        y_predicted_class =
        return metrics[metric](y, y_predicted_class)

    @staticmethod
    def _metric_f1(y: pd.Series, y_predicted: pd.Series) -> float:
        precision = MyLogReg._metric_precision(y, y_predicted)
        recall = MyLogReg._metric_recall(y, y_predicted)

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def _metric_accuracy(y: pd.Series, y_predicted: pd.Series) -> float:
        tp = MyLogReg._true_positive(y, y_predicted)
        tn = MyLogReg._true_negative(y, y_predicted)

        return (tp + tn) / (tp + tn + MyLogReg._false_positive(y, y_predicted) + MyLogReg._false_negative(y, y_predicted))

    @staticmethod
    def _metric_precision(y: pd.Series, y_predicted: pd.Series) -> float:
        tp = MyLogReg._true_positive(y, y_predicted)
        return tp / (tp + MyLogReg._false_positive(y, y_predicted))

    @staticmethod
    def _metric_recall(y: pd.Series, y_predicted: pd.Series) -> float:
        tp = MyLogReg._true_positive(y, y_predicted)
        return tp / (tp + MyLogReg._false_negative(y, y_predicted))

    @staticmethod
    def _true_positive(y: pd.Series, y_predicted: pd.Series) -> int:
        return (y & y_predicted).sum()

    @staticmethod
    def _true_negative(y: pd.Series, y_predicted: pd.Series) -> int:
        return ((y == 0) & (y_predicted == 0)).sum()

    @staticmethod
    def _false_positive(y: pd.Series, y_predicted: pd.Series) -> int:
        return ((y_predicted == 1) & (y == 0)).sum()

    @staticmethod
    def _false_negative(y: pd.Series, y_predicted: pd.Series) -> int:
        return ((y_predicted == 0) & (y == 1)).sum()

    @staticmethod
    def _logloss(y: pd.Series, y_predicted: pd.Series) -> float:
        y_predicted = y_predicted.apply(MyLogReg._sigmoid)

        result = ((y * y_predicted.apply(lambda x: np.log(x + MyLogReg._eps)) + (1 - y) *
                  y_predicted.apply(lambda x: np.log(1 - x + MyLogReg._eps))) * -1) / y.shape[0]
        return sum(result)

    @staticmethod
    def _logloss_derivative(y: pd.Series, y_predicted: pd.Series, X: pd.DataFrame) -> pd.Series:
        return (y_predicted.apply(MyLogReg._sigmoid) - y).dot(X) / y.shape[0]

    @staticmethod
    def _sigmoid(val: float) -> float:
        return 1 / (1 + np.exp(-val))
