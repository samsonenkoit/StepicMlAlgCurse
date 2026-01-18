import random
import pandas as pd
import numpy as np


class MySVM():

    def __init__(self,
                 learning_rate=0.001,
                 n_iter: int = 10,
                 weights: pd.Series = None,  # type: ignore
                 b: float = None,  # type: ignore
                 C: float = 1,
                 sgd_sample: float = None,  # type: ignore
                 random_state: int = 42
                 ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def predict(self, X: pd.DataFrame):
        predicted = (X.dot(self.weights) + self.b).apply(np.sign)
        return predicted.apply(lambda x: 0 if x < 0 else 1)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        random.seed(self.random_state)
        y = y.apply(lambda x: 1 if x == 1 else -1)

        self.weights = pd.Series(np.ones(X.shape[1]), index=X.columns)
        self.b = 1

        self._print_education_score_if_need(
            y, X, self.weights, verbose, 'start')

        for step in range(0, self.n_iter):
            dataset_indexes = self._get_learn_dataset_indexes(X.shape[0])
            x_buff = X.iloc[dataset_indexes]
            y_buff = pd.Series(
                y.values[dataset_indexes], index=y.index[dataset_indexes])

            for (_, x_feature), y_class in zip(x_buff.iterrows(), y_buff):
                grad = self._grad(x_feature, y_class)
                self.weights += grad[0] * -1 * self.learning_rate
                self.b += grad[1] * -1 * self.learning_rate

            self._print_education_score_if_need(
                y, X, self.weights, verbose, step)

    def get_coef(self):
        return (self.weights, self.b)

    def _get_learn_dataset_indexes(self, dataset_len: int):
        dataset_indexes = list(range(0, dataset_len))
        if not self.sgd_sample:
            return dataset_indexes

        sample_count = 1
        if self.sgd_sample < 1:
            sample_count = int(max(1, dataset_len * self.sgd_sample))
        else:
            sample_count = int(self.sgd_sample)

        dataset_indexes = random.sample(dataset_indexes, sample_count)
        return dataset_indexes

    def _grad(self, x_feature: pd.Series, y_class: int):
        if y_class * (x_feature.dot(self.weights) + self.b) >= 1:
            return (self.weights * 2, 0)
        else:
            return (self.weights * 2 - x_feature * y_class * self.C, -y_class * self.C)

    def _loss(self, X: pd.DataFrame, y: pd.Series, wg: pd.Series) -> float:
        predicted = X.dot(wg)
        loss = sum(((predicted + self.b) *
                   y).apply(lambda x: max(0, x))) / y.shape[0]
        return sum(y ** 2) + loss

    def _print_education_score_if_need(self, y: pd.Series,  X: pd.DataFrame, wg: pd.Series, verbose, step):
        if not verbose:
            return

        if type(step) is int and int(step) % verbose != 0:
            return

        prnt_str = f'{step}|loss {self._loss(X, y, wg)}'

        """ if self.metric:
            prnt_str += f'|{self.metric}: {self._best_score}' """

        print(prnt_str)
