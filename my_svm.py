import pandas as pd
import numpy as np


class MySVM():

    def __init__(self,
                 learning_rate=0.001,
                 n_iter: int = 10,
                 weights: pd.Series = None,  # type: ignore
                 b: float = None,  # type: ignore
                 ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b

    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        y = y.apply(lambda x: 1 if x == 1 else -1)

        self.weights = pd.Series(np.ones(X.shape[1]), index=X.columns)
        self.b = 1

        self._print_education_score_if_need(
            y, X, self.weights, verbose, 'start')

        for step in range(0, self.n_iter):
            for (_, x_feature), y_class in zip(X.iterrows(), y):
                grad = self._grad(x_feature, y_class)
                self.weights += grad[0] * -1 * self.learning_rate
                self.b += grad[1] * -1 * self.learning_rate

            self._print_education_score_if_need(
                y, X, self.weights, verbose, step)

    def get_coef(self):
        return (self.weights, self.b)

    def _grad(self, x_feature: pd.Series, y_class: int):
        if y_class * (x_feature.dot(self.weights) + self.b) >= 1:
            return (self.weights * 2, 0)
        else:
            return (self.weights * 2 - x_feature * y_class, -y_class)

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
