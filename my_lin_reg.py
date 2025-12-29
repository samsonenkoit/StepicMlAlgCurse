import numpy as np
import pandas as pd


class MyLineReg():

    def __init__(self, n_iter,
                 learning_rate,
                 weights: pd.Series = None,  # type: ignore
                 metric: str = None,  # type: ignore
                 reg: str = None,  # type: ignore
                 l1_coef: float = 0,
                 l2_coef: float = 0):
        self.n_inter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_inter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose):
        X.insert(0, 'x0', 1)

        weights = pd.Series(np.ones(X.shape[1]), index=X.columns)

        if (verbose > 0):
            self._print_education_score(
                y, X, weights, verbose, 'start', self.metric)

        for step in range(self.n_inter):
            mseGrad = self._grad(y, X, weights)

            weights += mseGrad.mul(-1).mul(self.learning_rate)
            self.weights = weights

            if (verbose > 0 and step % verbose == 0):
                self._print_education_score(
                    y, X, weights, verbose, str(step), self.metric)

            if self.metric is not None:
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

    def _grad(self, y: pd.Series, X: pd.DataFrame, wg: pd.Series) -> pd.DataFrame:
        features_count = X.shape[0]

        predicted_values = X.dot(wg)
        mseGrad = (predicted_values - y).dot(X).mul(2).div(features_count)

        if self.reg:
            mseGrad += self._regularization_der(self.reg, wg)

        return mseGrad

    def _print_education_score(self, y: pd.Series,  X: pd.DataFrame, wg: pd.Series, verbose: bool, step: str = 'start', metric: str = None):  # type: ignore
        if not verbose:
            return

        predicted = X.dot(wg)
        prnt_str = f'{step}|loss {self._loss(y, predicted, wg)}'

        if metric:
            prnt_str += f'|{metric}: {MyLineReg._calculate_metric(metric, y, predicted)}'

        print(prnt_str)

    def _loss(self, y: pd.Series, predicted: pd.Series, weights: pd.Series) -> float:
        loss = MyLineReg._mse(y, predicted)

        if (self.reg):
            loss += self._regularization(self.reg, weights)

        return loss

    def _regularization_der(self, reg: str, vec: pd.Series) -> pd.Series:
        if reg == 'l1':
            return MyLineReg._l1_der(vec)*self.l1_coef
        elif reg == 'l2':
            return MyLineReg._l2_der(vec)*self.l2_coef
        elif reg == 'elasticnet':
            return MyLineReg._elastic_net_der(vec, self.l1_coef, self.l2_coef)
        else:
            raise ValueError()

    def _regularization(self, reg: str, vec: pd.Series) -> float:
        if reg == 'l1':
            return MyLineReg._l1(vec)*self.l1_coef
        elif reg == 'l2':
            return MyLineReg._l2(vec)*self.l2_coef
        elif reg == 'elasticnet':
            return MyLineReg._elastic_net(vec, self.l1_coef, self.l2_coef)
        else:
            raise ValueError()

    @staticmethod
    def _calculate_metric(metric: str, y: pd.Series, predicted: pd.Series) -> float:
        if metric is None:
            return None
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
        return sum(abs((y - predicted) / y)) * (100/len(y))

    @staticmethod
    def _sgn(num: float):
        if num < 0:
            return -1
        elif num > 0:
            return 1
        else:
            return 0

    @staticmethod
    def _l1(vec: pd.Series) -> float:
        return sum(vec.apply(abs))

    @staticmethod
    def _l1_der(vec: pd.Series) -> pd.Series:
        return vec.apply(MyLineReg._sgn)

    @staticmethod
    def _l2(vec: pd.Series) -> float:
        return sum(vec ** 2)

    @staticmethod
    def _l2_der(vec: pd.Series) -> pd.Series:
        return 2 * vec

    @staticmethod
    def _elastic_net(vec: pd.Series, l1_coef: float, l2_coef: float) -> float:
        return MyLineReg._l1(vec) * l1_coef + MyLineReg._l2(vec) * l2_coef

    @staticmethod
    def _elastic_net_der(vec: pd.Series, l1_coef: float, l2_coef: float) -> pd.Series:
        return l1_coef * MyLineReg._l1_der(vec) + l2_coef * MyLineReg._l2_der(vec)
