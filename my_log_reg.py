import random
import pandas as pd
import numpy as np


class MyLogReg():

    _first_feature_col_name = 'x0'
    _eps = 1e-15

    def __init__(self,
                 learning_rate,
                 n_iter: int = 10,
                 weights: pd.Series = None,  # type: ignore
                 metric: str = None,  # type: ignore
                 reg: str = None,  # type: ignore
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 sgd_sample: float = None,  # type: ignore
                 random_state: int = 42):  # type: ignore
        self.n_inter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self._best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_inter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose):
        random.seed(self.random_state)

        X = X.copy()
        X.insert(0, MyLogReg._first_feature_col_name, 1)
        weights = pd.Series(np.ones(X.shape[1]), index=X.columns)

        self._print_education_score_if_need(
            y, X, weights, verbose, 'start')

        for step in range(1, self.n_inter + 1):
            grad = self._grad(y, X, weights)
            weights += grad * -1 * self._get_learning_rate(step)
            self.weights = weights

            if self.metric:
                self._best_score = self._calculate_metric(y, X)

            self._print_education_score_if_need(y, X, weights, verbose, step)

    def predict_proba(self, X: pd.DataFrame):
        X = X.copy()

        if MyLogReg._first_feature_col_name not in X.columns:
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

    def get_best_score(self):
        return self._best_score

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

    def _get_learning_rate(self, iter_step: int) -> float:
        if callable(self.learning_rate):
            return self.learning_rate(iter_step)  # type: ignore
        else:
            return self.learning_rate

    def _print_education_score_if_need(self, y: pd.Series,  X: pd.DataFrame, wg: pd.Series, verbose, step):
        if not verbose:
            return

        if type(step) is int and int(step) % verbose != 0:
            return

        prnt_str = f'{step}|loss {self._loss(X, y, wg)}'

        if self.metric:
            prnt_str += f'|{self.metric}: {self._best_score}'

        print(prnt_str)

    def _grad(self, y: pd.Series, X: pd.DataFrame, wg: pd.Series) -> pd.Series:
        step_learn_rows_indexes = self._get_learn_dataset_indexes(X.shape[0])

        x_buff = X.iloc[step_learn_rows_indexes]
        y_buff = pd.Series(
            y.values[step_learn_rows_indexes], index=y.index[step_learn_rows_indexes])

        y_predicted = x_buff.dot(wg)
        grad = MyLogReg._logloss_derivative(y_buff, y_predicted, x_buff)

        if self.reg:
            grad += self._regularization_der(self.reg, wg)

        return grad

    def _loss(self, X: pd.DataFrame, y: pd.Series, wg: pd.Series) -> float:
        y_predicted = X.dot(wg)
        loss = MyLogReg._logloss(y, y_predicted)

        if (self.reg):
            loss += self._regularization(self.reg, wg)

        return loss

    def _calculate_metric(self, y: pd.Series, X: pd.DataFrame) -> float:
        if not self.metric:
            raise ValueError('metric')

        if self.metric == 'roc_auc':
            y_predicted_score = self.predict_proba(X)
            y_predicted_score = y_predicted_score.round(10)
            return MyLogReg._metric_roc_auc(y, y_predicted_score)

        metrics = {
            'accuracy': MyLogReg._metric_accuracy,
            'precision': MyLogReg._metric_precision,
            'recall': MyLogReg._metric_recall,
            'f1': MyLogReg._metric_f1
        }

        y_predicted_class = self.predict(X)
        return metrics[self.metric](y, y_predicted_class)

    def _regularization_der(self, reg: str, vec: pd.Series) -> pd.Series:
        if reg == 'l1':
            return MyLogReg._l1_der(vec)*self.l1_coef
        elif reg == 'l2':
            return MyLogReg._l2_der(vec)*self.l2_coef
        elif reg == 'elasticnet':
            return MyLogReg._elastic_net_der(vec, self.l1_coef, self.l2_coef)
        else:
            raise ValueError()

    def _regularization(self, reg: str, vec: pd.Series) -> float:
        if reg == 'l1':
            return MyLogReg._l1(vec)*self.l1_coef
        elif reg == 'l2':
            return MyLogReg._l2(vec)*self.l2_coef
        elif reg == 'elasticnet':
            return MyLogReg._elastic_net(vec, self.l1_coef, self.l2_coef)
        else:
            raise ValueError()

    @staticmethod
    def _metric_f1(y: pd.Series, y_predicted: pd.Series) -> float:
        precision = MyLogReg._metric_precision(y, y_predicted)
        recall = MyLogReg._metric_recall(y, y_predicted)

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def _metric_roc_auc(y: pd.Series, y_predicted_score: pd.Series) -> float:
        df = pd.DataFrame({
            'score': y_predicted_score,
            'cls': y
        })

        df = df.sort_values(['score', 'cls'], ascending=[False, False])

        positive_counter_upper_score = 0
        roc_auc = 0
        current_score = -1
        positive_counter_current_score = 0
        for index, row in df.iterrows():
            if row.score != current_score:
                positive_counter_upper_score += positive_counter_current_score
                positive_counter_current_score = 0
                current_score = row.score

            if row.cls == 1:
                positive_counter_current_score += 1
            else:
                roc_auc += positive_counter_upper_score + positive_counter_current_score / 2

        return (1 / ((y == 1).sum() * (y == 0).sum())) * roc_auc

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

    @staticmethod
    def _l1(vec: pd.Series) -> float:
        return sum(vec.apply(lambda x: abs(x)))

    @staticmethod
    def _l1_der(vec: pd.Series) -> pd.Series:
        return vec.apply(MyLogReg._sgn)

    @staticmethod
    def _l2(vec: pd.Series) -> float:
        return sum(vec ** 2)

    @staticmethod
    def _l2_der(vec: pd.Series) -> pd.Series:
        return 2 * vec

    @staticmethod
    def _elastic_net(vec: pd.Series, l1_coef: float, l2_coef: float) -> float:
        return MyLogReg._l1(vec) * l1_coef + MyLogReg._l2(vec) * l2_coef

    @staticmethod
    def _elastic_net_der(vec: pd.Series, l1_coef: float, l2_coef: float) -> pd.Series:
        return l1_coef * MyLogReg._l1_der(vec) + l2_coef * MyLogReg._l2_der(vec)

    @staticmethod
    def _sgn(num: float):
        if num < 0:
            return -1
        elif num > 0:
            return 1
        else:
            return 0
