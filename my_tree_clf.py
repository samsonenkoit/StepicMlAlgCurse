import pandas as pd
import numpy as np


class MyTreeClf():
    def __init__(self, max_depth: int, min_samples_split: int, max_leafs: int, bins) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

        self.potential_leafs_cnt = 0
        self.leafs_sum = 0
        self.leafs_cnt = 0

        self.fit_tree = {}
        self.bins = bins
        self.histogram_thresholds = None

    def __str__(self) -> str:
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'

    def predict_proba(self, X: pd.DataFrame):
        return sum(self._predict_proba(X))

    def predict(self, X: pd.DataFrame):
        return sum(self._predict(X))

    def _predict_proba(self, X: pd.DataFrame):
        predict = X.apply(lambda row: self._predict_proba_series(
            row, self.fit_tree), axis=1)

        return predict.to_list()

    def _predict(self, X: pd.DataFrame):
        predicted = self._predict_proba(X)

        return [1 if x > 0.5 else 0 for x in predicted]

    def _predict_proba_series(self, item: pd.Series, node: dict) -> float:
        if node['type'] == 'leaf':
            return float(node['probability'])
        elif item[node['col']] > node['threshold']:
            return self._predict_proba_series(item, node['right'])
        else:
            return self._predict_proba_series(item, node['left'])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.bins:
            self.histogram_thresholds = {}
            for col in X.columns:
                _, thresholds = np.histogram(X[col], bins=self.bins)
                thresholds = thresholds[1:-1]
                self.histogram_thresholds[col] = thresholds

        self.fit_tree = self._fit(X, y, 0)

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.fit_tree
        if node['type'] == 'leaf':
            print(f"{indent}leaf: p={node['probability']:.4f}")
        else:
            print(f"{indent}{node['col']} > {node['threshold']:.4f}")
            print(f"{indent}├── left:")
            self.print_tree(node['left'], indent + "│   ")
            print(f"{indent}└── right:")
            self.print_tree(node['right'], indent + "    ")

    def _fit(self, X: pd.DataFrame, y: pd.Series, current_depth: int) -> dict:
        best_split = get_best_split(X, y, self.bins, self.histogram_thresholds)

        if best_split[0] is None:
            self.leafs_cnt += 1
            buff_p = (y == 1).sum() / len(y)
            self.leafs_sum += buff_p
            return {
                'type': 'leaf',
                'probability': buff_p
            }

        mask = X[best_split[0]] > best_split[1]
        y_right = y[mask]
        y_left = y[~mask]

        node = {
            'type': 'node',
            'col': best_split[0],
            'threshold': best_split[1]
        }

        current_depth += 1

        self.potential_leafs_cnt = self.leafs_cnt + 2
        if self._can_continue_split(len(y_left), current_depth):
            node['left'] = self._fit(X[~mask], y_left, current_depth)
        else:
            node['left'] = {
                'type': 'leaf',
                'probability': (y_left == 1).sum() / len(y_left)
            }
            self.leafs_cnt += 1
            self.leafs_sum += node['left']['probability']

        self.potential_leafs_cnt = self.leafs_cnt + 2

        if self._can_continue_split(len(y_right), current_depth):
            node['right'] = self._fit(X[mask], y_right, current_depth)
        else:
            node['right'] = {
                'type': 'leaf',
                'probability': (y_right == 1).sum() / len(y_right)
            }
            self.leafs_cnt += 1
            self.leafs_sum += node['right']['probability']

        return node

    def _can_continue_split(self, items_count: int, depth: int):
        return depth < self.max_depth and self.potential_leafs_cnt <= self.max_leafs and self.min_samples_split <= items_count


def get_thresholds(col, x_feature: pd.Series, bins, histogram_thresholds):
    if not bins or len(x_feature) - 1 < bins - 1:
        sorted_values = x_feature.sort_values().unique()
        thresholds = (sorted_values[1:] + sorted_values[:-1]) / 2
        is_leaf = False
        return (thresholds, is_leaf)
    else:
        thresholds = histogram_thresholds[col]
        max = x_feature.max()
        min = x_feature.min()

        thresholds = [th for th in thresholds if th >= min and th < max]

        if len(thresholds) > 0:
            return (thresholds, False)

        return (thresholds, True)


def get_best_split(X: pd.DataFrame, y: pd.Series, bins, histogram_thresholds):

    def _get_s_entropy(y: pd.Series) -> float:
        classes_probability = y.value_counts() / len(y)
        entropy = -np.sum(classes_probability *
                          np.log2(classes_probability))
        return entropy

    s0 = _get_s_entropy(y)
    result_col = None
    result_split_value = 0
    result_ig = 0

    # значит остался только один класс и делить смысла нет.
    if s0 == 0:
        return (result_col, float(result_split_value), float(result_ig))

    for col in X.columns:
        thresholds_result = get_thresholds(
            col, X[col], bins, histogram_thresholds)

        if thresholds_result[1]:
            continue

        for threshold in thresholds_result[0]:
            mask = X[col] <= threshold
            y_left = y[mask]
            y_right = y[~mask]

            buff_entroy = s0 - (len(y_left) / len(y)) * _get_s_entropy(
                y_left) - (len(y_right) / len(y)) * _get_s_entropy(y_right)

            if buff_entroy > result_ig:
                result_ig = buff_entroy
                result_col = col
                result_split_value = threshold

    return (result_col, float(result_split_value), float(result_ig))
