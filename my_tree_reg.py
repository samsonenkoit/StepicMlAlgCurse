import pandas as pd
import numpy as np


class MyTreeReg():

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __str__(self) -> str:
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'


def calc_mse(y: pd.Series) -> float:
    mean = y.mean()
    return 1/len(y) * ((y - mean) ** 2).sum()


def calc_node_split_mse(base_mse: float, base_len: int, y_left: pd.Series, y_right: pd.Series) -> float:
    return base_mse - (len(y_left)/base_len)*calc_mse(y_left) - \
        (len(y_right)/base_len)*calc_mse(y_right)


def get_best_split(X: pd.DataFrame, y: pd.Series):
    node_mse = 0
    result = {'col': "",
              'split_value': 0,
              'gain': 0}
    for col in X.columns:
        feature = X[col]
        uniq_values = feature.sort_values().unique()
        thresholds = (uniq_values[1:] + uniq_values[:-1])/2

        feature_base_mse = calc_mse(y)
        for th in thresholds:
            mask = X[col] > th
            y_left = y[~mask]
            y_right = y[mask]

            buff_node_mse = calc_node_split_mse(
                feature_base_mse, len(y), y_left, y_right)

            if buff_node_mse > node_mse:
                node_mse = buff_node_mse
                result = {
                    'col': col,
                    'split_value': th,
                    'gain': node_mse
                }

    return (result['col'], result['split_value'], result['gain'])
