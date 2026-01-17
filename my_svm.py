import pandas as pd
import numpy as np


class MySVM():

    def __init__(self, learning_rate=0.001, n_iter: int = 10) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
