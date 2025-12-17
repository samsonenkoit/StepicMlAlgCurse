import pandas


class MyLineReg():
    def __init__(self, n_iter, learning_rate, weights: pandas.DataFrame):
        self.n_inter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_inter}, learning_rate={self.learning_rate}"

    def fit()
