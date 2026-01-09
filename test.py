import pandas as pd
from my_log_reg import MyLogReg
from sklearn.datasets import make_regression


logReg = MyLogReg(n_iter=50, learning_rate=0.1)


X, y = make_regression(n_samples=1000, n_features=14,
                       n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

logReg.fit(X, y, 2)

t = 1
