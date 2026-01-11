import pandas as pd
from my_log_reg import MyLogReg
from sklearn.datasets import make_regression


logReg = MyLogReg(n_iter=50, learning_rate=0.1, weights=None, metric='roc_auc')

y = pd.Series([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
y_score = pd.Series([0.91, 0.86, 0.78, 0.6, 0.6, 0.55,
                    0.51, 0.46, 0.45, 0.45, 0.42])

g = MyLogReg._metric_roc_auc(y, y_score)

t = 1
