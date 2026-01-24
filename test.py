import pandas as pd
from my_log_reg import MyLogReg


g = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 6, 8]], columns=['x1', 'x2', 'x3'])
print(g.sum(axis=1))
t = 1
