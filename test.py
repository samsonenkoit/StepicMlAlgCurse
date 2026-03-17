import numpy
from sklearn.datasets import fetch_openml
import pandas as pd

from my_tree_clf import MyTreeClf
data = fetch_openml(name='banknote-authentication', version=1, as_frame=True)
X = data.data
y = data.target.astype(int)
y = y - 1

items, thresholds = numpy.histogram(X[X.columns[0]], 3)

tree = MyTreeClf(max_depth=4, min_samples_split=100, max_leafs=17, bins=16)
tree.fit(X, y)
tree.print_tree()
print(tree.leafs_sum)
print(tree.leafs_cnt)


g = tree.predict_proba(X)
g1 = 1
