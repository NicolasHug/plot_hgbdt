Small utility to plot sklearn's HistGradientBoostingRegressor or classifier,
for debugging. Can also plot lightgbm trees for comparison.

```py
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.datasets import make_regression
import numpy as np

from plot_hgbdt import plot_tree


X, y = make_regression()
est = HistGradientBoostingRegressor()
est.fit(X, y)
plot_tree(est)  # plot (first predictor of) estimator
plot_tree(est._predictors[0][0])  # directly plot predictor

# plot from grower, with more info
gradients = np.random.normal(size=100).astype(G_H_DTYPE)
hessians = np.ones(shape=1, dtype=G_H_DTYPE)
bin_mapper = _BinMapper()
X_binned = bin_mapper.fit_transform(X)
grower = TreeGrower(X_binned, gradients, hessians)
grower.grow()
plot_tree(grower)
```
