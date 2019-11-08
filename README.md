Small utility to plot sklearn's HistGradientBoostingRegressor or classifier,
for debugging. Can also plot lightgbm trees for comparison.


```py
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import make_regression
import numpy as np

from plot_hgbdt import plot_tree


X, y = make_regression()
est = HistGradientBoostingRegressor()
est.fit(X, y)
plot_tree(est)  # plot (first predictor of) estimator
plot_tree(est._predictors[0][0])  # directly plot predictor
```

See test file for all use-cases.
