from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.datasets import make_regression
import numpy as np

from plot_hgbdt import plot_tree


def test_plot():
    # just make sure estimators, growers and predictors are equally accepted
    X, y = make_regression()
    est = HistGradientBoostingRegressor()
    est.fit(X, y)
    plot_tree(est, view=False)
    plot_tree(est._predictors[0][0], view=False)

    gradients = np.random.normal(size=100).astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)
    grower = TreeGrower(X_binned, gradients, hessians)
    grower.grow()
    plot_tree(grower, view=False)
