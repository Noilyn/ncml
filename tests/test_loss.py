from ncml.datasets.loaders import load_dataset
from ncml.impl.gist import GISTClassifier, GISTRegressor
import pytest

# test different loss functions based on gist algorithm, lasso penalty

ret = load_dataset('w5a')
X_tr, y_tr, X_te, y_te = ret


def test_gist_logistic_mcp_clf():
    clf = GISTClassifier(loss='logistic', penalty='mcp', lambda_=1e-4,
                         theta=.25, max_iters=10000, tol=1e-4,  verbose=True,
                         warm_start=True)
    clf.fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.95


# failed
def test_gist_squaredloss_mcp_reg():
    reg = GISTRegressor(loss='squaredloss', penalty='mcp', lambda_=1e-4,
                        theta=.25, max_iters=10000, tol=1e-4, verbose=True,
                        warm_start=True)
    reg.fit(X_tr, y_tr)
    assert reg.score(X_tr, y_tr) > 0.95


if __name__ == "__main__":
    pytest.main([__file__])