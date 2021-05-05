from ncml.datasets.loaders import load_dataset
from ncml.impl.gist import GISTClassifier, GISTRegressor
from ncml.impl.untils import reg_synthetic_reproduce
import pytest

ret_clf = load_dataset('w5a')
X_tr_clf, y_tr_clf, X_te_clf, y_te_clf = ret_clf
X_tr_reg, y_tr_reg = reg_synthetic_reproduce(n_samples=1440, n_features=5120,
                                         nnz_num=160, csr_flag=True)


def test_gist_logistic_mcp_clf():
    clf = GISTClassifier(loss='logistic', tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_gist_squaredloss_mcp_reg():
    reg = GISTRegressor(loss='squaredloss', tol=1e-8)
    reg.fit(X_tr_reg, y_tr_reg)
    assert reg.score(X_tr_reg, y_tr_reg) > 0.95


if __name__ == "__main__":
    pytest.main([__file__])