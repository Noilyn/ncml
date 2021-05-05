from ncml.datasets.loaders import load_dataset
from ncml.impl.dca import DCAClassifier, DCARegressor
from ncml.impl.untils import reg_synthetic_reproduce
import pytest

ret = load_dataset('w5a')
X_tr_clf, y_tr_clf, X_te_clf, y_te_clf = ret
X_tr_reg, y_tr_reg = reg_synthetic_reproduce(n_samples=1440, n_features=5120,
                                         nnz_num=160, csr_flag=True)


def test_dca_no_momentum_clf():
    clf = DCAClassifier(loss='logistic', momentum_flag=False, tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_dca_adaptive_clf():
    clf = DCAClassifier(loss='logistic', restart_scheme='adaptive', tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_dca_fixed_clf():
    clf = DCAClassifier(loss='logistic', restart_scheme='fixed',
                        restart_epoch=500, tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_dca_no_momentum_reg():
    reg = DCARegressor(loss='squaredloss', momentum_flag=False, tol=1e-8)
    reg.fit(X_tr_reg, y_tr_reg)
    assert reg.score(X_tr_reg, y_tr_reg) > 0.95


if __name__ == "__main__":
    pytest.main([__file__])