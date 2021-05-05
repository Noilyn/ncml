from ncml.datasets.loaders import load_dataset
from ncml.impl.bbmpg_dca import BBMPG_DCAClassifier, BBMPG_DCARegressor
from ncml.impl.untils import reg_synthetic_reproduce
import pytest

ret = load_dataset('w5a')
X_tr_clf, y_tr_clf, X_te, y_te = ret
X_tr_reg, y_tr_reg = reg_synthetic_reproduce(n_samples=1440, n_features=5120,
                                         nnz_num=160, csr_flag=True)


def test_bbmpgdca_no_momentum_clf():
    clf = BBMPG_DCAClassifier(loss='logistic', momentum_flag=False)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


# failed
def test_bbmpgdca_adaptive_clf():
    clf = BBMPG_DCAClassifier(loss='logistic', restart_scheme='adaptive')
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


# failed
def test_bbmpgdca_fixed_clf():
    clf = BBMPG_DCAClassifier(loss='logistic', restart_scheme='fixed',
                              restart_epoch=500)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_bbmpg_no_momentum_reg():
    reg = BBMPG_DCARegressor(loss='squaredloss', momentum_flag=False)
    reg.fit(X_tr_reg, y_tr_reg)
    assert reg.score(X_tr_reg, y_tr_reg) > 0.95


if __name__ == "__main__":
    pytest.main([__file__])