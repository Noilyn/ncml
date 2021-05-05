"""
    Test bbmpg considering different scale and linesearch choices
"""

from ncml.datasets.loaders import load_dataset
from ncml.impl.bbmpg import BBMPGClassifier, BBMPGRegressor
from ncml.impl.untils import reg_synthetic_reproduce
import pytest

ret = load_dataset('w5a')
X_tr_clf, y_tr_clf, X_te_clf, y_te_clf = ret
X_tr_reg, y_tr_reg = reg_synthetic_reproduce(n_samples=1440, n_features=5120,
                                         nnz_num=160, csr_flag=True)

def test_bbmpg_digonalbb_nomonotonic_clf():
    clf = BBMPGClassifier(loss='logistic', scale_choice='diagonal_bb',
                          linesearch_choice='nonmonotonic', tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_bbmpg_bb1_nomonotonic_clf():
    clf = BBMPGClassifier(loss='logistic', scale_choice='bb_1',
                          linesearch_choice='nonmonotonic', tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_bbmpg_bb2_nomonotonic_clf():
    clf = BBMPGClassifier(loss='logistic', scale_choice='bb_2',
                          linesearch_choice='nonmonotonic', tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_bbmpg_diagonal_bb_monotonic_clf():
    clf = BBMPGClassifier(loss='logistic', scale_choice='diagonal_bb',
                          linesearch_choice='monotonic', tol=1e-4)
    clf.fit(X_tr_clf, y_tr_clf)
    assert clf.score(X_tr_clf, y_tr_clf) > 0.95


def test_bbmpg_diagonal_bb_nomonotonic_reg():
    reg = BBMPGRegressor(loss='squaredloss', scale_choice='diagonal_bb',
                         linesearch_choice='nonmonotonic', tol=1e-4)
    reg.fit(X_tr_reg, y_tr_reg)
    assert reg.score(X_tr_reg, y_tr_reg) > 0.95


if __name__ == "__main__":
    pytest.main([__file__])
