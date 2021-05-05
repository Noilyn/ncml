from ncml.datasets.loaders import load_dataset
from ncml.impl.gist import GISTClassifier
import pytest

ret = load_dataset('w5a')
X_tr, y_tr, X_te, y_te = ret


def test_gist_mcp_clf():
    clf = GISTClassifier(loss='logistic', penalty='mcp')
    clf.fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.95


def test_gist_scad_clf():
    clf = GISTClassifier(loss='logistic', penalty='scad', theta=2)
    clf.fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.95


def test_gist_lasso_clf():
    clf = GISTClassifier(loss='logistic', penalty='lasso')
    clf.fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.95


if __name__ == "__main__":
    pytest.main([__file__])