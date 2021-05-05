# not completed, just used for testing


import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from lightning.regression import AdaGradRegressor, SGDRegressor, \
    FistaRegressor, CDRegressor, SAGRegressor, SAGARegressor, SDCARegressor, \
    SVRGRegressor, LinearSVR
import warnings
from ncml.datasets.loaders import load_dataset
from ncml.impl.gist import GISTRegressor
from ncml.impl.dca import DCARegressor
from ncml.impl.bbmpg import BBMPGRegressor
from ncml.impl.bbmpg_dca import BBMPG_DCARegressor


def load_data(name):
    ret = load_dataset(name)
    X_tr_clf, y_tr_clf, X_te_clf, y_te_clf = ret
    return X_tr_clf, y_tr_clf


if __name__ == "__main__":
    X_tr, y_tr = load_data('abalone')
    reg = LogisticRegression().fit(X_tr, y_tr)
    print(X_tr)
    print('/////////////////////')
    print(y_tr)
    print('------------------------------')
    print(reg.score(X_tr, y_tr))

    bbmpgdca = BBMPG_DCARegressor(scale_choice='diagonal_bb',
                                       linesearch_choice='nonmonotonic',
                                       momentum_flag=False, verbose=False,
                                       tol=1e-6)
    bbmpgdca.fit(X_tr, y_tr)
    print(bbmpgdca.score(X_tr, y_tr))