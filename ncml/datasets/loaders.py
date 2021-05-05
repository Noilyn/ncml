# Notice: You have to save the data file in advance at data home path (
# corresponding to sklearn.datasets.get_data_home)

import os
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import get_data_home
from sklearn.preprocessing import MaxAbsScaler


def _load(train_file, test_file, name):
    if not os.path.exists(train_file) or (test_file is not None and not
    os.path.exists(test_file)):
        raise IOError("Dataset missing, " + "Run 'make download-%s' at the "
                                            "project root. " % name)
    if test_file:
        return load_svmlight_files((train_file, test_file))
    else:
        X, y = load_svmlight_files((train_file,))
        return X, y, None, None


def _todense(data):
    X_train, y_train, X_test, y_test = data
    X_train = X_train.toarray()
    if X_test is not None:
        X_test = X_test.toarray()
    return X_train, y_train, X_test, y_test


# regression
def load_abalone():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "abalone_scale")
    return _load(train_file, None, "abalone")


def load_cadata():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "cadata")
    return _load(train_file, None, "cadata")


def load_cpusmall():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "cpusmall_scale")
    return _load(train_file, None, "cpusmall")


def load_space_ga():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "space_ga_scale")
    return _load(train_file, None, "space_ga")


def load_YearPredictionMSD():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "YearPredictionMSD")
    test_file = os.path.join(data_home, "YearPredictionMSD.t")
    return _load(train_file, test_file, "YearPredictionMSD")


# binary classification
def load_madelon():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "madelon")
    return _load(train_file, None, "madelon")


def load_realsim():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "real-sim")
    return _load(train_file, None, "real-sim")


def load_news20_binary():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "news20.binary")
    return _load(train_file, None, "news20.binary")


def load_w5a():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "w5a")
    test_file = os.path.join(data_home, "w5a.t")
    return _load(train_file, test_file, "w5a")


def load_gisette_scale():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "gisette_scale")
    test_file = os.path.join(data_home, "gisette_scale.t")
    return _load(train_file, test_file, "gisette_scale")


def load_rcv1_binary():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "rcv1_train.binary")
    test_file = os.path.join(data_home, "rcv1_test.binary")
    return _load(train_file, test_file, "rcv1.binary")


def load_mnist5():
    X_train, y_train, X_test, y_test = load_mnist()
    selected = y_train == 5
    y_train[selected] = 1
    y_train[~selected] = 0
    selected = y_test == 5
    y_test[selected] = 1
    y_test[~selected] = 0
    return X_train, y_train, X_test, y_test


# multi-class
def load_mnist():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "mnist.scale")
    test_file = os.path.join(data_home, "mnist.scale.t")
    return _load(train_file, test_file, "mnist.scale")


LOADERS = {
    # mlproject competition problem
    "abalone": load_abalone,
    "cpusmall": load_cpusmall,
    "cadata": load_cadata,
    "space_ga": load_space_ga,
    "YearPredictionMSD": load_YearPredictionMSD,
    # binary classification
    "madelon": load_madelon,
    "real-sim": load_realsim,
    "news20.binary": load_news20_binary,
    "w5a": load_w5a,
    "gisette_scale": load_gisette_scale,
    "rcv1.binary": load_rcv1_binary,
    "mnist5": load_mnist5,
    # multi-class
    "mnist": load_mnist,
}


def load_dataset(dataset):
    ret =  LOADERS[dataset]()
    X_tr, y_tr, X_te, y_te = ret
    transformer = MaxAbsScaler().fit(X_tr)
    X_tr = transformer.transform(X_tr)
    if not X_te is None:
        transformer = MaxAbsScaler().fit(X_te)
        X_te = transformer.transform(X_te)
    return X_tr, y_tr, X_te, y_te


if __name__ == "__main__":
        ret = load_dataset("mnist5")
        X_tr, y_tr, X_te, y_te = ret
        print('X_tr', X_tr)
        print('y_tr', y_tr)
        print('X_te', X_te)
        print('y_te', y_te)