import mlflow
from sklearn.model_selection import train_test_split
import warnings
from ncml.datasets.loaders import load_dataset
from ncml.impl.gist import GISTClassifier
from ncml.impl.dca import DCAClassifier
from ncml.impl.bbmpg import BBMPGClassifier
from ncml.impl.bbmpg_dca import BBMPG_DCAClassifier


def load_data(name):
    ret = load_dataset(name)
    X_tr_clf, y_tr_clf, X_te_clf, y_te_clf = ret
    dataset_without_test = ('madelon', 'real-sim', 'news20.binary')
    if name in dataset_without_test:
        X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = train_test_split(X_tr_clf,
                        y_tr_clf, test_size=0.33, random_state=40)
    return X_tr_clf, y_tr_clf, X_te_clf, y_te_clf


def experiment(name, *params):
    X_tr, y_tr, X_te, y_te = load_data(name)
    (gist_tol, dca_tol, bbmpg_tol, bbmpgdca_tol) = params
    run_name = '%s -- cmp binary classification -- accuracy ' % name

    with mlflow.start_run(run_name=run_name):
        print('-----------gist test-----------')
        gist = GISTClassifier(tol=gist_tol)
        gist.fit(X_tr, y_tr)
        gist_tr_acu = gist.score(X_tr, y_tr)
        gist_te_acu = gist.score(X_te, y_te)

        print('-----------dca test-----------')
        dca = DCAClassifier(restart_scheme='adaptive', momentum_flag=True,
                            tol=dca_tol)
        dca.fit(X_tr, y_tr)
        dca_tr_acu = dca.score(X_tr, y_tr)
        dca_te_acu = dca.score(X_te, y_te)

        print('-----------bbmpg test-----------')
        bbmpg = BBMPGClassifier(scale_choice='diagonal_bb',
                                linesearch_choice='nonmonotonic',
                                tol=bbmpg_tol)
        bbmpg.fit(X_tr, y_tr)
        bbmpg_tr_acu = bbmpg.score(X_tr, y_tr)
        bbmpg_te_acu = bbmpg.score(X_te, y_te)

        print('-----------bbmpgdca test-----------')
        bbmpgdca = BBMPG_DCAClassifier(scale_choice='diagonal_bb',
                                       linesearch_choice='nonmonotonic',
                                       momentum_flag=False,
                                       tol=bbmpgdca_tol)
        bbmpgdca.fit(X_tr, y_tr)
        bbmpgdca_tr_acu = bbmpgdca.score(X_tr, y_tr)
        bbmpgdca_te_acu = bbmpgdca.score(X_te, y_te)

        mlflow.log_params({
            'dataset_name': name, 'gist_tol': gist_tol, 'dca_tol': dca_tol,
            'bbmpg_tol': bbmpg_tol, 'bbmpgdca_tol': bbmpgdca_tol,
        })
        mlflow.log_metrics({
            'gist_tr_accuracy': gist_tr_acu, 'gist_te_accuracy': gist_te_acu,
            'dca_tr_accuracy': dca_tr_acu, 'dca_te_accuracy': dca_te_acu,
            'bbmpg_tr_accuracy': bbmpg_tr_acu,
            'bbmpg_te_accuracy': bbmpg_te_acu,
            'bbmpgdca_tr_accuracy': bbmpgdca_tr_acu,
            'bbmpgdca_te_accuracy': bbmpgdca_te_acu,
        })


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    experiment('w5a', 1e-3, 1e-3, 1e-3, 1e-3)