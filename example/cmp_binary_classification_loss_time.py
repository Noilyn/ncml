import matplotlib.pyplot as plt
import mlflow
import os
import warnings
from ncml.datasets.loaders import load_dataset
from ncml.impl.gist import GISTClassifier
from ncml.impl.dca import DCAClassifier
from ncml.impl.bbmpg import BBMPGClassifier
from ncml.impl.bbmpg_dca import BBMPG_DCAClassifier


def load_data(name):
    ret = load_dataset(name)
    X_tr_clf, y_tr_clf, X_te_clf, y_te_clf = ret
    return X_tr_clf, y_tr_clf


def experiment(name, *params):
    X, y = load_data(name)
    (gist_tol, dca_tol, bbmpg_tol, bbmpgdca_tol) = params
    run_name = '%s -- cmp binary classification -- loss time ' % name

    with mlflow.start_run(run_name=run_name):
        print('-----------gist test-----------')
        gist = GISTClassifier(tol=gist_tol)
        gist.fit(X, y)
        gist_obj, gist_time = gist.obj_pass, gist.execution_time_pass

        print('-----------dca test-----------')
        dca = DCAClassifier(restart_scheme='adaptive', momentum_flag=True,
                            tol=dca_tol)
        dca.fit(X, y)
        dca_obj, dca_time = dca.obj_pass, dca.execution_time_pass

        print('-----------bbmpg test-----------')
        bbmpg = BBMPGClassifier(scale_choice='diagonal_bb',
                                linesearch_choice='nonmonotonic',
                                tol=bbmpg_tol)
        bbmpg.fit(X, y)
        bbmpg_obj, bbmpg_time = bbmpg.obj_pass, bbmpg.execution_time_pass

        print('-----------bbmpgdca test-----------')
        bbmpgdca = BBMPG_DCAClassifier(scale_choice='diagonal_bb',
                                       linesearch_choice='nonmonotonic',
                                       momentum_flag=False,
                                       tol=bbmpgdca_tol)
        bbmpgdca.fit(X, y)
        bbmpgdca_obj, bbmpgdca_time = bbmpgdca.obj_pass, bbmpgdca.execution_time_pass

        mlflow.log_params({'dataset_name': name, 'gist_tol': gist_tol,
                           'dca_tol': dca_tol,
                           'bbmpg_tol': bbmpg_tol, 'bbmpgdca_tol': bbmpgdca_tol})

        upper_path = os.path.abspath(os.path.dirname(os.getcwd()))
        save_path = os.path.join(upper_path, 'results', 'binary_classification')
        os.makedirs(save_path, exist_ok=True)

        ms, fs = 5, 15
        plt.figure()
        plt.plot(gist_time[5::5], gist_obj[5::5], marker='D', label='gist',
                 markersize=ms)
        plt.plot(dca_time[5::5], dca_obj[5::5], marker='<', label='dca',
                 markersize=ms)
        plt.plot(bbmpg_time[5::5], bbmpg_obj[5::5], marker='x', label='bbmpg',
                 markersize=ms)
        plt.plot(bbmpgdca_time[5::5], bbmpgdca_obj[5::5], marker='>',
                 label='bbmpgdca', markersize=ms)
        plt.xlabel('execution time(s)', fontsize=fs)
        plt.ylabel('loss', fontsize=fs)
        plt.tight_layout()
        plt.legend(loc='best', fontsize=fs)
        plt.savefig(os.path.join(save_path, '%s.pdf' % name))
        mlflow.log_artifact(os.path.join(save_path, '%s.pdf' % name))
        plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    experiment('news20.binary', 1e-3, 1e-3, 1e-3, 1e-3)