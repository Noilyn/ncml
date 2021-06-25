# NCML For Non-convex Machine Leaning Problems

NCML solver (non-convex machine learning solver) is a compositive solver for non-convex optimization. In this project, we provide all kinds of first-order algorithms based on the quasi-Newton method under the variable metric schema. It is written in Python3 and perform well on some non-convex machine learning problems. NCML is developed by Yilin Wang, SHUFE. Contact:ylwang228@hotmail.com.

### Highlights

- follows the [scikit-learn](https://github.com/scikit-learn/scikit-learn) API conventions
- supports both dense and sparse data representations
- quasi-Newton module implemented in Cython
- Multiple quasi-Newton update criterion (SR1, BFGS)

### Solvers supported

- BB method based algorithm (GIST, BBMPG, BBMPG_DCA)
- quasi-Newton based algorithm(GDVMPG)



## DC programming

In NCML, we mainly consider a standard family of non-convex non-smooth(possibly) optimization problems called DC programming:
$$
min_{x\in \chi} F(x):=g(x)+f(x)-h(x)
$$
where $x$ is the decision variable, $\chi$ is a closed and convex set in $\mathbb{R}^d$, $f,h:\mathbb{R}^d \rightarrow \mathbb{R}$ are real-valued lower-semicontinuous convex functions, $g:\mathbb{R}^d \rightarrow \mathbb{R}$ is a proper lower-semicontinuous functions. For an extended-real-value function $F:\mathbb{R}^d \rightarrow \mathbb{R} \cup \{+\infty\}$, the component function $g$ is often used to capture non-differentiable functions that plays the role of regularization, e.g., the convex $l_1$ norm or the non-convex $l_0,l_p$ norm with $p\in(0,1)$.



## Example

By now, we mainly consider the following learning problem:
$$
min_x \Psi(x):=\frac{1}{n}\sum^n_{i=1}L_i(x)+\alpha g(x)
$$
where $L_i(x)$ denotes the loss function, e.g., logistic loss for classification problems and quadratic loss for regression problems. And $g(x)$ is settled as non-convex regularizer including MCP, SCAD, etc.



## Usage

We show how to learn a binary classification problem with MCP penalty on the *news20.binary* dataset.

```
from ncml.datasets.loaders import load_dataset
from ncml.impl.bbmpg_dca import BBMPG_DCAClassifier

# load dataset
def load_data(name):
    ret = load_dataset(name)
    X_tr_clf, y_tr_clf, X_te_clf, y_te_clf = ret
    dataset_without_test = ('madelon', 'real-sim', 'news20.binary')
    if name in dataset_without_test:
        X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = train_test_split(X_tr_clf,
                        y_tr_clf, test_size=0.33, random_state=40)
    return X_tr_clf, y_tr_clf, X_te_clf, y_te_clf
X_tr, y_tr, X_te, y_te = load_data('news20.binary')

# Set classifier options
bbmpgdca = BBMPG_DCAClassifier(loss='logistic',
							   penalty='mcp',
							   scale_choice='diagonal_bb',
                               linesearch_choice='nonmonotonic',
                               momentum_flag=False,
                               tol=bbmpgdca_tol)
# Train the model
bbmpgdca.fit(X_tr, y_tr)

# Accuracy
bbmpgdca_tr_acu = bbmpgdca.score(X_tr, y_tr)
bbmpgdca_te_acu = bbmpgdca.score(X_te, y_te)
print('tr_acu', bbmpgdca_tr_acu)
print('te_acu', bbmpgdca_te_acu)
```

