# NCML for Non-convex Machine Learning Problems

NCML is a numerical package for solving large-scale non-convex optimization arising from machine learning and statistics. The solver provides some popular algorithms under the variable metric scheme which involves the quasi-Newton method. It is written in Python and performs well on many non-convex machine learning problems. NCML is developed by Yilin Wang, SHUFE. Contact: ylwang228@hotmail.com.

### Highlights

- follows the [scikit-learn](https://github.com/scikit-learn/scikit-learn) API conventions
- supports both dense and sparse data representations
- implements quasi-Newton module in Cython
- applies multiple quasi-Newton update criterion (SR1, BFGS)

### Solvers supported

- Barzilai-Borwein method based algorithm (GIST, BBMPG, BBMPG_DCA)
- quasi-Newton based algorithm(GDVMPG)



## DC Programming

In NCML, we mainly consider a family of non-convex possibly non-smooth optimization problems in the following form:

<center><a href="https://www.codecogs.com/eqnedit.php?latex=min_{x\in&space;\chi}&space;F(x):=f(x)&plus;g(x)-h(x)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?min_{x\in&space;\chi}&space;F(x):=f(x)&plus;g(x)-h(x)" title="min_{x\in \chi} F(x):=f(x)+g(x)-h(x)" /></a></center>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /></a> is the optimization parameters, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\chi" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\chi" title="\chi" /></a> is a closed and convex set in <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbb{R}^d" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbb{R}^d" title="\mathbb{R}^d" /></a>,<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f,h:\mathbb{R}^d\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;f,h:\mathbb{R}^d\rightarrow&space;\mathbb{R}" title="f,h:\mathbb{R}^d\rightarrow \mathbb{R}" /></a> are real-valued  convex functions,<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;g:\mathbb{R}^d\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;g:\mathbb{R}^d\rightarrow&space;\mathbb{R}" title="g:\mathbb{R}^d\rightarrow \mathbb{R}" /></a> is a proper lower-semicontinuous function. For an extended-real-value function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F:\mathbb{R}^d\rightarrow&space;\mathbb{R}\cup&space;\{&plus;\infty\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;F:\mathbb{R}^d\rightarrow&space;\mathbb{R}\cup&space;\{&plus;\infty\}" title="F:\mathbb{R}^d\rightarrow \mathbb{R}\cup \{+\infty\}" /></a>, the component function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;g" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;g" title="g" /></a> is a non-smooth regularizer that promotes sparsity, e.g., the convex <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;l_1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;l_1" title="l_1" /></a> norm or the non-convex <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;l_0,&space;l_p" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;l_0,&space;l_p" title="l_0, l_p" /></a> norm with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p&space;\in&space;(0,1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;p&space;\in&space;(0,1)" title="p \in (0,1)" /></a>.



## Example

We consider the following regularized risk minimization problem:

<center><a href="https://www.codecogs.com/eqnedit.php?latex=min_x&space;\Psi(x):=\frac{1}{n}\sum^n_{i=1}L_i(x)&plus;\alpha&space;r(x)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?min_x&space;\Psi(x):=\frac{1}{n}\sum^n_{i=1}L_i(x)&plus;\alpha&space;r(x)" title="min_x \Psi(x):=\frac{1}{n}\sum^n_{i=1}L_i(x)+\alpha r(x)" /></a></center>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L_i(x)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;L_i(x)" title="L_i(x)" /></a> denotes the loss function, e.g., logistic loss for classification problems and quadratic loss for regression problems. And <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;r(x)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;r(x)" title="r(x)" /></a> is a non-convex regularizer including [MCP, SCAD](https://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf), etc.



## Usage

We show how to solve a binary classification problem with MCP penalty on the *news20.binary* dataset.

```python
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



