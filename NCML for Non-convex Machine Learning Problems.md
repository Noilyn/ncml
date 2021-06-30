<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<script type="text/x-mathjax-config">   MathJax.Hub.Config({     tex2jax: {       inlineMath: [ ['$','$'], ["\\(","\\)"] ],       processEscapes: true     }   }); </script>

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

<a href="https://www.codecogs.com/eqnedit.php?latex=min_{x\in&space;\chi}&space;F(x):=g(x)&plus;f(x)-h(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min_{x\in&space;\chi}&space;F(x):=g(x)&plus;f(x)-h(x)" title="min_{x\in \chi} F(x):=g(x)+f(x)-h(x)" /></a>

$$min_{x\in \chi} F(x):=g(x)+f(x)-h(x)$$

