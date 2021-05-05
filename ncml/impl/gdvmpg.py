from .base import BaseClassifier, BaseRegressor
from .loss import Logistic, SquaredLoss
from .penalty import SCAD, MCP, LASSO
import numpy as np
from numpy.linalg import norm
import time


class _BaseGDVMPG(object):
    """Generazlied Determined Variable Metric Proximal Gradient"""
    def _get_loss(self, X, y):
        raise NotImplementedError()

    def _get_penalty(self):
        raise NotImplementedError()

    def _fit(self, X, y):
