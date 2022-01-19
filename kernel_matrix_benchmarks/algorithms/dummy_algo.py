from __future__ import absolute_import
import numpy as np
from kernel_matrix_benchmarks.algorithms.base import BaseProduct, BaseSolver


class DummyAlgoProduct(BaseProduct):
    """Random algorithm, for testing purposes."""

    def __init__(self):
        self.name = "DummyAlgoProduct"

    def fit(self, source_points, source_signal=None):
        if source_signal is None:
            self.output_dim = 1
        else:
            self.output_dim = source_signal.shape[-1]

    def query(self, target_point):
        return np.random.randn(self.output_dim)


class DummyAlgoSolver(BaseSolver):
    """Random algorithm, for testing purposes."""

    def __init__(self):
        self.name = "DummyAlgoSolver"

    def fit(self, source_points):
        pass

    def batch_query(self, target_signal):
        self.res = np.random.randn(*target_signal.shape)
