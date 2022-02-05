from __future__ import absolute_import
import numpy as np
from kernel_matrix_benchmarks.algorithms.base import BaseProduct, BaseSolver


class DummyProduct(BaseProduct):
    """Random algorithm, for testing purposes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DummyProduct()"

    def prepare_data(self, *, target_points, **kwargs):
        self.n_points = target_points.shape[0]

    def prepare_query(self, *, source_signal):
        self.output_dim = source_signal.shape[-1]

    def query(self):
        self.res = np.random.randn(self.n_points, self.output_dim)


class DummySolver(BaseSolver):
    """Random algorithm, for testing purposes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DummySolver()"

    def prepare_data(self, *, source_points, **kwargs):
        self.n_points = source_points.shape[0]

    def prepare_query(self, *, target_signal):
        self.output_dim = target_signal.shape[-1]

    def query(self):
        self.res = np.random.randn(self.n_points, self.output_dim)
