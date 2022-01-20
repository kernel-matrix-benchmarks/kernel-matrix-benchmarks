from __future__ import absolute_import
import numpy as np
import sklearn.neighbors
from kernel_matrix_benchmarks.distance import metrics as pd
from kernel_matrix_benchmarks.algorithms.base import BaseProduct

kernel_functions = {
    "gaussian": lambda sqdists: np.exp(-sqdists),
    "absolute exponential": lambda sqdists: np.exp(-np.sqrt(np.maximum(sqdists, 0))),
}


class BruteForceProductBLAS(BaseProduct):
    """Bruteforce implementation, using BLAS through NumPy."""

    def __init__(self, kernel="gaussian", normalize_rows=False, precision=np.float64):
        if kernel not in kernel_functions:
            raise NotImplementedError(
                f"BruteForceProductBLAS doesn't support kernel {kernel}."
            )
        self.kernel = kernel
        self.precision = precision
        self.normalize_rows = normalize_rows
        self.name = "BruteForceProductBLAS()"

    def fit(self, source_points, source_signal=None):
        """Pre-computes the Euclidean norms of the source points."""

        # Cast to the required precision and make sure
        # that everyone is contiguous for top performance:
        self.source_points = np.ascontiguousarray(source_points, dtype=self.precision)
        if source_signal is None:
            self.source_signal = None
        else:
            self.source_signal = np.ascontiguousarray(
                source_signal, dtype=self.precision
            )

        # Pre-compute the squared Euclidean norm of each point:
        self.source_sqnorms = (self.source_points ** 2).sum(-1)

    def batch_query(self, target_points):

        # Cast to the required precision and as contiguous array for top performance:
        # TODO: Check if target_points being contiguous vs its transpose
        #       being contiguous is faster.
        target_points = np.ascontiguousarray(target_points, dtype=self.precision)
        target_sqnorms = (target_points ** 2).sum(-1)

        sqdists = (
            self.source_sqnorms
            + target_sqnorms
            - 2 * self.source_points @ target_points.T
        )
        K_ij = kernel_functions[self.kernel](sqdists)

        if self.normalize_rows:
            # Normalized rows for e.g. attention layers.
            if self.source_signal is None:
                # Density estimation: the source signal is equal to 1
                # -> trivial result since the lines of the kernel matrix sum up to 1.
                self.res = np.ones_like(self.target_points[:, :1])
            else:
                # We compute both the product and the normalization in one sweep:
                # this should optimize memory transfers.
                signal_1 = np.concatenate(
                    (self.source_signal, np.ones_like(self.source_signal[..., :1])),
                    axis=1,
                )
                res_sum = K_ij @ signal_1
                self.res = res_sum[..., :-1] / res_sum[..., -1:]
        else:
            # Standard kernel matrix product.
            if self.source_signal is None:
                # Density estimation: the source signal is equal to 1
                self.res = np.sum(K_ij, -1, keepdims=True)
            else:
                #  General case: we use a matrix product
                self.res = K_ij @ self.source_signal

