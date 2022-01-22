from __future__ import absolute_import
import numpy as np
import scipy
from kernel_matrix_benchmarks.distance import metrics as pd
from kernel_matrix_benchmarks.algorithms.base import BaseProduct, BaseSolver


def inverse_square_root(sqdists):
    """Inefficient implementation of "rsqrt", which is not supported by NumPy."""
    res = 1 / np.sqrt(sqdists)
    res[sqdists == 0] = 0
    return res


kernel_functions = {
    "inverse-distance": inverse_square_root,
    "gaussian": lambda sqdists: np.exp(-sqdists),
    "absolute-exponential": lambda sqdists: np.exp(-np.sqrt(np.maximum(sqdists, 0))),
}


class BruteForceProductBLAS(BaseProduct):
    """Bruteforce implementation, using BLAS through NumPy."""

    def __init__(self, kernel="gaussian", normalize_rows=False, precision=np.float64):

        # Save the kernel_name, precision type and normalize_rows boolean:
        super().__init__(
            kernel=kernel, normalize_rows=normalize_rows, precision=precision
        )

        if kernel not in kernel_functions:
            raise NotImplementedError(
                f"BruteForceProductBLAS doesn't support kernel {kernel}."
            )
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
        self.source_sqnorms = (self.source_points ** 2).sum(-1)  # (M,)

    def prepare_batch_query(self, target_points):
        # Cast to the required precision and as contiguous array for top performance:
        # TODO: Check if target_points being contiguous vs its transpose
        #       being contiguous is faster.
        self.target_points = np.ascontiguousarray(target_points, dtype=self.precision)

    def batch_query(self):
        M, D = self.source_points.shape
        N, _ = self.target_points.shape

        target_sqnorms = (self.target_points ** 2).sum(-1)

        sqdists = (
            self.source_sqnorms.reshape(1, M)  # (1,M)
            + target_sqnorms.reshape(N, 1)  # (N,1)
            - 2 * self.target_points @ self.source_points.T  # (N,D) @ (D,M) = (N,M)
        )
        K_ij = kernel_functions[self.kernel](sqdists)  # (N,M)

        if self.normalize_rows:
            # Normalized rows for e.g. attention layers.
            if self.source_signal is None:
                # Density estimation: the source signal is equal to 1
                # -> trivial result since the lines of the kernel matrix sum up to 1.
                self.res = np.ones_like(self.target_points[:, :1])
            else:
                # We compute both the product and the normalization in one sweep:
                # this should optimize memory transfers.
                ones_column = np.ones_like(self.source_signal[..., :1])
                signal_1 = np.concatenate((self.source_signal, ones_column), axis=1)
                res_sum = K_ij @ signal_1
                self.res = res_sum[..., :-1] / res_sum[..., -1:]
        else:
            # Standard kernel matrix product.
            if self.source_signal is None:
                # Density estimation: the source signal is equal to 1
                self.res = np.sum(K_ij, -1, keepdims=True)  # (N,1)
            else:
                #  General case: we use a matrix product
                self.res = K_ij @ self.source_signal  # (N,M) @ (M,E)


class BruteForceSolverLAPACK(BaseSolver):
    """Bruteforce implementation, using LAPACK ?POSV through SciPy.
    
    We assume that the kernel matrix is symmetric, positive definite.
    """

    def __init__(self, kernel="gaussian", normalize_rows=False, precision=np.float64):
        # Save the kernel_name, precision type and normalize_rows boolean:
        super().__init__(
            kernel=kernel, normalize_rows=normalize_rows, precision=precision
        )

        if kernel not in kernel_functions:
            raise NotImplementedError(
                f"BruteForceSolverLAPACK doesn't support kernel {kernel}."
            )
        self.name = "BruteForceSolverLAPACK()"

    def fit(self, source_points):
        """Pre-computes the kernel matrix."""

        # Cast to the required precision and make sure
        # that everyone is contiguous for top performance:
        source_points = np.ascontiguousarray(source_points, dtype=self.precision)
        M, _ = source_points.shape

        # Pre-compute the squared Euclidean norm of each point:
        source_sqnorms = (self.source_points ** 2).sum(-1)  # (M,D) -> (M,)
        sqdists = (
            source_sqnorms.reshape(M, 1)  # (M,1)
            + source_sqnorms.reshape(1, M)  # (1,M)
            - 2 * source_points @ source_points.T  # (M,D) @ (D,M)
        )

        self.K_ij = kernel_functions[self.kernel](sqdists)

    def prepare_batch_query(self, target_signal):
        # Cast to the required precision and as contiguous array for top performance:
        self.target_signal = np.ascontiguousarray(target_signal, dtype=self.precision)

    def batch_query(self):
        self.res = scipy.linalg.solve(self.K_ij, self.target_signal, assume_a="pos")
