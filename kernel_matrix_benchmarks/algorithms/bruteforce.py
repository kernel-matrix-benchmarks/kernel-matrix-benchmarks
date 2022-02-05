from __future__ import absolute_import
import numpy as np
import scipy
from scipy.linalg import solve, lstsq
from kernel_matrix_benchmarks.algorithms.base import BaseProduct, BaseSolver


def inverse_square_root(sqdists):
    """Inefficient implementation of "rsqrt", which is not supported by NumPy."""
    sqdists_0 = np.maximum(sqdists, 0)
    res = 1 / np.sqrt(sqdists_0)
    # Put zeroes on the diagonal:
    buffer = res.reshape(-1)
    buffer[:: res.shape[1] + 1] = 0
    return buffer.reshape(res.shape)


kernel_functions = {
    "inverse-distance": inverse_square_root,
    "gaussian": lambda sqdists: np.exp(-sqdists),
    "absolute-exponential": lambda sqdists: np.exp(-np.sqrt(np.maximum(sqdists, 0))),
}


def kernel_matrix(*, kernel, source_points, target_points=None, fast_sqdists=False):

    if target_points is None:
        target_points = source_points  # (N,D) = (M,D)

    # Extract the shape of the data:
    M, D = source_points.shape
    N, _ = target_points.shape

    if fast_sqdists:
        # Pre-compute the squared Euclidean norm of each point:
        source_sqnorms = (source_points**2).sum(-1)  # (M,)
        if target_points is None:
            target_sqnorms = source_sqnorms  # (N,)
        else:
            target_sqnorms = (target_points**2).sum(-1)  # (N,)

        # Compute the matrix of squared distances with BLAS:
        # N.B.: Due to floating point errors, sqdists may not always
        #       be non-negative or zero when "x_i = y_j".
        sqdists = (
            target_sqnorms.reshape(N, 1)  # (N,1)
            + source_sqnorms.reshape(1, M)  # (1,M)
            - 2 * target_points @ source_points.T  # (N,D) @ (D,M) = (N,M)
        )
    else:
        # Slow computation with a (N,M,D) memory buffer,
        # but a guarantee of positivity:
        diffs = target_points.reshape(N, 1, D) - source_points.reshape(1, M, D)
        sqdists = np.sum(diffs**2, axis=-1)  # (N,M,D) -> (N,M)

    # Apply the kernel function pointwise:
    K_ij = kernel_functions[kernel](sqdists)  # (N,M)
    return K_ij


class BruteForceProductBLAS(BaseProduct):
    """Bruteforce implementation, using BLAS through NumPy."""

    def __init__(
        self,
        *,
        kernel,
        dimension,
        normalize_rows=False,
        precision=np.float64,
        fast_sqdists=False,
    ):

        # Save the kernel_name, dimension, precision type and normalize_rows boolean:
        super().__init__(
            kernel=kernel,
            dimension=dimension,
            normalize_rows=normalize_rows,
            precision=precision,
        )

        if kernel not in kernel_functions:
            raise NotImplementedError(
                f"BruteForceProductBLAS doesn't support kernel {kernel}."
            )
        self.fast_sqdists = fast_sqdists
        self.name = f"BruteForceProductBLAS({precision}, fast_sqdists={fast_sqdists})"

    def prepare_data(
        self,
        *,
        source_points,
        target_points,
        same_points=False,
        density_estimation=False,
    ):
        """Casts data to the required precision."""
        # Cast to the required precision and make sure
        # that everyone is contiguous for top performance:
        self.source_points = np.ascontiguousarray(source_points, dtype=self.precision)
        # TODO: Check if target_points being contiguous vs its transpose
        #       being contiguous is faster.
        self.target_points = (
            None
            if same_points
            else np.ascontiguousarray(target_points, dtype=self.precision)
        )
        # Remember if the source and target points are identical:
        self.same_points = same_points
        # Remember if this is a density estimation benchmark:
        self.density_estimation = density_estimation

    def fit(self):
        """Pre-computes the kernel matrix."""
        self.K_ij = kernel_matrix(
            kernel=self.kernel,
            source_points=self.source_points,
            target_points=self.target_points,
            fast_sqdists=self.fast_sqdists,
        )

    def prepare_query(self, *, source_signal):
        # Cast to the required precision and as contiguous array for top performance:
        self.source_signal = (
            None
            if self.density_estimation
            else np.ascontiguousarray(source_signal, dtype=self.precision)
        )

    def query(self):

        if self.normalize_rows:
            # Normalized rows for e.g. attention layers.
            if self.density_estimation:
                # Density estimation: the source signal is equal to 1
                # -> trivial result since the lines of the kernel matrix sum up to 1.
                points = self.source_points if self.same_points else self.target_points
                self.res = np.ones_like(points[:, :1])
            else:
                # We compute both the product and the normalization in one sweep:
                # this should optimize memory transfers.
                ones_column = np.ones_like(self.source_signal[..., :1])
                signal_1 = np.concatenate((self.source_signal, ones_column), axis=1)
                res_sum = self.K_ij @ signal_1
                self.res = res_sum[..., :-1] / res_sum[..., -1:]
        else:
            # Standard kernel matrix product.
            if self.density_estimation:
                # Density estimation: the source signal is equal to 1
                self.res = np.sum(self.K_ij, -1, keepdims=True)  # (N,1)
            else:
                #  General case: we use a matrix product
                self.res = self.K_ij @ self.source_signal  # (N,M) @ (M,E)


class BruteForceSolverLAPACK(BaseSolver):
    """Bruteforce implementation, using LAPACK ?POSV through SciPy.

    We assume that the kernel matrix is symmetric, positive definite.
    """

    def __init__(
        self,
        *,
        kernel,
        dimension,
        normalize_rows=False,
        precision=np.float64,
        fast_sqdists=False,
    ):
        # Save the kernel_name, dimension, precision type and normalize_rows boolean:
        super().__init__(
            kernel=kernel,
            dimension=dimension,
            normalize_rows=normalize_rows,
            precision=precision,
        )

        if kernel not in kernel_functions:
            raise NotImplementedError(
                f"BruteForceSolverLAPACK doesn't support kernel {kernel}."
            )
        self.fast_sqdists = fast_sqdists
        self.name = f"BruteForceSolverLAPACK({precision}, fast_sqdists={fast_sqdists})"

    def prepare_data(self, *, source_points):
        """Casts data to the required precision."""

        # Cast to the required precision and make sure
        # that everyone is contiguous for top performance:
        self.source_points = np.ascontiguousarray(source_points, dtype=self.precision)

    def fit(self):
        """Pre-computes the kernel matrix."""
        self.K_ij = kernel_matrix(
            kernel=self.kernel,
            source_points=self.source_points,
            fast_sqdists=self.fast_sqdists,
        )

    def prepare_query(self, *, target_signal):
        # Cast to the required precision and as contiguous array for top performance:
        self.target_signal = np.ascontiguousarray(target_signal, dtype=self.precision)

    def query(self):
        # self.res = solve(self.K_ij, self.target_signal, assume_a="pos")
        self.res = lstsq(self.K_ij, self.target_signal)[0]
