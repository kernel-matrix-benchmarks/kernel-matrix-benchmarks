from __future__ import absolute_import
import numpy as np
import scipy
from scipy.linalg import solve, lstsq
from kernel_matrix_benchmarks.algorithms.base import BaseProduct, BaseSolver
import HtoolKernelMatrixBenchmarks as HtoolBench


HtoolSupportedKernels = {
    "inverse-distance": "InverseDistanceKernel",
    "gaussian": "GaussianKernel",
}


class HtoolProduct(BaseProduct):
    """Htool implementation."""

    def __init__(
        self,
        *,
        kernel,
        dimension,
        normalize_rows=False,
        eta=10,
        epsilon=1e-3,
        maxblocksize=100,
        target_minclustersize=10,
        source_minclustersize=10,
        symmetry="N",
        UPLO="N",
        precision="Double",
    ):

        # Save the kernel_name, dimension, precision type and normalize_rows boolean:
        super().__init__(
            kernel=kernel,
            dimension=dimension,
            normalize_rows=normalize_rows,
        )

        if kernel not in HtoolSupportedKernels:
            raise NotImplementedError(f"HtoolProduct doesn't support kernel {kernel}.")

        self.name = f"HtoolProduct({precision},epsilon={epsilon},eta={eta},maxblocksize={maxblocksize},target_minclustersize={target_minclustersize},source_minclustersize={source_minclustersize},symmetry={symmetry},UPLO={UPLO})"
        self.eta = eta
        self.epsilon = epsilon
        self.maxblocksize = maxblocksize
        self.symmetry = symmetry
        self.target_minclustersize = target_minclustersize
        self.source_minclustersize = source_minclustersize
        self.UPLO = UPLO
        self.precision = precision
        if self.precision == "Double":
            self.bench = HtoolBench.HtoolBenchmarkPCARegularClusteringDouble(
                dimension,
                HtoolSupportedKernels[kernel],
                "partialACA",
                self.symmetry,
                self.UPLO,
            )
            self.numpy_precision = np.double
        elif self.precision == "Single":
            self.bench = HtoolBench.HtoolBenchmarkPCARegularClusteringFloat(
                dimension,
                HtoolSupportedKernels[kernel],
                "partialACA",
                self.symmetry,
                self.UPLO,
            )
            self.numpy_precision = np.single
        else:
            raise NotImplementedError(
                f"HtoolProduct doesn't support precision {precision}."
            )

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
        self.source_points = np.asfortranarray(
            np.transpose(source_points), dtype=np.double
        )
        self.NbCols = self.source_points.shape[1]
        self.target_points = (
            None
            if same_points
            else np.asfortranarray(np.transpose(target_points), dtype=np.double)
        )
        if same_points:
            self.NbRows = self.NbCols
        else:
            self.NbRows = self.target_points.shape[1]

        # Remember if the source and target points are identical:
        self.same_points = same_points
        # Remember if this is a density estimation benchmark:
        self.density_estimation = density_estimation

    def fit(self):
        """Pre-computes the kernel matrix."""
        if self.same_points:
            self.bench.build_clusters(
                self.NbCols, self.source_points, self.source_minclustersize
            )
            self.bench.build_HMatrix(
                self.source_points, self.epsilon, self.eta, 0, 0, self.maxblocksize
            )
        else:
            self.bench.build_clusters(
                self.NbRows,
                self.NbCols,
                self.target_points,
                self.source_points,
                self.target_minclustersize,
                self.source_minclustersize,
            )
            self.bench.build_HMatrix(
                self.target_points,
                self.source_points,
                self.epsilon,
                self.eta,
                0,
                0,
                100,
            )

    def prepare_query(self, *, source_signal):
        # Cast to the required precision and as contiguous array for top performance:
        self.source_signal = np.asfortranarray(
            source_signal, dtype=self.numpy_precision
        )

        if self.normalize_rows:
            ones_column = np.ones_like(self.source_signal[..., :1])
            signal_1 = np.concatenate((self.source_signal, ones_column), axis=1)
            self.source_signal = signal_1
        self.bench.print_HMatrix_infos()

    def query(self):
        if self.normalize_rows:
            res_sum = np.zeros(
                (self.NbRows, self.source_signal.shape[1]),
                dtype=self.numpy_precision,
                order="F",
            )
            self.bench.product(self.source_signal, res_sum)
            # Normalized rows for e.g. attention layers.
            # We compute both the product and the normalization in one sweep:
            # this should optimize memory transfers.

            self.res = res_sum[:, :-1] / res_sum[:, -1:]
        else:
            self.res = np.zeros(
                (self.NbRows, self.source_signal.shape[1]),
                dtype=self.numpy_precision,
                order="F",
            )
            self.bench.product(self.source_signal, self.res)
