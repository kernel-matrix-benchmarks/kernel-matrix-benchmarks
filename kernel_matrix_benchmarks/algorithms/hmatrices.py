from kernel_matrix_benchmarks.algorithms.base import BaseProduct, BaseSolver
import juliacall
from juliacall import Main as jl
jl.seval("using KernelMatrixBenchmarks") # assume this is in the main environment
import numpy as np
import time

class HMatricesProduct(BaseProduct):
    """HMatrices implementation."""

    def __init__(
        self,
        *,
        kernel,
        dimension,
        eta=3,
        epsilon=1e-3,
        maxleafsize=200,
        precision="Double",
        normalize_rows=False,
    ):

        # Save the kernel_name, dimension, precision type and normalize_rows boolean:
        super().__init__(
            kernel=kernel,
            dimension=dimension,
            normalize_rows=normalize_rows,
        )
        self.name = f"HMatricesProduct({precision},epsilon={epsilon},eta={eta},maxleafsize={maxleafsize})"
        self.eta = eta
        self.epsilon = epsilon
        self.maxleafsize = maxleafsize
        self.precision = precision
        if self.precision == "Double":
            self.numpy_precision = np.double
        elif self.precision == "Single":
            self.numpy_precision = np.single
        else:
            raise NotImplementedError(
                f"HMatricesProduct doesn't support precision {precision}."
            )
        # FIXME: what is the equivalent of "nothing" or "missing"?
        self.hmat = None

    def prepare_data(
        self,
        *,
        source_points,
        target_points,
        same_points=False,
        density_estimation=False
    ):
        """Casts data to the required precision."""
        # Cast to the required precision
        self.source_points = np.asfortranarray(
            source_points, dtype=self.numpy_precision
        )
        if same_points:
            self.target_points = self.source_points
        else:
            self.target_points = np.asfortranarray(
                target_points, dtype=self.numpy_precision
            )

    def fit(self):
        """Pre-computes the kernel matrix."""
        # assemble 
        self.hmat = jl.KernelMatrixBenchmarks.pyassemble(self.kernel,self.target_points,self.source_points,eta=self.eta,nmax=self.maxleafsize,threads=False)

    def prepare_query(self, *, source_signal):
        # Cast to the required precision and shape
        nsources = self.source_points.shape[0]
        self.source_signal = np.asfortranarray(
            source_signal, dtype=self.numpy_precision
        ).reshape(nsources)
        self.res = np.zeros(
            self.target_points.shape[0],
            dtype=self.numpy_precision,
            order="F",
        )

    def query(self):
        jl.KernelMatrixBenchmarks.pygemv(self.res,self.hmat,self.source_signal)
        # because the errors are computed using a reference solution of shape
        # (N,1), we must reshape res 
        ntargets = self.target_points.shape[0]
        self.res = self.res.reshape(ntargets,1)
