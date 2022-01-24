from __future__ import absolute_import
from multiprocessing.pool import ThreadPool
import psutil
import numpy as np


class BaseAlgorithm(object):
    def __init__(self, *, kernel, normalize_rows=False, precision=np.float64):
        """Sets up some of the basic attributes of our algorithm.

        Args:
            kernel (str):
                The kernel function that is expected to be used on the dataset.
            normalize_rows (bool, optional): 
                Should we normalize the rows of the kernel matrix so that 
                they sum up to 1? 
                Defaults to False.
            precision (numpy dtype, optional): 
                Desired precision level of the computation. Note that the output 
                array will always be cast as a float64 NumPy array. 
                Defaults to np.float64.
        """
        self.kernel = kernel
        self.precision = precision
        self.normalize_rows = normalize_rows
        self.name = "BaseAlgorithm()"

    def done(self):
        """Frees memory and files after a computation even if an exception has been thrown."""
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        return psutil.Process().memory_info().rss / 1024

    def set_query_arguments(self, *args):
        """Sets additional arguments, after the pre-computation step but before the query."""
        pass

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name


class BaseProduct(BaseAlgorithm):
    """Base class for kernel matrix products and attention layers."""

    task = "product"

    def prepare_data(
        self,
        *,
        source_points,
        target_points,
        same_points=False,
        density_estimation=False,
    ):
        """Load data for the pre-processing step, outside of the timer.

        This routine is not included in the timer and may be used
        to e.g. load the input data from the RAM to a GPU device.

        Args:
            source_points ((M,D) array): the reference point cloud.
            target_points ((N,D) array): query points.
            same_points (bool): 
                Should we assume that the target point cloud is equal to the source?
                Defaults to False.
            density_estimation (bool): 
                Should we assume that the source signal is equal to 1?
                Defaults to False.
        """
        pass

    def fit(self):
        """Fits the algorithm to a source distribution - this operation is timed."""
        pass

    def prepare_query(self, *, source_signal):
        """Reformat or recasts the input target points, outside of the timer.
        
        To ensure a fair benchmark, we may need to 
        e.g. load queries on the GPU or change the numerical precision.

        Args:
            source_signal ((M,E) array or None): the reference signal.
                Note that if self.density_estimation=None, the algorithm may assume 
                that E=1 and that the source signal is uniformly equal to 1, 
                i.e. we perform kernel density estimation.
        """
        pass

    def query(self):
        """Performs the computation of interest for all target points - this operation is timed.

        Returns:
            None: see the get_result() method below.
        """
        self.res = None

    def get_result(self):
        """Returns the result of query() as a float64 NumPy array, outside of the timer.

        This routine is not included in the timer and may be used
        to e.g. unload the result from a GPU device and cast it as a NumPy array.

        Returns:
            (N,E) array: output of the computation at the N points x_i.
        """
        return np.ascontiguousarray(self.res, dtype=np.float64)


class BaseSolver(BaseAlgorithm):
    """Base class for kernel matrix solvers."""

    task = "solver"

    def prepare_data(self, source_points):
        """Load data for the pre-processing step, outside of the timer.

        This routine is not included in the timer and may be used
        to e.g. load the result from the RAM to a GPU device.

        Args:
            source_points ((M,D) array): the reference point cloud.
        """
        pass

    def fit(self):
        """Fits the algorithm to a source distribution - this operation is timed."""
        pass

    def prepare_query(self, target_signal):
        """Reformat or recasts the input target signal, outside of the timer.
        
        To ensure a fair benchmark, we may need to 
        e.g. load the signal on the GPU or change the numerical precision.
        """
        pass

    def query(self, target_signal):
        """Computes the solution of a kernel linear system and store it in self.res - this operation is timed.

        Args:
            target_signal ((N,E) array): output of the kernel matrix product.

        Returns:
            None: see the get_result() method below.
        """
        raise NotImplementedError()

    def get_result(self):
        """Returns the result of query() as a float64 NumPy array, outside of the timer.

        This routine is not included in the timer and may be used
        to e.g. unload the result from a GPU device and cast it as a NumPy array.

        Returns:
            (M,E) array: output of the computation at the M points y_j.
        """
        return np.ascontiguousarray(self.res, dtype=np.float64)

