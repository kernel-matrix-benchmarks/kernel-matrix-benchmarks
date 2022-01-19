from __future__ import absolute_import
from multiprocessing.pool import ThreadPool
import psutil


class BaseAlgorithm(object):
    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name


class BaseProduct(BaseAlgorithm):
    """Base class for kernel matrix products and attention layers."""

    def fit(self, source_points, source_signal=None):
        """Fits the algorithm to a source distribution.

        Args:
            source_points ((M,D) array): the reference point cloud.
            source_signal ((M,E) array or None): the reference signal.
                If None, we assume that E=1 and that the source signal 
                is uniformly equal to 1, i.e. we perform kernel density estimation.
        """

        # By default, we do nothing:
        pass

    def query(self, target_point):
        """Performs the computation of interest for a single target point.

        Args:
            target_point ((D,) vector): query point x_i.

        Returns:
            (E,) vector: output of the computation at point x_i.
        """

        raise NotImplementedError()

    def batch_query(self, target_points):
        """Provide all queries at once and let the algorithm figure out how to handle it.

        Default implementation uses a ThreadPool to parallelize query processing.

        Args:
            target_points ((N,D) array): query points.

        Returns:
            None: see the get_batch_results() method below.
        """
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q), target_points)

    def get_batch_results(self):
        """Returns the result of batch_query() as a NumPy array.

        This routine is not included in the timer and may be used
        to e.g. unload the result from a GPU device and cast it as a NumPy array.

        Returns:
            (N,E) array: output of the computation at the N points x_i.
        """
        return self.res


class BaseSolver(BaseAlgorithm):
    """Base class for kernel matrix solvers."""

    def fit(self, source_points):
        """Fits the algorithm to a source distribution.

        Args:
            source_points ((M,D) array): the reference point cloud.
        """

        # By default, we do nothing:
        pass

    def batch_query(self, target_signal):
        """Computes the solution of a kernel linear system and store it in self.res

        Args:
            target_signal ((N,E) array): output of the kernel matrix product.

        Returns:
            None: see the get_batch_results() method below.
        """
        raise NotImplementedError()

    def get_batch_results(self):
        """Returns the result of batch_query() as a NumPy array.

        This routine is not included in the timer and may be used
        to e.g. unload the result from a GPU device and cast it as a NumPy array.

        Returns:
            (M,E) array: output of the computation at the M points y_j.
        """
        return self.res

