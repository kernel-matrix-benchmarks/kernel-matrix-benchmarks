from __future__ import absolute_import
from multiprocessing.pool import ThreadPool
import psutil

# !!! Obsolete name
class BaseANN(object):
    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X):
        """Fits the algorithm to a training distribution.

        Args:
            X ((M, D) array): the reference point cloud.
        """

        # By default, we do nothing:
        pass

    def query(self, q, n):
        """Performs the computation of interest for a single point q.

        Args:
            q ((D,) vector): query point.
            n (int): number of neighbors, !!! obsolete.

        Returns:
            !!! (E,) vector?: output of the computation at point q.
        """

        # By default, we don't return anything:
        # !!! Obsolete behavior
        return []  # array of candidate indices

    def batch_query(self, X, n):
        """Provide all queries at once and let the algorithm figure out how to handle it.

        Default implementation uses a ThreadPool to parallelize query processing.

        Args:
            X ((N, D) array): query points.
            n (int): number of neighbors, !!! obsolete.

        Returns:
            None: see the get_batch_results() method below.
        """
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self):
        """Returns the result of batch_query() as a NumPy array.

        This routine is not included in the timer and may be used
        to e.g. unload the result from a GPU device and cast it as a NumPy array.

        Returns:
            !!! (N,E) array?: output of the computation at the N points X.
        """
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name
