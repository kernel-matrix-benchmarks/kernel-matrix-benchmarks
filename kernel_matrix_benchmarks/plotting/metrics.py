"""Defines all the performance metrics that may be displayed on the website.

Metrics are stored in the dict "all_metrics" with the following syntax:

all_metrics = {
    "key-for-metric-1": {
        "description": str, 
            a label to put on the x/y axes.
        "worst": float, typically float("inf") or float("-inf"),
            useful to plot the Pareto frontier of optimal values.
        "lim": [vmin, vmax] optional pair of floats,
            useful to explicitely set the axes boundaries.
        "function": function,
            Scalar-valued function that implements the performance metric.
            The expected keyword arguments are:
                - dataset, the hdf5 file that defines the expected problem.
                - result, float64 NumPy array, the output of the method.
                - error, float64 NumPy array, = result - true_value.
                - properties, dict of metadata associated to the algorithm run.
                - metrics_cache, hdf5 file group where we can store some computations.
    },
    "key-for-metric-1": {
        ...
    }
}



The 
(
            dataset=dataset,
            result=result,
            error=error,
            properties=properties,
            metrics_cache=metrics_cache,
        )
"""


from __future__ import absolute_import
import numpy as np


# Compute errors ===============================================================


def result_errors(*, error, metrics_cache, **kwargs):
    """Computes a collection of metrics: median, average and max error + rmse.

    Args:
        error ((N,E) or (M,E) float64 array): the pointwise error values 
            "output - true_value" of an algorithm run.
        metrics_cache (HDF5 file category): cache to store the computed values.

    Returns:
        HDF5 file category: the cache where the error values and statistics are stored.
    """
    if "errors" not in metrics_cache:
        # Create a sub-category in the HDF5 file:
        errors_cache = metrics_cache.create_group("errors")

        # Compute the L2-Euclidean norm of every output E-vector:
        # (remember that in most cases, E=1 for scalar-valued computations)
        norms = np.sqrt(np.sum(error ** 2, axis=-1))  # (N,E) -> (N,)

        # Fill in the cache with statistics:
        errors_cache.attrs["max"] = np.max(norms)
        errors_cache.attrs["mean"] = np.mean(norms)
        errors_cache.attrs["median"] = np.median(norms)
        errors_cache.attrs["rmse"] = np.sqrt(np.mean(norms ** 2))

    return metrics_cache["errors"]


# Read the metadata defined in results.py ======================================


def build_time(*, properties, **kwargs):
    return properties["build_time"]


def query_time(*, properties, **kwargs):
    return properties["query_time"]


def total_time(**kwargs):
    return build_time(**kwargs) + query_time(**kwargs)


def memory_footprint(*, properties, **kwargs):
    # TODO: should replace this with peak memory usage, including the query
    return properties.get("memory_footprint", 0)


# Summary ======================================================================
# All possible choices of performance metrics that can be displayed
# on the "x" or "y" axes of our plots:
all_metrics = {
    "max-error": {
        "description": "Maximum error",
        "function": lambda **kwargs: result_errors(**kwargs).attrs["max"],
        "worst": float("inf"),
    },
    "mean-error": {
        "description": "Average error",
        "function": lambda **kwargs: result_errors(**kwargs).attrs["mean"],
        "worst": float("inf"),
    },
    "median-error": {
        "description": "Median error",
        "function": lambda **kwargs: result_errors(**kwargs).attrs["median"],
        "worst": float("inf"),
    },
    "rmse-error": {
        "description": "Root mean squared error",
        "function": lambda **kwargs: result_errors(**kwargs).attrs["rmse"],
        "worst": float("inf"),
    },
    "build-time": {
        "description": "Build time (s)",
        "function": build_time,
        "worst": float("inf"),
    },
    "query-time": {
        "description": "Query time (s)",
        "function": query_time,
        "worst": float("inf"),
    },
    "total-time": {
        "description": "Build+Query time (s)",
        "function": total_time,
        "worst": float("inf"),
    },
    "memory-footprint": {
        "description": "Memory footprint (kB)",
        "function": memory_footprint,
        "worst": float("inf"),
    },
}
