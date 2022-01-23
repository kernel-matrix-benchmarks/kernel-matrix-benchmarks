from __future__ import absolute_import
import numpy as np


def memory_footprint(queries, attrs):
    # TODO: should replace this with peak memory usage, including the query
    return attrs.get("memory_footprint", 0)


def build_time(queries, attrs):
    return attrs["build_time"]


# All possible choices of performance metrics that can be displayed
# on the "x" or "y" axes of our plots:
all_metrics = {
    # TODO: Obsolete
    "k-nn": {
        "description": "Recall",
        "function": lambda true_distances, run_distances, metrics, run_attrs: knn(
            true_distances, run_distances, run_attrs["count"], metrics
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    # TODO: Obsolete
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(
            true_distances, run_distances, run_attrs["count"], metrics
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
    },
    # TODO: Obsolete
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(
            true_distances, run_distances, run_attrs["count"], metrics, 0.1
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
    },
    # TODO: Slightly obsolete
    "rel": {
        "description": "Relative Error",
        "function": lambda true_distances, run_distances, metrics, run_attrs: rel(
            true_distances, run_distances, metrics
        ),  # noqa
        "worst": float("inf"),
    },
    # TODO: Obsolete
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: queries_per_second(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("-inf"),
    },
    # TODO: Slightly obsolete
    "distcomps": {
        "description": "Distance computations",
        "function": lambda true_distances, run_distances, metrics, run_attrs: dist_computations(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: build_time(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    # TODO: Obsolete
    "candidates": {
        "description": "Candidates generated",
        "function": lambda true_distances, run_distances, metrics, run_attrs: candidates(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    # TODO: Slightly obsolete
    "indexsize": {
        "description": "Index size (kB)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: index_size(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    # TODO: Slightly obsolete
    "queriessize": {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: index_size(
            true_distances, run_attrs
        )
        / queries_per_second(true_distances, run_attrs),  # noqa
        "worst": float("inf"),
    },
}
