"""Code to store the results of a computation.

All results are stored as HDF5 files at:
"results/dataset/algorithm/params_json_dump.hdf5".

A result file "f" contains the following tables:

- f["result"] = (N,E) or (M,E) float64 array.
    The output of the computation, 
    i.e. the target_signal "a_i"  with shape (N,E) (if algo.task == "product")
    or the source signal "b_j" with shape (M,E) (if algo.task == "solver").

- f["error"] = (N,E) or (M,E) float64 array.
    The difference "result - true_answer".


And the following metadata: 

- f.attrs["dataset"] = str.
    The name of the dataset.

- f.attrs["algo"] = str.
    The name of the algorithm used, from "algos.yaml".

- f.attrs["name"] = str.
    The name of the method, algo.name.

- f.attrs["kernel"] = str.
    The name of the kernel function used.

- f.attrs["run_count"] = int.
    The number of independent runs used to benchmark the precomputation
    and query times.

- f.attrs["build_time"] = float number.
    The minimum time (among several runs) needed to warm-up the algorithm
    and e.g. pre-compute a tree representation of the data or 
    a Cholesky decomposition of the kernel matrix.

- f.attrs["query_time"] = float number.
    The minimum time (among several runs) needed to compute the result,
    without taking into account the precomputation time.

- f.attrs["memory_footprint"] = float number.
    The memory size (in kiloBytes) of the pre-computed representation.

The output of algo.get_additional() is also appended to the dict f.attrs.
"""

from __future__ import absolute_import

import h5py
import json
import os
import re
import traceback


def get_result_filename(
    dataset=None, definition=None, query_arguments=None,
):
    """Creates a filename that looks like "results/dataset/algorithm/M_4_L_0_5.hdf5"."""
    d = ["results"]
    if dataset:
        d.append(dataset)
    if definition:
        d.append(definition.algorithm)
        data = dict(definition.arguments, **query_arguments)
        # The filename is a "flat" expansion of the dict of parameters,
        # with all "non alphanumerical symbols" replaced by "_".
        d.append(
            re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".hdf5"
        )
    return os.path.join(*d)


def store_result(*, dataset, definition, query_arguments, attrs, result, error):
    """Stores the raw output of a computation."""

    # The result filename looks like
    # "results/dataset/algorithm/M_4_L_0_5.hdf5"
    fn = get_result_filename(dataset, definition, query_arguments)

    # Creates the folder "results/dataset/algorithm/":
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)

    # Creates the file "M_4_L_0_5.hdf5":
    f = h5py.File(fn, "w")

    # attrs is a dictionary with keys:
    # "build_time", "memory_footprint", "algo", "dataset",
    # "best_query_time", "name", "run_count",
    # and algorithm-specific "extras".
    # All of this is saved in the hdf5 file.
    for k, v in attrs.items():
        f.attrs[k] = v

    # Stores the entries of the result with their computation times:
    f["result"] = result
    f["error"] = error  #  error = result - true_answer

    f.close()


def load_all_results(dataset=None):
    """Python iterator that returns all the "results" hdf5 files with the correct attributes."""
    for root, _, files in os.walk(get_result_filename(dataset)):
        for fn in files:
            if os.path.splitext(fn)[-1] != ".hdf5":
                continue
            try:
                f = h5py.File(os.path.join(root, fn), "r+")
                properties = dict(f.attrs)
                # "yield" = "return", but for iterators
                yield properties, f
                f.close()
            except:
                print("Was unable to read", fn)
                traceback.print_exc()


def get_unique_algorithms():
    algorithms = set()
    for properties, _ in load_all_results():
        algorithms.add(properties["algo"])
    return algorithms
