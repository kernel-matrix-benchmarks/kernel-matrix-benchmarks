from __future__ import absolute_import

import h5py
import json
import os
import re
import traceback


def get_result_filename(
    dataset=None, count=None, definition=None, query_arguments=None, batch_mode=False
):
    """Creates a filename thatlookss like "results/dataset/algorithm/M_4_L_0_5.hdf5"."""
    d = ["results"]
    if dataset:
        d.append(dataset)
    if count:  # TODO: "count" is obsolete, a k-nn only variable TODO:
        # We could replace it with "kernel"
        d.append(str(count))
    if definition:
        d.append(definition.algorithm + ("-batch" if batch_mode else ""))
        data = definition.arguments + query_arguments
        # The filename is a "flat" expansion of the dict of parameters,
        # with all "non alphanumerical symbols" replaced by "_".
        d.append(
            re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".hdf5"
        )
    return os.path.join(*d)


def store_results(dataset, count, definition, query_arguments, attrs, results, batch):
    """Stores the raw output of a computation."""

    # The result filename looks like
    # "results/dataset/algorithm/M_4_L_0_5.hdf5"
    fn = get_result_filename(dataset, count, definition, query_arguments, batch)

    # Creates the folder "results/dataset/algorithm/":
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)

    # Creates the file "M_4_L_0_5.hdf5":
    f = h5py.File(fn, "w")

    # attrs is a dictionary with keys:
    # "build_time", "index_size", "algo", "dataset",
    # "batch_mode", "best_search_time", "candidates",
    # "expect_extra", "name", "run_count", "count" (obsolete)
    # and algorithm-specific "extras".
    # All of this is saved in the hdf5 file.
    for k, v in attrs.items():
        f.attrs[k] = v

    # Stores the entries of the result with their computation times:
    # TODO: Currently, this method is ANN-specific
    times = f.create_dataset("times", (len(results),), "f")
    neighbors = f.create_dataset("neighbors", (len(results), count), "i")
    distances = f.create_dataset("distances", (len(results), count), "f")
    for i, (time, ds) in enumerate(results):
        times[i] = time
        neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
        distances[i] = [d for n, d in ds] + [float("inf")] * (count - len(ds))
    f.close()


def load_all_results(dataset=None, count=None, batch_mode=False):
    """Python iterator that returns all the "results" hdf5 files with the correct attributes."""
    for root, _, files in os.walk(get_result_filename(dataset, count)):
        for fn in files:
            if os.path.splitext(fn)[-1] != ".hdf5":
                continue
            try:
                f = h5py.File(os.path.join(root, fn), "r+")
                properties = dict(f.attrs)
                if batch_mode != properties["batch_mode"]:
                    # "continue" -> skip to the next file
                    continue

                # "yield" = "return", but for iterators
                yield properties, f
                f.close()
            except:
                print("Was unable to read", fn)
                traceback.print_exc()


def get_unique_algorithms():
    algorithms = set()
    for batch_mode in [False, True]:
        for properties, _ in load_all_results(batch_mode=batch_mode):
            algorithms.add(properties["algo"])
    return algorithms
