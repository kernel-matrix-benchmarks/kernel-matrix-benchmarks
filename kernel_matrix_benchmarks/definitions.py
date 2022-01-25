from __future__ import absolute_import
from os import sep as pathsep
import collections
import importlib
import os
import sys
import traceback
import yaml
import fnmatch
from enum import Enum
from itertools import product


# This file instantiates the algorithm descriptions located in "algos.yaml".

Definition = collections.namedtuple(
    "Definition",
    [
        "algorithm",
        "constructor",
        "module",
        "docker_tag",
        "arguments",
        "query_argument_groups",
    ],
)


def instantiate_algorithm(definition):
    """Loads a library and creates the module specified by the experiment definition.

    Args:
        definition (Definition): A namedtuple that describes a method to benchmark.

    Returns:
        A python object that answers numerical queries.
    """
    print(
        "Trying to instantiate %s.%s(%s)"
        % (definition.module, definition.constructor, definition.arguments)
    )
    module = importlib.import_module(definition.module)
    constructor = getattr(module, definition.constructor)
    return constructor(*definition.arguments)


class InstantiationStatus(Enum):
    AVAILABLE = 0
    NO_CONSTRUCTOR = 1
    NO_MODULE = 2


def algorithm_status(definition):
    """Checks that an algorithm can be loaded."""
    try:
        module = importlib.import_module(definition.module)
        if hasattr(module, definition.constructor):
            return InstantiationStatus.AVAILABLE
        else:
            return InstantiationStatus.NO_CONSTRUCTOR
    except ImportError:
        return InstantiationStatus.NO_MODULE


def _get_definitions(definition_file):
    """Parses "algos.yaml"."""
    with open(definition_file, "r") as f:
        return yaml.load(f, yaml.SafeLoader)


def list_algorithms(definition_file):
    """High-level overview of "algos.yaml"."""

    definitions = _get_definitions(definition_file)

    print("The following algorithms are supported...")
    for tag, algorithm in definitions.items():
        print(f"\t{tag} for the tasks:")
        print(f"\t\tproduct:   {algorithm.get('product', False)}")
        print(f"\t\tsolver:    {algorithm.get('solver', False)}")
        print(f"\t\tattention: {algorithm.get('attention', False)}")


def get_unique_algorithms(definition_file):
    """Removes doublons from "algos.yaml"."""
    definitions = _get_definitions(definition_file)
    return list(sorted(set(definitions)))


def get_definitions(
    definition_file="algos.yaml",
    dimension=3,
    dataset="uniform-sphere-D3-E1-M1000-N1000-inverse-distance",
    task="product",
    hardware="CPU",
    kernel="gaussian",
    normalize_rows=False,
    run_disabled=False,
):
    # Step 1: Load the .yaml file --------------------------
    # Load "algos.yaml" using the standard .yaml parser:
    all_definitions = _get_definitions(definition_file)

    # Step 2: Process the experiments/libraries ---------------------
    definitions = []
    for (name, algo) in all_definitions.items():
        # Step 2.a: Check that the definition is meant to be used.
        if (  # We skip the current iteration if:
            (  # The algorithm has been disabled
                algo.get("disabled", False) and not run_disabled
            )
            or algo.get("hardware", "CPU") != hardware  # The hardware is not right
            or not algo.get(task, False)  # The algo does not support the current task
        ):
            continue

        # Step 2.b: Check that each algorithm is defined by:
        # - a docker image
        # - a python module (i.e. a python file)
        # - a python constructor (i.e. a python class)
        for k in ["docker-tag", "module", "constructor"]:
            if k not in algo:
                raise Exception(
                    'algorithm %s does not define a "%s" property' % (name, k)
                )

        # Step 2.c: Loop over the "run groups" that define
        # several ways of using the same python constructor.
        for run_group in algo["run-groups"].values():

            # Step 2.d: Check that the run-group is meant to be used on the dataset
            if not any(
                fnmatch.fnmatch(dataset, pattern) for pattern in run_group["datasets"]
            ):
                continue

            # Step 2.e: Load the lists of args and of query args.
            all_args = run_group.get("args", [{}])
            all_query_args = run_group.get("query-args", [{}])

            # Step 2.f: Turn these lists of arguments into full, self-contained
            # descriptions of our experiments.
            for args in all_args:
                base_args = {
                    "kernel": kernel,
                    "dimension": dimension,
                    "normalize_rows": normalize_rows,
                }
                aargs = dict(base_args, **args)

                # The final definition object - a namedtuple:
                definitions.append(
                    Definition(
                        algorithm=name,
                        docker_tag=algo["docker-tag"],
                        module=algo["module"],
                        constructor=algo["constructor"],
                        arguments=aargs,
                        query_argument_groups=all_query_args,
                    )
                )

    return definitions
