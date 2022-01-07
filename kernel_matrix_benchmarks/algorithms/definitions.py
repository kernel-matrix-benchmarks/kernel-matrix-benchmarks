from __future__ import absolute_import
from os import sep as pathsep
import collections
import importlib
import os
import sys
import traceback
import yaml
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
        "disabled",
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


def _generate_combinations(args):
    """Returns the Cartesian product of a list/dict of lists."""

    if isinstance(args, list):
        args = [el if isinstance(el, list) else [el] for el in args]
        return [list(x) for x in product(*args)]
    elif isinstance(args, dict):
        flat = []
        for k, v in args.items():
            if isinstance(v, list):
                flat.append([(k, el) for el in v])
            else:
                flat.append([(k, v)])
        return [dict(x) for x in product(*flat)]
    else:
        raise TypeError("No args handling exists for %s" % type(args).__name__)


def _substitute_variables(arg, vs):
    """Replaces the magic keywords "@kernel", etc. used in "algos.yaml"."""
    if isinstance(arg, dict):
        return dict([(k, _substitute_variables(v, vs)) for k, v in arg.items()])
    elif isinstance(arg, list):
        return [_substitute_variables(a, vs) for a in arg]
    elif isinstance(arg, str) and arg in vs:
        return vs[arg]
    else:
        return arg


def _get_definitions(definition_file):
    """Parses "algos.yaml"."""
    with open(definition_file, "r") as f:
        return yaml.load(f, yaml.SafeLoader)


def list_algorithms(definition_file):
    """High-level overview of "algos.yaml"."""

    definitions = _get_definitions(definition_file)

    print("The following algorithms are supported...")
    for point in definitions:
        print('\t... for the point type "%s"...' % point)
        for metric in definitions[point]:
            print('\t\t... and the distance metric "%s":' % metric)
            for algorithm in definitions[point][metric]:
                print("\t\t\t%s" % algorithm)


def get_unique_algorithms(definition_file):
    """Removes doublons from "algos.yaml"."""
    definitions = _get_definitions(definition_file)
    algos = set()
    for point in definitions:
        for metric in definitions[point]:
            for algorithm in definitions[point][metric]:
                algos.add(algorithm)
    return list(sorted(algos))


def get_definitions(
    definition_file,
    dimension,
    point_type="float",
    distance_metric="euclidean",
    count=10,
):
    # Step 1: Load the .yaml file --------------------------
    # Load "algos.yaml" using the standard .yaml parser:
    definitions = _get_definitions(definition_file)

    algorithm_definitions = {}

    # Load the algorithms that support "any" metric:
    if "any" in definitions[point_type]:
        algorithm_definitions.update(definitions[point_type]["any"])

    # And add the algorithms that "only" support the target metric, if any:
    algorithm_definitions.update(definitions[point_type].get(distance_metric, {}))

    # Step 2: Process the experiments/libraries ---------------------
    definitions = []
    for (name, algo) in algorithm_definitions.items():

        # Step 2.a: Check that each algorithm is defined by:
        # - a docker image
        # - a python module (i.e. a python file)
        # - a python constructor (i.e. a python class)
        for k in ["docker-tag", "module", "constructor"]:
            if k not in algo:
                raise Exception(
                    'algorithm %s does not define a "%s" property' % (name, k)
                )

        # Step 2.b: Load the "basic arguments" such as the kernel type
        # that will be needed by all "children" experiments.
        base_args = []
        if "base-args" in algo:
            base_args = algo["base-args"]

        # Step 2.c: Loop over the "run groups" that define
        # several ways of using the same python constructor.
        for run_group in algo["run-groups"].values():

            # Step 2.d: Expand the lists of arguments into a long list
            # of possible configurations for the python constructor.
            #
            # N.B.: "_generate_combinations" generates Cartesian products
            # of a list/dict of lists, so that:
            # _generate_combinations([[0, 1], [2, 3]])
            #   = [[0,2], [0,3], [1,2], [1,3]]
            # and
            # _generate_combinations({"A" : [0, 1], "B" : [2, 3]})
            #   = [{"A":0, "B":2}, ..., {"A":1, "B":3}]

            # Case 1 - heavy: The group defines several groups of args.
            if "arg-groups" in run_group:
                groups = []
                for arg_group in run_group["arg-groups"]:
                    if isinstance(arg_group, dict):
                        # Dictionaries need to be expanded into lists in order
                        # for the subsequent call to _generate_combinations to
                        # do the right thing
                        groups.append(_generate_combinations(arg_group))
                    else:
                        groups.append(arg_group)
                args = _generate_combinations(groups)
            # Case 2 - simpler: The group is associated to a list of args.
            elif "args" in run_group:
                args = _generate_combinations(run_group["args"])
            # Case 3 - error: the group has not defined any arguments.
            else:
                assert False, "? what? %s" % run_group

            # Step 2.e: We do the same thing for the arguments "at query time".
            if "query-arg-groups" in run_group:
                groups = []
                for arg_group in run_group["query-arg-groups"]:
                    if isinstance(arg_group, dict):
                        groups.append(_generate_combinations(arg_group))
                    else:
                        groups.append(arg_group)
                query_args = _generate_combinations(groups)
            elif "query-args" in run_group:
                query_args = _generate_combinations(run_group["query-args"])
            else:
                query_args = []

            # Step 2.f: Turn these lists of arguments into full, self-contained
            # descriptions of our experiments.
            for arg_group in args:

                # Concatenate the "constant" and "variable" parameters,
                # i.e. "aargs = base_args + arg_group", with careful handling of
                # lists and "singleton" parameters.
                aargs = []
                aargs.extend(base_args)
                if isinstance(arg_group, list):
                    aargs.extend(arg_group)
                else:
                    aargs.append(arg_group)

                # Magic keywords:
                vs = {
                    "@count": count,  # !!! We should replace this !!!
                    "@metric": distance_metric,
                    "@dimension": dimension,
                }
                aargs = [_substitute_variables(arg, vs) for arg in aargs]

                # The final definition object - a namedtuple:
                definitions.append(
                    Definition(
                        algorithm=name,
                        docker_tag=algo["docker-tag"],
                        module=algo["module"],
                        constructor=algo["constructor"],
                        arguments=aargs,
                        query_argument_groups=query_args,
                        disabled=algo.get("disabled", False),
                    )
                )

    return definitions
