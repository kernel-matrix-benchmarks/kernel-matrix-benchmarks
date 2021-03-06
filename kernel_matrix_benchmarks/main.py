from __future__ import absolute_import
import argparse
import logging
import logging.config

import docker
import multiprocessing.pool
import os
import psutil
import random
import shutil
import sys
import traceback

from kernel_matrix_benchmarks.datasets import get_dataset, DATASETS
from kernel_matrix_benchmarks.definitions import (
    get_definitions,
    list_algorithms,
    algorithm_status,
    InstantiationStatus,
)
from kernel_matrix_benchmarks.results import get_result_filename
from kernel_matrix_benchmarks.runner import run, run_docker


def positive_int(s):
    """Converts the input to an integer, raises an exception if it is <= 0."""
    i = None
    try:
        i = int(s)
    except ValueError:
        pass
    if not i or i < 1:
        raise argparse.ArgumentTypeError("%r is not a positive integer" % s)
    return i


def run_worker(cpu, args, queue):
    """Runs all the jobs in the queue, possibly using Docker.

    Args:
        cpu (int): the max number of CPUs that should run the job.
        args (parsed arguments): the arguments given by the user when typing
            `python run.py --arguments...`
        queue (multiprocessing Queue): methods asked by the user.
    """

    while not queue.empty():

        # Definitions are instantiated at the end of this script.
        definition = queue.get()

        if args.local:
            # Case 1: the user does not want to bother with Docker,
            #    e.g. when writing and testing a pull request on a local machine.
            run(definition=definition, dataset=args.dataset, runs=args.runs)

        else:
            # Case 2: the user is using Docker, e.g. when rendering the website.
            memory_margin = 500e6  # reserve some extra memory (~500 Mb) for misc stuff
            mem_limit = int((psutil.virtual_memory().available - memory_margin))
            #  Use all available CPUs:
            cpu_limit = "0-%d" % (multiprocessing.cpu_count() - 1)
            run_docker(
                definition=definition,
                dataset=args.dataset,
                runs=args.runs,
                timeout=args.timeout,
                cpu_limit=cpu_limit,
                mem_limit=mem_limit,
            )


def main():

    # This function is called when the user types `python run.py --arguments...`

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="the dataset to load training points from",
        default="product-sphere-D3-E1-M1000-N1000-inverse-distance",
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "--hardware",
        metavar="INSTANCE",
        help="the type of instance that is currently running the script",
        default="CPU",
        choices=["CPU", "GPU"],
    )
    parser.add_argument(
        "--definitions",
        metavar="FILE",
        help="load algorithm definitions from FILE",
        default="algos.yaml",
    )
    parser.add_argument(
        "--algorithm", metavar="NAME", help="run only the named algorithm", default=None
    )
    parser.add_argument(
        "--docker-tag",
        metavar="NAME",
        help="run only algorithms in a particular docker image",
        default=None,
    )
    parser.add_argument(
        "--list-algorithms",
        help="print the names of all known algorithms and exit",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        help="re-run algorithms even if their results already exist",
        action="store_true",
    )
    parser.add_argument(
        "--runs",
        metavar="COUNT",
        type=positive_int,
        help="run each algorithm instance %(metavar)s times and use only"
        " the best result. This is especially useful for methods that rely on"
        " just-in-time compiling: the first run includes compiling times"
        " whereas the second doesn't.",
        default=2,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout (in seconds) for each individual algorithm run, or -1"
        "if no timeout should be set",
        default=2 * 600,  # Max 10mn per run to keep costs manageable.
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set, then will run everything locally (inside the same "
        "process) rather than using Docker",
    )
    parser.add_argument(
        "--max-n-algorithms",
        type=int,
        help="Max number of algorithms to run (just used for testing)",
        default=-1,
    )
    parser.add_argument(
        "--run-disabled",
        help="run algorithms that are disabled in algos.yml",
        action="store_true",
    )

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None

    if args.list_algorithms:
        # `python run.py --list-algorithms`
        # -> We just display all the possible algorithms and exit.
        list_algorithms(args.definitions)
        sys.exit(0)

    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger("kmb")

    # Load the dataset:
    dataset, dimension = get_dataset(args.dataset)

    # Properties of the dataset:
    kernel = dataset.attrs["kernel"]
    task = dataset.attrs["task"]
    normalize_rows = dataset.attrs.get("normalize_rows", False)
    # Don't forget to close the HDF5 file:
    dataset.close()

    # Definition of the input problem.
    # These correspond to the experiments listed in algos.yaml
    definitions = get_definitions(
        definition_file=args.definitions,
        dimension=dimension,
        dataset=args.dataset,
        task=task,
        hardware=args.hardware,
        kernel=kernel,
        normalize_rows=normalize_rows,
        run_disabled=args.run_disabled,
    )

    # Filter out, from the loaded definitions, all those query argument groups
    # that correspond to experiments that have already been run. (This might
    # mean removing a definition altogether, so we can't just use a list
    # comprehension.)
    filtered_definitions = []
    for definition in definitions:

        query_argument_groups = definition.query_argument_groups  # = [{}] in most cases

        # Filter out, for this specific "definition" (i.e. Python object + parameters)
        # what are the arguments "at query time" that have already been tried.
        not_yet_run = []
        for query_arguments in query_argument_groups:
            # The result filename looks like
            # "results/dataset/algorithm/M_4_L_0_5.hdf5"
            fn = get_result_filename(args.dataset, definition, query_arguments)
            if args.force or not os.path.exists(fn):
                not_yet_run.append(query_arguments)

        if not_yet_run:  # ...is not empty, i.e. some experiments remain to be run:
            if definition.query_argument_groups:
                definition = definition._replace(query_argument_groups=not_yet_run)
            filtered_definitions.append(definition)

    # "definitions" is a list of "experiment definitions" whose results do not
    # already appear on the hard drive.
    definitions = filtered_definitions

    # N.B.: We shuffle all the experiments. This could help us to avoid
    # unnecessary bias against e.g. the last experiments in "algos.yaml",
    # since some GPUs may overheat and experience a sharp decline in performance
    # after several hours of continuous work.
    random.shuffle(definitions)

    # If the user has specified a single algorithm,
    # we filter out all the other experiments:
    if args.algorithm:
        logger.info(f"running only {args.algorithm}")
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    # Case 1: The user is working with Docker, i.e. "not on the local machine".
    if not args.local:
        # See which Docker images we have available
        docker_client = docker.from_env()
        docker_tags = set()
        for image in docker_client.images.list():
            for tag in image.tags:
                tag = tag.split(":")[0]
                docker_tags.add(tag)

        # If the user has specified a single Docker image,
        # we filter out all the other experiments:
        if args.docker_tag:
            logger.info(f"running only {args.docker_tag}")
            definitions = [d for d in definitions if d.docker_tag == args.docker_tag]

        # If some docker images are referenced in "algos.yaml" but cannot
        # be found in the docker environment, we add a warning in the log file:
        if set(d.docker_tag for d in definitions).difference(docker_tags):
            logger.info(f"not all docker images available, only: {set(docker_tags)}")
            logger.info(
                f"missing docker images: "
                f"{str(set(d.docker_tag for d in definitions).difference(docker_tags))}"
            )
            definitions = [d for d in definitions if d.docker_tag in docker_tags]

    # Case 2: The user is working without Docker, i.e. "on the local machine".
    else:

        # Check that all the modules referenced in "algos.yaml" can actually
        # be loaded.
        def _test(df):
            status = algorithm_status(df)
            # If the module was loaded but doesn't actually have a constructor
            # of the right name, then the definition is broken
            if status == InstantiationStatus.NO_CONSTRUCTOR:
                raise Exception(
                    "%s.%s(%s): error: the module '%s' does not"
                    " expose the named constructor"
                    % (df.module, df.constructor, df.arguments, df.module)
                )

            if status == InstantiationStatus.NO_MODULE:
                # If the module couldn't be loaded (presumably because
                # of a missing dependency), print a warning and remove
                # this definition from the list of things to be run
                logging.warning(
                    "%s.%s(%s): the module '%s' could not be "
                    "loaded; skipping"
                    % (df.module, df.constructor, df.arguments, df.module)
                )
                return False
            else:
                return True

        # Keep the modules that can be loaded:
        definitions = [d for d in definitions if _test(d)]

    #  For debugging, the user may wish to only run the first "n" methods:
    if args.max_n_algorithms >= 0:
        definitions = definitions[: args.max_n_algorithms]

    if len(definitions) == 0:
        raise Exception("Nothing to run")
    else:
        logger.info(f"Order: {definitions}")

    # Multiprocessing magic to farm this out to all CPUs
    queue = multiprocessing.Queue()
    for definition in definitions:
        queue.put(definition)

    workers = [
        multiprocessing.Process(target=run_worker, args=(i + 1, args, queue))
        for i in range(1)
    ]
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]

    # TODO: need to figure out cleanup handling here
