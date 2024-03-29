import argparse
import json
import logging
import os
import threading
import time
import traceback

import colors
import docker
import numpy
import psutil

from kernel_matrix_benchmarks.definitions import (
    Definition,
    instantiate_algorithm,
)
from kernel_matrix_benchmarks.results import CustomJSONizer
from kernel_matrix_benchmarks.datasets import get_dataset, DATASETS
from kernel_matrix_benchmarks.results import store_result


def run(*, definition, dataset, runs):
    """Runs a method "runs" times."""

    # Load the input data from the HDF5 file:
    f, _ = get_dataset(dataset)
    # N.B.: - The specification of our dataset format is detailed in datasets.py.
    #       - The f["key"][:] syntax forces the conversion of the data
    #         to a NumPy array, loaded in RAM.
    source_points = f["source_points"][:]  # (M,D) float64 array
    target_points = f["target_points"][:]  # (N,D) float64 array
    source_signal = f["source_signal"][:]  # (M,E) float64 array
    target_signal = f["target_signal"][:]  # (N,E) float64 array
    M, D = source_points.shape
    N, E = target_signal.shape

    # Metadata:
    point_type = f.attrs["point_type"]  # = "float", usually
    kernel = f.attrs["kernel"]  # = "gaussian", "inverse-distance", etc.
    same_points = f.attrs["same_points"]  # = True if source_points = target_points
    normalize_rows = f.attrs["normalize_rows"]  # = False, usually
    density_estimation = f.attrs["density_estimation"]  # = False, usually

    # Don't forget to close the file:
    f.close()

    # Display some info:
    print("")
    print("========================================")
    print(
        f"M={M:,} source points, N={N:,} target points in dimension {D:,} "
        f"with a signal of dimension E={E:,}."
    )
    print(f"kernel='{kernel}'")
    print(
        f"same_points? {same_points}, "
        f"normalize_rows? {normalize_rows}, "
        f"density_estimation? {density_estimation}."
    )

    # We run our algorithm in a try-catch structure:
    try:

        # We try an experiment "runs" times and keep the best run time:
        algo = None
        build_time = float("inf")
        mem_footprint = float("inf")

        for i in range(runs):

            # Step 0: instantiate the algorithm
            _algo = instantiate_algorithm(definition)

            # Step 1: Pre-computation, benchmarked both for time and memory usage.
            # Step 1.a: Load data, outside of the timer
            if _algo.task == "product":
                _algo.prepare_data(
                    source_points=source_points,
                    target_points=target_points,
                    same_points=same_points,
                    density_estimation=density_estimation,
                )
                query_data = {"source_signal": source_signal}
                true_answer = target_signal

            elif _algo.task == "solver":
                _algo.prepare_data(source_points=source_points)
                query_data = {"target_signal": target_signal}
                true_answer = source_signal

            else:
                raise NotImplementedError()

            #  Step 1.b: Actual pre-computation, within the timer
            memory_usage_before = _algo.get_memory_usage()
            t0 = time.time()
            _algo.fit()
            _build_time = time.time() - t0
            _mem_footprint = _algo.get_memory_usage() - memory_usage_before

            if _build_time <= build_time:
                algo = _algo
                build_time = _build_time
                mem_footprint = _mem_footprint

        print(f"Precomputation done in {build_time:.2e}s.")
        print(f"Memory usage: {mem_footprint:.2e}kB.")

        # Step 2: We may run the same "trained" algorithm with different parameters
        # "at query" time.
        query_argument_groups = definition.query_argument_groups
        # Make sure that algorithms with no query argument groups still get run
        # once by providing them with a single, empty, harmless group
        if not query_argument_groups:
            query_argument_groups = [{}]

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print(
                "Running query argument group %d of %d..."
                % (pos, len(query_argument_groups))
            )
            algo.set_query_arguments(**query_arguments)

            # Step 3: Actual query ---------------------------------------------
            # We try an experiment "runs" times and keep the best run time:
            query_time = float("inf")
            result = None

            for i in range(runs):
                print("Run %d/%d..." % (i + 1, runs))

                # To ensure a fair benchmark, we may need to reformat queries
                # before launching the timer, e.g. to load them on the GPU or change the input format:
                algo.prepare_query(**query_data)

                # Actual benchmark!
                start = time.time()
                algo.query()
                _query_time = time.time() - start

                # Unload the output from the GPU, etc.
                _result = algo.get_result()

                # We are interested in the "minimum query time" among all runs:
                if _query_time <= query_time:
                    query_time = _query_time
                    result = _result

            # Return all types of metadata (attrs) and the results array:
            attrs = dict(
                {
                    "dataset": dataset,
                    "algo": definition.algorithm,
                    "name": str(algo),
                    "kernel": kernel,
                    "run_count": runs,
                    "build_time": build_time,
                    "query_time": query_time,
                    "memory_footprint": mem_footprint,
                },
                **algo.get_additional(),
            )

            # Store the raw output of the algorithm.
            store_result(
                dataset=dataset,
                definition=definition,
                query_arguments=query_arguments,
                result=result,
                error=result - true_answer,
                attrs=attrs,
            )
    finally:
        if algo is not None:
            algo.done()


def run_from_cmdline():
    """Command-Line Interface used by the Docker runner."""

    # In practice, the user calls "run.py" -> "main.py" -> "run_worker(...)".
    # If Docker is being used, "run_worker(...)" calls "run_docker(...)".
    # The entry point for each method's Dockerfile is "run_algorithm.py",
    # that points to this function "run_from_cmdline()"".
    parser = argparse.ArgumentParser(
        """NOTICE: You probably want to use run.py rather than this script."""
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        help=f"Dataset to benchmark on.",
        required=True,
    )
    parser.add_argument(
        "--algorithm", help="Name of algorithm for saving the results.", required=True
    )
    parser.add_argument(
        "--module",
        help='Python module containing algorithm. E.g. "kernel_matrix_benchmarks.algorithms.bruteforce"',
        required=True,
    )
    parser.add_argument(
        "--constructor",
        help='Constructer to load from module. E.g. "BruteForceProductBLAS"',
        required=True,
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "build",
        help='JSON of arguments to pass to the constructor. E.g. {"precision": "float32"}',
    )
    parser.add_argument(
        "queries",
        help='JSON of a list of arguments to pass prior to the queries. E.g. [{"H": 5}, {"H":10}]',
        nargs="*",
        default=[{}],
    )
    args = parser.parse_args()
    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
    )

    # Presumably, we execute this command inside a Docker:
    run(definition=definition, dataset=args.dataset, runs=args.runs)


def run_docker(*, definition, dataset, runs, timeout, cpu_limit, mem_limit=None):
    # Arguments for the entry point defined in "install/Dockerfile",
    # the parent Docker image for all our models:
    # `python3 -u run_algorithm.py` + the options below
    # This links to the function "run_from_cmdline()" defined above.
    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
    ]

    # Arguments of the "constructor":
    cmd.append(json.dumps(definition.arguments, cls=CustomJSONizer))
    # Arguments at query time (for a fixed constructor):
    cmd += [
        json.dumps(qag, cls=CustomJSONizer) for qag in definition.query_argument_groups
    ]

    # Wake-up Docker:
    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available

    # Run our command:
    # N.B.: We expect the user to be called "app", not e.g. "jean" or "ubuntu".
    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath("kernel_matrix_benchmarks"): {
                "bind": "/home/app/kernel_matrix_benchmarks",
                "mode": "ro",
            },
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
        },
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        detach=True,
    )

    # Logging:
    # kmb = Kernel Matrix Benchmarks
    logger = logging.getLogger(f"kmb.{container.short_id}")

    logger.info(
        "Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s"
        % (container.short_id, cpu_limit, mem_limit, timeout, cmd)
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    # Launch the container, wait at most timeout seconds (2 * 10mn by default):
    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except:
        logger.error(
            "Container.wait for container %s failed with exception" % container.short_id
        )
        traceback.print_exc()
    finally:
        container.remove(force=True)


def _handle_container_return_value(return_value, container, logger):
    """Displays an error message if the Docker returned an error."""

    base_msg = "Child process for container %s" % (container.short_id)
    if (
        type(return_value) is dict
    ):  # The return value from container.wait changes from int to dict in docker 3.0.0
        error_msg = return_value["Error"]
        exit_code = return_value["StatusCode"]
        msg = base_msg + "returned exit code %d with message %s" % (
            exit_code,
            error_msg,
        )
    else:
        exit_code = return_value
        msg = base_msg + "returned exit code %d" % (exit_code)

    if exit_code not in [0, None]:
        logger.error(colors.color(container.logs().decode(), fg="red"))
        logger.error(msg)
