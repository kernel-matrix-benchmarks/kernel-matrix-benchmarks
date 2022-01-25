from __future__ import absolute_import

import itertools
import numpy
from kernel_matrix_benchmarks.plotting.metrics import all_metrics as metrics


def get_or_create_metrics(run):
    """Returns the "subfolder" of the HDF5 file "run/metric" where metrics may be stored."""
    if "metrics" not in run:
        run.create_group("metrics")
    return run["metrics"]


def create_pointset(*, data, x_name, y_name):
    """Extracts the Pareto frontier of a set of performance metrics.

    Args:
        data (list): list of (algo, algo_name, x_value, y_value).
            x_value and y_value are numerical performance "grades",
            while algo_name will be used as a point label on the interactive website.
        x_name (string): name of the performance metric for the x axis.
        y_name (string): name of the performance metric for the y axis.

    Returns:
        dict of dict of lists: 
          points["front"]["x"], points["front"]["y"] and points["front"]["labels"]
          correspond to the Pareto front and
          points["all"]["x"], points["all"]["y"] and points["all"]["labels"]
          correspond to all points outside of the x and y axes, 
          sorted "from best to worst" value of y.
    """
    # Load the relevant "metrics" functions for the x and y axes.
    x_metric, y_metric = (metrics[x_name], metrics[y_name])
    # Shall we compute the Pareto frontiers "upside down"?
    # This is typically the case for "recall" or "queries per second" metrics,
    # but not for "errors".
    rev_y = -1 if y_metric["worst"] < 0 else 1
    rev_x = -1 if x_metric["worst"] < 0 else 1
    # Sort the list of values according to the last two coordinates:
    # the values "yv" (most important) and "xv" (to break ties).
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    points = {
        "front": {"x": [], "y": [], "labels": [],},
        "all": {"x": [], "y": [], "labels": [],},
    }
    last_x = x_metric["worst"]
    comparator = (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)

    # Loop over all points in the benchmark:
    # We sweep "from the best values of y to the worst ones".
    for algo, algo_name, xv, yv in data:
        if not xv or not yv:  # zero values -> skip
            continue
        points["all"]["x"].append(xv)
        points["all"]["y"].append(yv)
        points["all"]["labels"].append(algo_name)

        # We sweep "from the best values of y to the worst ones".
        # Along the way, we pick up points that have "the best value of x"
        # seen so far.
        if comparator(xv, last_x):  # Is xv better than last_x?
            last_x = xv
            points["front"]["x"].append(xv)
            points["front"]["y"].append(yv)
            points["front"]["labels"].append(algo_name)
    return points


def compute_metrics(*, dataset, results, x_name, y_name, recompute=False):
    """Computes a list of (x,y) values to fill our plots.

    Args:
        dataset (HDF5 file): original file, that contains ground truth values.
            This may be useful if e.g. we need to compute the residual error
            associated to a solver by making a bruteforce kernel computation
            on a set of target points.
        results (list of (dict, hdf5 file)): list of results per run.
        x_name (string): name of the property to put on the x axis.
        y_name (string): name of the property to put on the y axis.
        recompute (bool, optional): shall we recompute metrics if they
            are already in the attributes of res? Defaults to False.

    Returns:
        dict of {algo: list}: for each algorithm, a list of 4-uples
            that contain the (algo, algo_name, x_value, y_value)
            for the requested performance metrics.
    """
    all_results = {}
    for i, (properties, run) in enumerate(results):
        # Properties is a dict, run is an hdf5 file:
        algo = properties["algo"]  # str, as in algos.yaml
        algo_name = properties["name"]  # str, attribute "algo.name" defined in Python
        # Cache in RAM the results and errors (= result - true_answer)
        # using the "file["key"][:]" syntax to avoid access to
        # the hdf5 file on the hard drive:
        result = run["result"][:]
        error = run["error"][:]

        # The HDF5 file may contain information about the metrics.
        #  If required, we delete it:
        if recompute and "metrics" in run:
            del run["metrics"]
        #  And eventually, load it again:
        metrics_cache = get_or_create_metrics(run)

        # Compute the metrics using the original dataset (if needed),
        # the output of the algorithm, the difference with the ground truth value,
        # the metadata of the experiment and, possibly, a cache of pre-computed
        # metrics:
        x_value = metrics[x_name]["function"](
            dataset=dataset,
            result=result,
            error=error,
            properties=properties,
            metrics_cache=metrics_cache,
        )
        y_value = metrics[y_name]["function"](
            dataset=dataset,
            result=result,
            error=error,
            properties=properties,
            metrics_cache=metrics_cache,
        )

        print("%3d: %80s %12.3f %12.3f" % (i, algo_name, x_value, y_value))

        # Append the result to all_results["algo"], which is initially set to []:
        all_results.setdefault(algo, []).append((algo, algo_name, x_value, y_value))

    return all_results


def compute_all_metrics(dataset, run, properties, recompute=False):
    """Evaluates all metrics for a given experiment.

    Args:
        dataset (HDF5 file): original file, that contains ground truth values.
            This may be useful if e.g. we need to compute the residual error
            associated to a solver by making a bruteforce kernel computation
            on a set of target points.
        run (hdf5 file): hdf5 file that stores the output of an experiment.
        properties (dict): properties of the experiment with keys "algo" and "name".
            In practice, properties = dict(run.attrs).
        recompute (bool, optional): shall we recompute metrics if they
            are already in the attributes of run? Defaults to False.

    Returns:
        (str, str, dict) 3-uple: algo, algo_name and a {metric: value} dict.
    """

    algo = properties["algo"]  # str, as in algos.yaml
    algo_name = properties["name"]  # str, attribute "algo.name" defined in Python
    print("--")
    print(algo_name)
    results = {}

    # Cache in RAM the results and errors (= result - true_answer)
    # using the "file["key"][:]" syntax to avoid access to
    # the hdf5 file on the hard drive:
    result = run["result"][:]
    error = run["error"][:]

    # The HDF5 file may contain information about the metrics.
    # If required, we delete it:
    if recompute and "metrics" in run:
        del run["metrics"]
    #  And eventually, load it again:
    metrics_cache = get_or_create_metrics(run)

    # Apply every possible metric (from "metrics = all_metrics") on the hdf5 file:
    for name, metric in metrics.items():
        # Compute the metrics using the original dataset (if needed),
        # the output of the algorithm, the difference with the ground truth value,
        # the metadata of the experiment and, possibly, a cache of pre-computed
        # metrics:
        value = metric["function"](
            dataset=dataset,
            result=result,
            error=error,
            properties=properties,
            metrics_cache=metrics_cache,
        )
        results[name] = value
        if value:
            print("%s: %g" % (name, value))
    return (algo, algo_name, results)


def generate_n_colors(n):
    """
    Creates n distinct colors by farthest point sampling in a domain of the RGB cube.

    Args:
        n (int): number of distinct colors.

    Returns:
        list of 4-uples of floats: list of RGBA colors with values in [0,1].
    """
    # Our grid for possible colors is [.3, .4, .5,..., .9]^3
    vs = numpy.linspace(0.3, 0.9, 7)

    #  Our first color - reddish salmon:
    colors = [(0.9, 0.4, 0.4, 1.0)]

    def euclidean(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    while len(colors) < n:
        # Select a color in our discrete domain which is as far as possible
        # from the previous choices:
        new_color = max(
            itertools.product(vs, vs, vs),
            key=lambda a: min(euclidean(a, b) for b in colors),
        )
        colors.append(new_color + (1.0,))
    return colors


def create_linestyles(unique_algorithms):
    """Generates distinct linestyles for a list of algorithm names.

    Args:
        unique_algorithms (list of str): list of algorithm names.

    Returns:
        dict: {algo: ((r,g,b,1), (r,g,b,.3), linestyle, markerstyle)}
    """
    colors = dict(zip(unique_algorithms, generate_n_colors(len(unique_algorithms))))

    # N.B.: 4 and 5 are coprime, which ensures that we loop among all 20 combinations:
    linestyles = dict(
        (algo, ["--", "-.", "-", ":"][i % 4])
        for i, algo in enumerate(unique_algorithms)
    )
    markerstyles = dict(
        (algo, ["+", "<", "o", "*", "x"][i % 5])
        for i, algo in enumerate(unique_algorithms)
    )
    faded = dict((algo, (r, g, b, 0.3)) for algo, (r, g, b, a) in colors.items())
    return dict(
        (algo, (colors[algo], faded[algo], linestyles[algo], markerstyles[algo]))
        for algo in unique_algorithms
    )


def get_up_down(metric):
    if metric["worst"] > 0:
        return "down"
    return "up"


def get_left_right(metric):
    if metric["worst"] > 0:
        return "left"
    return "right"


def get_plot_label(x_metric, y_metric):
    template = (
        "%(xlabel)s-%(ylabel)s tradeoff - %(updown)s and"
        " to the %(leftright)s is better"
    )
    return template % {
        "xlabel": x_metric["description"],
        "ylabel": y_metric["description"],
        "leftright": get_left_right(x_metric),
        "updown": get_up_down(y_metric),
    }
