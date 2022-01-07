from __future__ import absolute_import

import itertools
import numpy
from kernel_matrix_benchmarks.plotting.metrics import all_metrics as metrics


def get_or_create_metrics(run):
    if "metrics" not in run:
        run.create_group("metrics")
    return run["metrics"]


def create_pointset(data, xn, yn):
    """Extracts the Pareto frontier of a set of performance metrics.

    Args:
        data (list): list of (algo, algo_name, x_value, y_value).
            x_value and y_value are numerical performance "grades",
            while algo_name will be used as a point label on the interactive website.
        xn (string): name of the performance metric for the x axis.
        yn (string): name of the performance metric for the y axis.

    Returns:
        6-uple of lists: (xs, ys, labels) that correspond to the Pareto front
          and (all_xs, all_ys, all_labels) that correspond to all points
          outside of the x and y axes, sorted "from best to worst" value of y.
    """
    # Load the relevant "metrics" functions for the x and y axes.
    xm, ym = (metrics[xn], metrics[yn])
    # Shall we compute the Pareto frontiers "upside down"?
    # This is typically the case for "recall" or "queries per second" metrics,
    # but not for "errors".
    rev_y = -1 if ym["worst"] < 0 else 1
    rev_x = -1 if xm["worst"] < 0 else 1
    # Sort the list of values according to the last two coordinates:
    # the values "yv" (most important) and "xv" (to break ties).
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    axs, ays, als = [], [], []
    # Generate Pareto frontier:
    xs, ys, ls = [], [], []
    last_x = xm["worst"]
    comparator = (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)

    # Loop over all points in the benchmark:
    # We sweep "from the best values of y to the worst ones".
    for algo, algo_name, xv, yv in data:
        if not xv or not yv:  # zero values -> skip
            continue
        axs.append(xv)
        ays.append(yv)
        als.append(algo_name)

        # We sweep "from the best values of y to the worst ones".
        # Along the way, we pick up points that have "the best value of x"
        # seen so far.
        if comparator(xv, last_x):  # Is xv better than last_x?
            last_x = xv
            xs.append(xv)
            ys.append(yv)
            ls.append(algo_name)
    return xs, ys, ls, axs, ays, als


def compute_metrics(true_nn_distances, res, metric_1, metric_2, recompute=False):
    """Computes a list of (x,y) values to fill our plots.

    Args:
        true_nn_distances (array): ground truth values.
        res (list of (dict, hdf5 file)): list of results per run.
        metric_1 (string): name of the property to put on the x axis.
        metric_2 (string): name of the property to put on the y axis.
        recompute (bool, optional): shall we recompute metrics if they
            are already in the attributes of res? Defaults to False.

    Returns:
        dict of {algo: list}: for each algorithm, a list of 4-uples
            that contain the (algo, algo_name, x_value, y_value)
            for the requested performance metrics.
    """
    all_results = {}
    for i, (properties, run) in enumerate(res):
        # Properties is a dict, run is an hdf5 file:
        algo = properties["algo"]
        algo_name = properties["name"]
        # Cache distances to avoid access to the hdf5 file:
        run_distances = numpy.array(run["distances"])
        if recompute and "metrics" in run:
            del run["metrics"]
        metrics_cache = get_or_create_metrics(run)

        # Compute the metrics by comparing the ground truth results
        # with the experiment results ("run_distances")
        # and miscellaneous performance metrics ("metrics_cache")
        # such as query time, etc.
        metric_1_value = metrics[metric_1]["function"](
            true_nn_distances, run_distances, metrics_cache, properties
        )
        metric_2_value = metrics[metric_2]["function"](
            true_nn_distances, run_distances, metrics_cache, properties
        )

        print(
            "%3d: %80s %12.3f %12.3f" % (i, algo_name, metric_1_value, metric_2_value)
        )

        # Append the result to all_results["algo"], which is initially set to []:
        all_results.setdefault(algo, []).append(
            (algo, algo_name, metric_1_value, metric_2_value)
        )

    return all_results


def compute_all_metrics(true_nn_distances, run, properties, recompute=False):
    """Evaluates all metrics for a given experiment.

    Args:
        true_nn_distances (array): ground truth values.
        run (hdf5 file): hdf5 file that stores the output of an experiment.
        properties (dict): properties of the experiment with keys "algo" and "name".
        recompute (bool, optional): shall we recompute metrics if they
            are already in the attributes of res? Defaults to False.

    Returns:
        (str, str, dict) 3-uple: algo, algo_name and a {metric: value} dict.
    """
    algo = properties["algo"]
    algo_name = properties["name"]
    print("--")
    print(algo_name)
    results = {}
    # Cache distances to avoid access to the hdf5 file:
    run_distances = numpy.array(run["distances"])
    if recompute and "metrics" in run:
        del run["metrics"]
    metrics_cache = get_or_create_metrics(run)

    # Apply every possible metric (from "metrics = all_metrics") on the hdf5 file:
    for name, metric in metrics.items():
        v = metric["function"](
            true_nn_distances, run_distances, metrics_cache, properties
        )
        results[name] = v
        if v:
            print("%s: %g" % (name, v))
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
    if metric["worst"] == float("inf"):
        return "down"
    return "up"


def get_left_right(metric):
    if metric["worst"] == float("inf"):
        return "left"
    return "right"


def get_plot_label(xm, ym):
    template = (
        "%(xlabel)s-%(ylabel)s tradeoff - %(updown)s and"
        " to the %(leftright)s is better"
    )
    return template % {
        "xlabel": xm["description"],
        "ylabel": ym["description"],
        "updown": get_up_down(ym),
        "leftright": get_left_right(xm),
    }
