import os
import matplotlib as mpl

mpl.use("Agg")  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse

from kernel_matrix_benchmarks.datasets import get_dataset
from kernel_matrix_benchmarks.definitions import get_definitions
from kernel_matrix_benchmarks.plotting.metrics import all_metrics as metrics
from kernel_matrix_benchmarks.plotting.utils import (
    get_plot_label,
    compute_metrics,
    create_linestyles,
    create_pointset,
)
from kernel_matrix_benchmarks.results import (
    load_all_results,
    get_unique_algorithms,
)


def create_plot(*, data, raw, x_scale, y_scale, x_name, y_name, fn_out, linestyles):
    """Creates a .png file and save it at location 'fn_out'.
    
    This routine is called using the command line, thanks to the API
    that is defined at the end of this file.
    """
    x_metric, y_metric = (metrics[x_name], metrics[y_name])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        points = create_pointset(data=data[algo], x_name=x_name, y_name=y_name)
        return -np.log(np.array(points["front"]["y"])).mean()

    for algo in sorted(data.keys(), key=mean_y):
        points = create_pointset(data=data[algo], x_name=x_name, y_name=y_name)
        min_x = min(points["front"]["x"])
        max_x = max(points["front"]["x"])
        color, faded, linestyle, marker = linestyles[algo]
        (handle,) = plt.plot(
            points["front"]["x"],
            points["front"]["y"],
            label=algo,
            color=color,
            ms=7,
            mew=3,
            lw=3,
            linestyle=linestyle,
            marker=marker,
        )
        handles.append(handle)
        if raw:
            (handle2,) = plt.plot(
                points["all"]["x"],
                points["all"]["y"],
                label=algo,
                color=faded,
                ms=5,
                mew=2,
                lw=2,
                linestyle=linestyle,
                marker=marker,
            )
        labels.append(algo)

    ax = plt.gca()
    ax.set_ylabel(y_metric["description"])
    ax.set_xlabel(x_metric["description"])
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(get_plot_label(x_metric, y_metric))
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9}
    )
    plt.grid(b=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    if "lim" in x_metric:
        plt.xlim(x_metric["lim"])
    if "lim" in y_metric:
        plt.ylim(y_metric["lim"])

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines["bottom"]._adjust_location()

    plt.savefig(fn_out, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-25-angular")
    parser.add_argument(
        "--definitions",
        metavar="FILE",
        help="load algorithm definitions from FILE",
        default="algos.yaml",
    )
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "-x",
        "--x-axis",
        help="Which metric to use on the X-axis",
        choices=metrics.keys(),
        default="total-time",
    )
    parser.add_argument(
        "-y",
        "--y-axis",
        help="Which metric to use on the Y-axis",
        choices=metrics.keys(),
        default="rmse-error",
    )
    parser.add_argument(
        "-X",
        "--x-scale",
        help="Scale to use when drawing the X-axis.",
        choices=["linear", "log", "symlog", "logit"],
        default="log",
    )
    parser.add_argument(
        "-Y",
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="log",
    )
    parser.add_argument(
        "--raw",
        help="Show raw results (not just Pareto frontier) in faded colours",
        action="store_true",
    )
    parser.add_argument(
        "--recompute",
        help="Clears the cache and recomputes the metrics",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.output:
        args.output = "results/%s.png" % (args.dataset)
        print("writing output to %s" % args.output)

    dataset, _ = get_dataset(args.dataset)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(dataset=args.dataset)
    linestyles = create_linestyles(sorted(unique_algorithms))
    runs = compute_metrics(
        dataset=dataset,
        results=results,
        x_name=args.x_axis,
        y_name=args.y_axis,
        recompute=args.recompute,
    )
    if not runs:
        raise Exception("Nothing to plot")

    create_plot(
        all_data=runs,
        raw=args.raw,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        x_name=args.x_axis,
        y_name=args.y_axis,
        fn_out=args.output,
        linestyles=linestyles,
    )
