import matplotlib as mpl

mpl.use("Agg")  # noqa
import argparse
import os
import json
import pickle
import yaml
import numpy
import hashlib
from jinja2 import Environment, FileSystemLoader

from kernel_matrix_benchmarks import results
from kernel_matrix_benchmarks.datasets import get_dataset
from kernel_matrix_benchmarks.plotting.plot_variants import (
    all_plot_variants as plot_variants,
)
from kernel_matrix_benchmarks.plotting.metrics import all_metrics as metrics
from kernel_matrix_benchmarks.plotting.utils import (
    get_plot_label,
    compute_all_metrics,
    create_pointset,
    create_linestyles,
)
import plot

colors = [
    "rgba(166,206,227,1)",
    "rgba(31,120,180,1)",
    "rgba(178,223,138,1)",
    "rgba(51,160,44,1)",
    "rgba(251,154,153,1)",
    "rgba(227,26,28,1)",
    "rgba(253,191,111,1)",
    "rgba(255,127,0,1)",
    "rgba(202,178,214,1)",
]

point_styles = {
    "o": "circle",
    "<": "triangle",
    "*": "star",
    "x": "cross",
    "+": "rect",
}


def convert_color(color):
    r, g, b, a = color
    return "rgba(%(r)d, %(g)d, %(b)d, %(a)d)" % {
        "r": r * 255,
        "g": g * 255,
        "b": b * 255,
        "a": a,
    }


def convert_linestyle(ls):
    new_ls = {}
    for algo in ls.keys():
        algostyle = ls[algo]
        new_ls[algo] = (
            convert_color(algostyle[0]),
            convert_color(algostyle[1]),
            algostyle[2],
            point_styles[algostyle[3]],
        )
    return new_ls


def directory_path(s):
    if not os.path.isdir(s):
        os.makedirs(s)
    return s + "/"


def prepare_data(*, data, x_name, y_name):
    """Change format from {"algo": ..., "algo_name": ..., "metrics": dict} to (algo, instance, x, y)."""
    res = []
    for met in data:
        res.append(
            (
                met["algo"],
                met["algo_name"],
                met["metrics"][x_name],
                met["metrics"][y_name],
            )
        )
    return res


parser = argparse.ArgumentParser()
parser.add_argument(
    "--plottype",
    help="Generate only the plots specified",
    nargs="*",
    choices=plot_variants.keys(),
    default=plot_variants.keys(),
)
parser.add_argument(
    "--outputdir",
    help="Select output directory",
    default="website",
    type=directory_path,
    action="store",
)
parser.add_argument(
    "--latex", help="generates latex code for each plot", action="store_true"
)
parser.add_argument(
    "--scatter", help="create scatterplot for data", action="store_true"
)
parser.add_argument(
    "--recompute",
    help="Clears the cache and recomputes the metrics",
    action="store_true",
)
args = parser.parse_args()


def get_lines(*, data, x_name, y_name, render_all_points):
    """For each algorithm run on a dataset, obtain its performance curve coords.
    
    x_name, y_name are string identifiers for performance metrics,
    i.e. keys for the dict "all_metrics"
    defined in kernel_matrix_benchmarks/plotting/metrics.py.
    """
    plot_data = []
    for algo in sorted(data.keys(), key=lambda x: x.lower()):
        points = create_pointset(
            data=prepare_data(data=data[algo], x_name=x_name, y_name=y_name),
            x_name=x_name,
            y_name=y_name,
        )["all" if render_all_points else "front"]
        plot_data.append(
            {
                "name": algo,  # Label of the "line" or "scatter" plot.
                "coords": zip(points["x"], points["y"]),  # Coordinates in the plots.
                # Hover label on points in the interactive visualization:
                "labels": points["labels"],
                "scatter": render_all_points,  # Scatter vs. line plot
            }
        )
    return plot_data


def create_plot(
    *, data, x_name, y_name, linestyle, j2_env, additional_label="", plottype="line"
):
    # Load the relevant "metrics" functions for the x and y axes.
    x_metric, y_metric = (metrics[x_name], metrics[y_name])
    render_all_points = plottype == "bubble"  # "line" vs "bubble"
    # Extract the Pareto frontier vs just retrieve the full point set:
    plot_data = get_lines(
        data=data, x_name=x_name, y_name=y_name, render_all_points=render_all_points
    )

    # Insert the point coordinates in a tikzpicture:
    latex_code = j2_env.get_template("latex.template").render(
        plot_data=plot_data,
        caption=get_plot_label(x_metric, y_metric),
        xlabel=x_metric["description"],
        ylabel=y_metric["description"],
    )

    #  Mmmm... Do we really need to recompute the Pareto frontier here?
    # TODO: I leave it here just in case.
    plot_data = get_lines(
        data=data, x_name=x_name, y_name=y_name, render_all_points=render_all_points
    )

    button_label = hashlib.sha224(
        (get_plot_label(x_metric, y_metric) + additional_label).encode("utf-8")
    ).hexdigest()

    # TODO: currently, all the details plot have a linear x axis
    #       and a logarithmic y axis.
    # Insert the point coordinates in a javascript interactive plot:
    return j2_env.get_template("chartjs.template").render(
        args=args,
        latex_code=latex_code,
        button_label=button_label,
        data_points=plot_data,
        xlabel=x_metric["description"],
        ylabel=y_metric["description"],
        plottype=plottype,
        plot_label=get_plot_label(x_metric, y_metric),
        label=additional_label,
        linestyle=linestyle,
        render_all_points=render_all_points,
    )


def build_detail_site(*, full_data, j2_env, linestyles):
    """Builds a detailed interactive page for every algorithm and dataset."""

    for (name, data) in full_data.items():
        print("Building '%s'" % name)
        title = data["title"]
        point_data = {"normal": [], "scatter": []}

        # Loop over the required pairs of (xaxis, yaxis):
        for plottype in args.plottype:
            # Select the names of the variables on the "x" and "y" axes.
            x_name, y_name = plot_variants[plottype]

            # Javascript rendering of the runs in full_data[name]["runs"]
            # (Pareto fronts only):
            point_data["normal"].append(
                create_plot(
                    data=data["runs"],
                    x_name=x_name,
                    y_name=y_name,
                    linestyle=convert_linestyle(linestyles),
                    j2_env=j2_env,
                )
            )
            # If required, we also display the full point clouds.
            # This is especially useful when tuning the experiments
            # in algos.yaml.
            if args.scatter:
                # Javascript rendering of the runs in full_data[name]["runs"]
                # (all performance points, including the sub-optimal ones):
                point_data["scatter"].append(
                    create_plot(
                        data=data["runs"],
                        x_name=x_name,
                        y_name=y_name,
                        linestyle=convert_linestyle(linestyles),
                        j2_env=j2_env,
                        additional_label="Scatterplot ",
                        plottype="bubble",
                    )
                )

        # Create a .png plot for the summary page:
        data_for_plot = {}
        for k in data["runs"].keys():
            data_for_plot[k] = prepare_data(
                data=data["runs"][k], x_name="total-time", y_name="rmse-error"
            )

        plot.create_plot(
            data=data_for_plot,
            raw=False,
            x_scale="log",
            y_scale="log",
            x_name="total-time",
            y_name="rmse-error",
            fn_out=args.outputdir + name + ".png",
            linestyles=linestyles,
        )

        # Final render of the detailed page:
        output_path = args.outputdir + name + ".html"
        with open(output_path, "w") as text_file:
            text_file.write(
                j2_env.get_template("detail_page.html").render(
                    title=title, plot_data=point_data, args=args
                )
            )


def build_index_site(*, runs, j2_env, file_name="index.html"):
    """Builds the front page of our website (index.html)."""

    with open(args.outputdir + file_name, "w") as text_file:
        text_file.write(
            j2_env.get_template("summary.html").render(
                title="Kernel-Matrix-Benchmarks", runs=runs,
            )
        )


def load_all_results():
    """Read all result files and compute all metrics.
    
    Returns a dict: all_runs = {
            "by_dataset" : {
                "dataset-1": {
                    "title": "Dataset 1";
                    "runs": {
                        "algo-1": [
                            {"metric-1": val1, "metric-2": val2, ...},  # Run for params1
                            {"metric-1": val1, "metric-2": val2, ...},  # Run for params2
                            ...
                        ],
                        "algo-2": [
                            ...
                        ],
                        ...
                    }
                },
                "dataset-2": {
                    ...
                }
            },
            "by_algorithm" : {
                "algo-1": {
                    "title": "Algo-1",
                    "runs": {
                        "dataset-1": [
                            {"metric-1": val1, "metric-2": val2, ...},  # Run for params1
                            {"metric-1": val1, "metric-2": val2, ...},  # Run for params2
                            ...
                        ],
                        "dataset-2": [
                            ...
                        ],
                        ...
                    }
                },
                "algo-2": {
                    ...
                }
            }
            }
    """

    all_runs = {"by_dataset": {}, "by_algorithm": {}}
    old_sdn = None

    # f is an hdf5 file, properties = dict(f.attrs) contains the relevant metadata:
    for properties, f in results.load_all_results():
        # String identifier for a problem (dataset, kernel, ...):
        dataset_name = properties["dataset"]
        dataset_file, _ = get_dataset(dataset_name)

        algo = properties["algo"]
        ms = compute_all_metrics(
            dataset=dataset_file, run=f, properties=properties, recompute=args.recompute
        )

        ms["short_description"] = 2

        # Put the run in a dict sorted by dataset...
        all_runs["by_dataset"].setdefault(
            dataset_name, {"title": dataset_file.attrs["description"], "runs": {}}
        )["runs"].setdefault(algo, []).append(ms)
        # ... and in a dict sorted by method:
        all_runs["by_algorithm"].setdefault(algo, {"title": algo, "runs": {}})[
            "runs"
        ].setdefault(dataset_name, []).append(ms)

        # Don't forget to close the dataset file:
        dataset_file.close()

    return all_runs


# We use the Jinja templating system:
j2_env = Environment(loader=FileSystemLoader("./templates/"), trim_blocks=True)
j2_env.globals.update(zip=zip, len=len)

# Load the data:
runs = load_all_results()
# Retrieve the names of the datasets - each of whom receives a detailed page:
dataset_names = list(runs["by_dataset"].keys())
# Retrieve the names of the algorithms - each of whom receives a detailed page:
algorithm_names = list(runs["by_algorithm"].keys())

linestyles = {**create_linestyles(dataset_names), **create_linestyles(algorithm_names)}

# Detailed pages for datasets:
build_detail_site(full_data=runs["by_dataset"], j2_env=j2_env, linestyles=linestyles)

# Detailed pages for algorithms:
build_detail_site(full_data=runs["by_algorithm"], j2_env=j2_env, linestyles=linestyles)

# Index page:
build_index_site(runs=runs, j2_env=j2_env, file_name="index.html")
