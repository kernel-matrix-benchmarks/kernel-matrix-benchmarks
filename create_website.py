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
    compute_metrics,
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


def get_run_desc(properties):
    # TODO: Reference to count is obsolete!
    return "%(dataset)s_%(count)d_%(distance)s" % properties


def get_dataset_from_desc(desc):
    return desc.split("_")[0]


def get_count_from_desc(desc):
    return desc.split("_")[1]


def get_distance_from_desc(desc):
    return desc.split("_")[2]


def get_dataset_label(desc):
    # TODO: Reference to count is obsolete!
    return "{} (k = {})".format(get_dataset_from_desc(desc), get_count_from_desc(desc))


def directory_path(s):
    if not os.path.isdir(s):
        os.makedirs(s)
    return s + "/"


def prepare_data(data, xn, yn):
    """Change format from (algo, instance, dict) to (algo, instance, x, y)."""
    res = []
    for algo, algo_name, result in data:
        res.append((algo, algo_name, result[xn], result[yn]))
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


def get_lines(all_data, xn, yn, render_all_points):
    """For each algorithm run on a dataset, obtain its performance
    curve coords.
    
    xn, yn are string identifiers for performance metrics,
    i.e. keys for the dict "all_metrics"
    defined in kernel_matrix_benchmarks/plotting/metrics.py.
    """
    plot_data = []
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        xs, ys, ls, axs, ays, als = create_pointset(
            prepare_data(all_data[algo], xn, yn), xn, yn
        )
        if render_all_points:
            xs, ys, ls = axs, ays, als
        plot_data.append(
            {
                "name": algo,  # Label of the "line" or "scatter" plot.
                "coords": zip(xs, ys),  # Coordinates in the plots.
                "labels": ls,  # Hover on points in the interactive visualization.
                "scatter": render_all_points,  # Scatter vs. line plot
            }
        )
    return plot_data


def create_plot(
    all_data, xn, yn, linestyle, j2_env, additional_label="", plottype="line"
):
    # Load the relevant "metrics" functions for the x and y axes.
    xm, ym = (metrics[xn], metrics[yn])
    render_all_points = plottype == "bubble"  # "line" vs "bubble"
    # Extract the Pareto frontier vs just retrieve the full point set:
    plot_data = get_lines(all_data, xn, yn, render_all_points)

    # Insert the point coordinates in a tikzpicture:
    latex_code = j2_env.get_template("latex.template").render(
        plot_data=plot_data,
        caption=get_plot_label(xm, ym),
        xlabel=xm["description"],
        ylabel=ym["description"],
    )

    #  Mmmm... Do we really need to recompute the Pareto frontier here?
    # TODO: I leave it here just in case.
    plot_data = get_lines(all_data, xn, yn, render_all_points)

    button_label = hashlib.sha224(
        (get_plot_label(xm, ym) + additional_label).encode("utf-8")
    ).hexdigest()

    # Insert the point coordinates in a javascript interactive plot:
    return j2_env.get_template("chartjs.template").render(
        args=args,
        latex_code=latex_code,
        button_label=button_label,
        data_points=plot_data,
        xlabel=xm["description"],
        ylabel=ym["description"],
        plottype=plottype,
        plot_label=get_plot_label(xm, ym),
        label=additional_label,
        linestyle=linestyle,
        render_all_points=render_all_points,
    )


def build_detail_site(data, label_func, j2_env, linestyles, batch=False):
    """Builds a detailed interactive page for every algorithm and dataset."""

    for (name, runs) in data.items():
        print("Building '%s'" % name)
        all_runs = runs.keys()
        label = label_func(name)
        data = {"normal": [], "scatter": []}

        # Loop over the required pairs of (xaxis, yaxis):
        for plottype in args.plottype:
            # Select the names of the variables on the "x" and "y" axes.
            xn, yn = plot_variants[plottype]
            # Display the Pareto fronts:
            data["normal"].append(
                create_plot(runs, xn, yn, convert_linestyle(linestyles), j2_env)
            )
            # If required, we also display the full point clouds.
            # This is especially useful when tuning the experiments
            # in algos.yaml.
            if args.scatter:
                data["scatter"].append(
                    create_plot(
                        runs,
                        xn,
                        yn,
                        convert_linestyle(linestyles),
                        j2_env,
                        "Scatterplot ",
                        "bubble",
                    )
                )

        # Create a .png plot for the summary page:
        # TODO: Right now, this is an ann-only recall vs time plot.
        data_for_plot = {}
        for k in runs.keys():
            data_for_plot[k] = prepare_data(runs[k], "k-nn", "qps")

        plot.create_plot(
            data_for_plot,
            False,
            "linear",
            "log",
            "k-nn",  # Obsolete choice!
            "qps",
            args.outputdir + name + ".png",
            linestyles,
            batch,
        )

        # Final render of the detailed page:
        output_path = args.outputdir + name + ".html"
        with open(output_path, "w") as text_file:
            text_file.write(
                j2_env.get_template("detail_page.html").render(
                    title=label, plot_data=data, args=args, batch=batch
                )
            )


def build_index_site(datasets, algorithms, j2_env, file_name):
    """Builds the front page of our website (index.html)."""

    dataset_data = {"batch": [], "non-batch": []}
    for mode in ["batch", "non-batch"]:
        # Arbitrary sorting order: first by "metric" name...
        # TODO: Obsolete choice?
        distance_measures = sorted(
            set([get_distance_from_desc(e) for e in datasets[mode].keys()])
        )
        # ...then by dataset name:
        sorted_datasets = sorted(
            set([get_dataset_from_desc(e) for e in datasets[mode].keys()])
        )

        for dm in distance_measures:
            d = {"name": dm.capitalize(), "entries": []}
            for ds in sorted_datasets:
                # Extract all experiments with the correct info...
                matching_datasets = [
                    e
                    for e in datasets[mode].keys()
                    if get_dataset_from_desc(e) == ds
                    and get_distance_from_desc(e) == dm  # noqa
                ]
                # ...and sort them by increasing number of "K"-Nearest Neighbors:
                # TODO: obsolete
                sorted_matches = sorted(
                    matching_datasets, key=lambda e: int(get_count_from_desc(e))
                )
                # Add the relevant data to a list in a dict, that will
                # be matched by a Jinja template:
                for idd in sorted_matches:
                    d["entries"].append({"name": idd, "desc": get_dataset_label(idd)})
            dataset_data[mode].append(d)

    with open(args.outputdir + "index.html", "w") as text_file:
        text_file.write(
            j2_env.get_template("summary.html").render(
                title="Kernel-Matrix-Benchmarks",
                dataset_with_distances=dataset_data,
                algorithms=algorithms,
            )
        )


def load_all_results():
    """Read all result files and compute all metrics"""

    all_runs_by_dataset = {"batch": {}, "non-batch": {}}
    all_runs_by_algorithm = {"batch": {}, "non-batch": {}}
    cached_true_dist = []
    old_sdn = None

    # N.B.: We keep experiments "one query at a time" and "all queries at once"
    #       separate, as they address different use cases.
    for mode in ["non-batch", "batch"]:
        for properties, f in results.load_all_results(batch_mode=(mode == "batch")):
            # String identifier for a problem (dataset, kernel, ...):
            sdn = get_run_desc(properties)
            # If this is a new problem, we must recompute some variables:
            if sdn != old_sdn:
                dataset, _ = get_dataset(properties["dataset"])
                # TODO: "distances" is obsolete, ANN-only
                cached_true_dist = list(dataset["distances"])
                old_sdn = sdn

            algo_ds = get_dataset_label(sdn)
            desc_suffix = "-batch" if mode == "batch" else ""
            algo = properties["algo"] + desc_suffix
            sdn += desc_suffix
            ms = compute_all_metrics(cached_true_dist, f, properties, args.recompute)

            # Put the run in a dict sorted by method...
            all_runs_by_algorithm[mode].setdefault(algo, {}).setdefault(
                algo_ds, []
            ).append(ms)
            # ... and in a dict sorted by dataset:
            all_runs_by_dataset[mode].setdefault(sdn, {}).setdefault(algo, []).append(
                ms
            )

    return (all_runs_by_dataset, all_runs_by_algorithm)


# We use the Jinja templating system:
j2_env = Environment(loader=FileSystemLoader("./templates/"), trim_blocks=True)
j2_env.globals.update(zip=zip, len=len)

# Load the data:
runs_by_ds, runs_by_algo = load_all_results()
# Retrieve the names of the datasets - each of whom receives a detailed page:
dataset_names = [
    get_dataset_label(x)
    for x in list(runs_by_ds["batch"].keys()) + list(runs_by_ds["non-batch"].keys())
]
# Retrieve the names of the algorithms - each of whom receives a detailed page:
algorithm_names = list(runs_by_algo["batch"].keys()) + list(
    runs_by_algo["non-batch"].keys()
)

linestyles = {**create_linestyles(dataset_names), **create_linestyles(algorithm_names)}
ds_l = lambda label: get_dataset_label(label)

# Detailed pages for datasets processed "one query at a time":
build_detail_site(runs_by_ds["non-batch"], ds_l, j2_env, linestyles, False)

# Detailed pages for datasets processed "all queries at once":
build_detail_site(runs_by_ds["batch"], ds_l, j2_env, linestyles, True)

# Detailed pages for algorithms run "one query at a time":
build_detail_site(runs_by_algo["non-batch"], lambda x: x, j2_env, linestyles, False)

# Detailed pages for algorithms run "all queries at once":
build_detail_site(runs_by_algo["batch"], lambda x: x, j2_env, linestyles, True)

# Index page:
build_index_site(runs_by_ds, runs_by_algo, j2_env, "index.html")
