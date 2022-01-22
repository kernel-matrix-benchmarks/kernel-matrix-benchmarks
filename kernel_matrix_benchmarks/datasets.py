"""Code to create the input tables and ground truth results for our benchmarks.

All datasets are hosted as HDF5 files at:
"http://kernel-matrix-benchmarks.com/datasets/my-dataset-name.hdf5".

A dataset file "f" contains the following attributes and tables:

- f["source_points"] = (M,D) float64 array ("y").
    The positions of the M source points y_j in dimension D.

- f["target_points"] = (N,D) float64 array ("x").
    The positions of the N target points x_i in dimension D.

- f["source_signal"] = (M,E) float64 array ("b").
    The M signal vectors b_j associated to each point y_j.

- f["target_signal"] = (N,E) float64 array ("a").
    The N signal vectors a_i associated to each point x_i.

- f.attrs["point_type"] = "float"
    For now, we only support real-valued vectors - but permutations and other
    discrete objects may be supported in the future.

- f.attrs["kernel"] = "absolute-exponential" | "gaussian" | ...
    A string identifier for the kernel function.
    We assume that the data points are scaled so that we can use
    the most simple formula for the kernel k(x, y), without any scaling constant.

    For instance, "absolute-exponential" refers to the kernel formula:
        k(x, y) = exp(-|x - y|_2)
    whereas "gaussian" refers to:
        k(x, y) = exp(-|x - y|^2_2).

- f.attrs["normalize_rows"] = True | [False]
    If True, we normalize the rows of the kernel matrix so that they sum up to 1.
    This is especially relevant for attention layers in transformer architectures,
    which rely on a row-normalized exponential kernel.

- f.attrs["same_points"] = True | [False]
    If True, we assume that the array f["target_points"] is equal to f["source_points"],
    i.e. M = N and x_i = y_i for all i in [1, N].

- f.attrs["density_estimation"] = True | [False]
    If True, we assume that f["source_signal"] is uniformly equal to 1:
    the kernel product operation becomes a simple sum over the rows
    of the kernel matrix.


The four data arrays should be in correspondance with each other 
up to float64 numerical precision, i.e.

a[i] = sum_{j in range(M)} k(x[i], y[j]) * b[j]
if  f.attrs["normalize_rows"] == False (default)

or

a[i] = sum_{j in range(M)} k(x[i], y[j]) * b[j] /  sum_{j in range(M)} k(x[i], y[j])
if  f.attrs["normalize_rows"] == True (for attention layers).
"""

import h5py
import numpy
import os
import random

from urllib.request import urlopen
from urllib.request import urlretrieve

from kernel_matrix_benchmarks.algorithms.bruteforce import (
    BruteForceProductBLAS as GroundTruth,
)


def download(src, dst):
    """Retrieves an online dataset, typically hosted on kernel-matrix-benchmarks.com."""

    if not os.path.exists(dst):
        # TODO: should be atomic
        print("downloading %s -> %s..." % (src, dst))
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    """Returns the name of the .hdf5 file for a given dataset."""
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", "%s.hdf5" % dataset)


def get_dataset(which):
    """Returns a loaded .hdf5 file and the dimension of the points."""
    hdf5_fn = get_dataset_fn(which)

    # We first try to download the dataset from our website:
    try:
        url = "http://kernel-matrix-benchmarks.com/datasets/%s.hdf5" % which
        download(url, hdf5_fn)

    # If this fails, we try to download it from an "original" repository
    # and process it as required:
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            DATASETS[which](hdf5_fn)
    hdf5_f = h5py.File(hdf5_fn, "r")

    # Cast to integer because the json parser (later on) cannot interpret numpy integers.
    dimension = int(hdf5_f["source_points"].shape[-1])

    return hdf5_f, dimension


# Everything below this line is related to creating datasets ===================
# You probably never need to do this at home,
# just rely on the prepared datasets at http://kernel-matrix-benchmarks.com


def write_output(
    filename,
    kernel,
    source_points,
    target_points=None,
    source_signal=None,
    point_type="float",
    normalize_rows=False,
):
    """Compute the ground truth output signal and save to a HDF5 file."""
    from kernel_matrix_benchmarks.algorithms.bruteforce import BruteForceProduct

    # Handles file opening/closure:
    with h5py.File(filename, "w") as f:
        # First attributes: kernel type ("gaussian"...), point_type ("float"),
        # are we normalizing the rows of the kernel matrix (for attention layers).
        f.attrs["kernel"] = kernel
        f.attrs["point_type"] = point_type
        f.attrs["normalize_rows"] = normalize_rows

        # First data array: source points y_1, ..., y_M:
        f["source_points"] = source_points

        # Second data array: target points x_1, ..., x_N:
        if target_points is None:  # Special case: x == y
            f["target_points"] = source_points
            f.attrs["same_points"] = True
        else:  # x != y
            f["target_points"] = target_points
            f.attrs["same_points"] = False

        # Third data array: source signal b_1, ..., b_M:
        if source_signal is None:  # Special case: b == 1
            f["source_signal"] = numpy.ones((len(source_points), 1))
            f.attrs["density_estimation"] = True
        else:  # Usual case:
            f["source_signal"] = source_signal
            f.attrs["density_estimation"] = False

        # Bruteforce computation for the "ground truth" output signal:
        gt = GroundTruth(kernel=kernel, normalize_rows=normalize_rows)
        # N.B.: The [:] syntax is there to make sure that we convert
        # the content of the hdf5 file to a NumPy array:
        gt.fit(f["source_points"][:], f["source_signal"][:])
        gt.batch_query(f["target_points"][:])

        # Fourth data array: target signal a_1, ..., a_N:
        f["target_signal"] = gt.get_batch_results()


# Synthetic test case: uniform sample on a sphere ------------------------------


def uniform_sphere(
    n_points=1000, dimension=3, radius=1, kernel="gaussian", normalize_rows=False
):
    def write_to(filename):
        # Set the seed for reproducible results:
        numpy.random.seed(n_points + dimension)

        # Generate the source point cloud as a uniform sample on the sphere
        # (isotropic Gaussian sample followed by a normalization):
        source_points = numpy.random.randn(n_points, dimension)
        norms = numpy.linalg.norm(source_points, 2, -1)
        norms[norms == 0] = 1
        source_points = source_points / norms.reshape(n_points, 1)
        # Rescale the point cloud by the desired radius:
        source_points = radius * source_points

        # Generate source signal:
        source_signal = numpy.random.randn(n_points, 1)

        # Compute the ground truth output signal and save to file:
        write_output(
            filename,
            kernel,
            source_points,
            target_points=None,  # == source_points
            source_signal=source_signal,
            point_type="float",
            normalize_rows=normalize_rows,
        )

    return write_to


# TODO: GloVE and MNIST should be updated
# GloVE 25, 50, 100 and 200 ----------------------------------------------------


def train_test_split(X, test_size=10000, dimension=None):
    import sklearn.model_selection

    if dimension == None:
        dimension = X.shape[1]
    print("Splitting %d*%d into train/test" % (X.shape[0], dimension))
    return sklearn.model_selection.train_test_split(
        X, test_size=test_size, random_state=1
    )


def glove(out_fn, d):
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X_train, X_test = train_test_split(X)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")


# MNIST and Fashion-MNIST ------------------------------------------------------


def _load_mnist_vectors(fn):
    import gzip
    import struct

    print("parsing vectors in %s..." % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d"),
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0] for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append(
            [struct.unpack(format_string, f.read(b))[0] for j in range(entry_size)]
        )
    return numpy.array(vectors)


def mnist(out_fn):
    download(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "mnist-train.gz"
    )  # noqa
    download(
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "mnist-test.gz"
    )  # noqa
    train = _load_mnist_vectors("mnist-train.gz")
    test = _load_mnist_vectors("mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


def fashion_mnist(out_fn):
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-train.gz",
    )
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-test.gz",
    )
    train = _load_mnist_vectors("fashion-mnist-train.gz")
    test = _load_mnist_vectors("fashion-mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


# Full list of supported datasets ----------------------------------------------

DATASETS = {
    "uniform-sphere-1k-3-inverse-distance": uniform_sphere(
        n_points=1000, dimension=3, radius=1, kernel="inverse-distance"
    ),
    # "mnist-784-euclidean": mnist,
    # "fashion-mnist-784-euclidean": fashion_mnist,
    # "glove-25-angular": lambda out_fn: glove(out_fn, 25),
    # "glove-50-angular": lambda out_fn: glove(out_fn, 50),
    # "glove-100-angular": lambda out_fn: glove(out_fn, 100),
    # "glove-200-angular": lambda out_fn: glove(out_fn, 200),
}
