# Benchmarking kernel matrix vector products and inversions

[![Build Status](https://img.shields.io/github/workflow/status/kernel-matrix-benchmarks/kernel-matrix-benchmarks/kernel%20matrix%20benchmarks?style=flat-square)](https://github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/actions?query=workflow:benchmarks)

Computations with kernel matrices are a key bottleneck in many applied fields -- from numerical physics to machine learning.
This website compares acceleration methods for these problems in an objective way.

Specifically, we are interested in...

This project contains tools to benchmark various implementations of these operations in a wide range of settings:

- We use both standard **CPU** hardware and massively parallel **GPU** accelerators.
- We work in spaces of varying dimension (1 to 1,000) and geometry (Euclidean, curved, discrete, etc.).
- We study a wide range of kernels that may be oscillating, exhibit singularities, etc.
- We benchmark both **exact** and **approximate** methods.

Our main purpose is to establish a clear reference on the state-of-the-art for kernel computations.
We hope that this work will promote cross-pollination between communities.

Please note that this ongoing benchmark is **open to all contributions**.
We have pre-generated data sets with relevant evaluation metrics and provide a Docker container for each algorithm. We also rely on a [test suite](https://travis-ci.org/kernel-matrix-benchmarks/kernel-matrix-benchmarks) to make sure that every algorithm works.

## Evaluated implementations and methods

- [KeOps](https://www.kernel-operations.io): on-the-fly bruteforce computations on CPU and GPU.

## Data sets

We provide a varied collection of test cases for the methods above.
All data sets are pre-split into train/test and come with ground truth data. We store the inputs and expected output in HDF5 format:

| Dataset                                                           | Dimensions | Train size | Test size | Kernel | Distance  | Download                                                                   |
| ----------------------------------------------------------------- | ---------: | ---------: | --------: | ----------: | --------- | -------------------------------------------------------------------------- |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                        |        784 |     60,000 |    10,000 |    Gaussian | Euclidean | [HDF5](http://ann-benchmarks.com/mnist-784-euclidean.hdf5) (217MB)         |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |        784 |     60,000 |    10,000 |    Gaussian | Euclidean | [HDF5](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5) (217MB) |
| [GloVe](http://nlp.stanford.edu/projects/glove/)                  |         25 |  1,183,514 |    10,000 | Exponential | Angular   | [HDF5](http://ann-benchmarks.com/glove-25-angular.hdf5) (121MB)            |
| GloVe                                                             |         50 |  1,183,514 |    10,000 | Exponential | Angular   | [HDF5](http://ann-benchmarks.com/glove-50-angular.hdf5) (235MB)            |
| GloVe                                                             |        100 |  1,183,514 |    10,000 | Exponential | Angular   | [HDF5](http://ann-benchmarks.com/glove-100-angular.hdf5) (463MB)           |
| GloVe                                                             |        200 |  1,183,514 |    10,000 | Exponential | Angular   | [HDF5](http://ann-benchmarks.com/glove-200-angular.hdf5) (918MB)           |

## Results

Interactive plots can be found at <http://kernel-matrix-benchmarks.com>. These are all as of December 2021, running all benchmarks on a r5.4xlarge machine on AWS with `--parallelism 7`:

### fashion-mnist-784-euclidean

![fashion-mnist-784-euclidean](https://raw.github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/master/results/fashion-mnist-784-euclidean.png)

### glove-25-angular

![glove-25-angular](https://raw.github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/master/results/glove-25-angular.png)

### glove-100-angular

![glove-100-angular](https://raw.github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/master/results/glove-100-angular.png)

## Install

The only dependencies are Python (tested with 3.6) and Docker.
To install them on a new AWS instance:

1. Update the list of packages with `sudo apt update`.
2. Install the Python package manager with `sudo apt install python3-pip`.
3. Install Docker with `sudo apt install docker.io`.
4. Add the current user to the Docker group. Assuming that you are "ubuntu", this is done with `sudo usermod -a -G docker ubuntu`.
5. Refresh the Docker group with `newgrp docker`.

Then:

1. Clone the repo (`git clone https://github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks.git`).
2. Enter the main directory (`cd kernel-matrix-benchmarks`).
3. Run `pip3 install -r requirements.txt`.
4. Run `python3 install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).

## Running

1. Run `python3 run.py` (this can take an extremely long time, potentially days)
2. Run `python3 plot.py` or `python3 create_website.py` to plot results.

You can customize the algorithms and datasets if you want to:

- Check that `algos.yaml` contains the parameter settings that you want to test
- To run experiments on Glove embeddings in dimension 100, invoke `python run.py --dataset glove-100-angular`. See `python run.py --help` for more information on possible settings. Note that experiments can take a long time.
- To process the results, either use `python plot.py --dataset glove-100-angular` or `python create_website.py`. An example call: `python create_website.py --plottype recall/time --latex --scatter --outputdir website/`.

## Including your algorithm

1. Add your algorithm into `kernel_matrix_benchmarks/algorithms` by providing a small Python wrapper.
2. Add a Dockerfile in `install/` for it
3. Add it to `algos.yaml`
4. Add it to `.github/workflows/benchmarks.yml`

## Principles

Open science:

- Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
- In particular: if you are the author of one of these libraries and you think that the benchmark can be improved, please consider submitting a pull request.
- This is meant to be an ongoing project and represent the current state-of-the-art.
- Make everything easy to replicate, including the installation of the libraries and the data pre-processing.

Diverse use cases:

- Showcase challenging datasets from all fields that rely on kernel computations.
- Benchmark both pre-computation and query times.
- Support both CPU and GPU implementations.
- Try many different values of parameters for each library -- and ignore the points that are not on the [precision-performance frontier](https://en.wikipedia.org/wiki/Pareto_front).

## Authors

- [Jean Feydy](https://www.jeanfeydy.com), [HeKA team](https://team.inria.fr/heka/), [INRIA Paris](https://www.inria.fr/en/centre-inria-de-paris).

We rely heavily on the template of the [ANN-benchmarks](https://github.com/erikbern/ann-benchmarks) website, built by [Erik Bernhardsson](https://erikbern.com) with significant contributions from [Martin Aumüller](http://itu.dk/people/maau/) and [Alexander Faithfull](https://github.com/ale-f).

## Related Publications

We will document our framework and results with a publication in due time.
Meanwhile, the following paper details design principles behind the ANN-benchmark framework:

- M. Aumüller, E. Bernhardsson, A. Faithfull:
[ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://arxiv.org/abs/1807.05614). Information Systems 2019. DOI: [10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)

## Related Projects

- [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) is the reference website for approximate nearest neighbor search.
- [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) is a benchmarking effort for billion-scale approximate nearest neighbor search as part of the [NeurIPS'21 Competition track](https://neurips.cc/Conferences/2021/CompetitionTrack).
