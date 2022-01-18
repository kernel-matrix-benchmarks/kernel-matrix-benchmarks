# Benchmarking kernel matrix vector products and inversions

[![Build Status](https://img.shields.io/github/workflow/status/kernel-matrix-benchmarks/kernel-matrix-benchmarks/kernel%20matrix%20benchmarks?style=flat-square)](https://github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/actions?query=workflow:benchmarks)

Computations with kernel matrices are a key bottleneck in many applied fields, from numerical physics to machine learning.
This website compares acceleration methods for these problems in an objective way.

Specifically, we are interested in **three main computations**:

**1. Kernel matrix products.** Let us consider:

- A **kernel** function k(x,y) defined for any pair of points in dimension D - for instance,
  a Gaussian kernel.
- N **target** points x<sub>1</sub>, ..., x<sub>N</sub> in dimension D, encoded as a `(N,D)` array.
- M **source** points y<sub>1</sub>, ..., y<sub>M</sub> in dimension D, encoded as a `(M,D)` array.
- M **source** signals v<sub>1</sub>, ..., v<sub>M</sub> in dimension E, encoded as a `(M,E)` array.
  E=1 in most applications.

Then, we compute the `(N,E)` array of **target** signals 
a<sub>1</sub>, ..., a<sub>N</sub> with, for all i between 1 and N:

<p align="center">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+a_i+%5Cgets+%5Csum_%7Bj%3D1%7D%5E%5Ctext%7BM%7D+k%28x_i%2Cy_j%29%5C%2Cv_j+." 
alt="a_i \gets \sum_{j=1}^\text{M} k(x_i,y_j)\,v_j .">
</p>

We understand this computation as the matrix-matrix product between the `(N,M)` 
**kernel matrix** K<sub>i,j</sub> = k(x<sub>i</sub>, y<sub>j</sub>) and
the `(M,E)` matrix of source signals.
Depending on the context, this operation is known as
a kernel **density** estimation, a **N-body** computation
or a point/kernel/spline **convolution**.
Special cases also include the (non-uniform) Discrete **Fourier Transform**
and operators that are relevant to the **Boundary Element Method**.

**2. Kernel matrix solver.**


**3. Attention layers.**

## Scope

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

Full interactive plots can be found at <http://kernel-matrix-benchmarks.com>.
The main performance graphs are summarized below, with one figure per dataset:

### fashion-mnist-784-euclidean

![fashion-mnist-784-euclidean](https://raw.github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/master/results/fashion-mnist-784-euclidean.png)

### glove-25-angular

![glove-25-angular](https://raw.github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/master/results/glove-25-angular.png)

### glove-100-angular

![glove-100-angular](https://raw.github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks/master/results/glove-100-angular.png)

## Run the benchmarks

### Install

The only dependencies are Python (tested with 3.6) and Docker.
To install them on a fresh Ubuntu instance:

1. Update the list of packages with `sudo apt update`.
2. Install the Python package manager with `sudo apt install python3-pip`.
3. Install Docker with `sudo apt install docker.io`.
4. Add the current user to the Docker group. Assuming that you are "ubuntu", this is done with `sudo usermod -a -G docker ubuntu`.
5. Refresh the Docker group with `newgrp docker`.

Then:

1. Clone the repo  with `git clone https://github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks.git`.
2. Enter the main directory with `cd kernel-matrix-benchmarks`.
3. Run `pip3 install -r requirements.txt`.
4. Run `python3 install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).

### Running

1. Run `python3 run.py` (this can take an extremely long time, potentially days).
   Note that with Docker, the root user owns all the output files.
2. Run `sudo python3 plot.py` or `sudo python3 create_website.py` to plot results.

You can customize the algorithms and datasets if you want to:

- Check that [algos.yaml](algos.yaml) contains the parameter settings that you want to test
- To run experiments on Glove embeddings in dimension 100, invoke `python run.py --dataset glove-100-angular`. See `python run.py --help` for more information on possible settings. Note that experiments can take a long time.
- To process the results, either use `python plot.py --dataset glove-100-angular` or `python create_website.py`. An example call: `python create_website.py --plottype recall/time --latex --scatter --outputdir website/`.

### On the Amazon cloud

To reproduce these results in the cloud:

1. Create an account on [AWS EC2](https://aws.amazon.com/aws/ec2).
2. Log in to the [AWS CloudShell](https://console.aws.amazon.com/cloudshell/home?region=us-east-1).
3. Use the "Actions" button in the upper-right corner of the window to upload
  the specification file [kmb-instance.json](kmb-instance.json) in your CloudShell session.
  You may find comments and alternative options in [kmb-instance-full.js](kmb-instance-full.js).
4. Create a new instance (Ubuntu 20.04) with the following AWS CloudShell commands:

```bash
aws ec2 create-security-group \
  --group-name KmbSsh \
  --description "SSH access for Kernel Matrix Benchmarks"
aws ec2 authorize-security-group-ingress \
  --group-name KmbSsh \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0
aws ec2 request-spot-instances \
  --type "persistent" \
  --instance-interruption-behavior "stop" \
  --launch-specification file://kmb-instance.json \
  --tag-specification 'ResourceType=spot-instances-request,Tags=[{Key=Task,Value=KmbCPU}]'
```

5. On startup, the instance will automatically clone this repository
  and run the [create_website_AWS.sh](create_website_AWS.sh) script.
6. Log in to the cloud instance via ssh, with

```bash
ssh -i "kernel-matrix-benchmarks.pem" ubuntu@ec2-1-234-567-890.compute-1.amazonaws.com
```

  Where `kernel-matrix-benchmarks.pem` is your [encryption key](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#KeyPairs:) and `ec2-1-234-567-890` is the id of your instance.

7. You can monitor progress with:

  - `tmux a` to get access to the terminal running [create_website_AWS.sh](create_website_AWS.sh).
  - `less -R kernel-matrix-benchmarks/kmb.log` to read the log file.

8. Once all benchmarks have been run, the full results will be located in `your-instance:/home/ubuntu/kernel-matrix-benchmarks/website.zip`. Download the archive on your local machine with:

```bash
scp -i "kernel-matrix-benchmarks.pem" ubuntu@ec2-1-234-567-890.compute-1.amazonaws.com:/home/ubuntu/kernel-matrix-benchmarks/website.zip website.zip
```

9. Finally, unzip the file `website.zip` and open `website/index.html` to inspect your results.

Please note that the AWS Console allows you to keep track of your
[running instances](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:v=3),
[available storage volumes](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Volumes:),
[current requests for Spot instances](https://console.aws.amazon.com/ec2sp/v2/home?region=us-east-1#/spot)
and [billing information](https://console.aws.amazon.com/billing/home?region=us-east-1#/).
Once you are done with your instance,
**don't forget to cancel your Spot request, terminate your instances and destroy your storage volumes**.


## Including your algorithm

1. Add your algorithm into [kernel_matrix_benchmarks/algorithms](kernel_matrix_benchmarks/algorithms)
   by providing a small Python wrapper.
2. Add a Dockerfile in [install/](install/) for it
3. Add it to [algos.yaml](algos.yaml)


## Main files

- [algos.yaml](algos.yaml): lists all supported methods and parameter values.
- [install.py](install.py): builds the Docker images from [install/](install/).
- [kernel_matrix_benchmarks/](kernel_matrix_benchmarks/):
  - [datasets.py](kernel_matrix_benchmarks/datasets.py): supported datasets.
  - [main.py](kernel_matrix_benchmarks/main.py): runs all supported experiments on a given dataset.
  - [runner.py](kernel_matrix_benchmarks/runner.py): runs a specific experiment and saves results in a HDF5 file.
  - [algorithms/](kernel_matrix_benchmarks/algorithms/):
    - [definitions.py](kernel_matrix_benchmarks/algorithms/definitions.py): parser for [algos.yaml](algos.yaml).
    - [base.py](kernel_matrix_benchmarks/algorithms/base.py): common interface for the methods included in the benchmark.
  - [plotting/](kernel_matrix_benchmarks/plotting/):
    - [metrics.py](kernel_matrix_benchmarks/plotting/metrics.py): supported performance metrics.
    - [plot_variants.py](kernel_matrix_benchmarks/plotting/plot_variants.py): interesting pairs of metrics for the detailed webpages.
    - [utils.py](kernel_matrix_benchmarks/plotting/utils.py): computes the performance metrics and Pareto fronts.
- [plot.py](plot.py): renders png images.
- [create_website.py](create_website.py): renders the website using the
  [Jinja](https://jinja.palletsprojects.com/en/3.0.x/) templates from [templates/](templates/).

## Principles

Open science:

- Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
- In particular: if you are the author of one of these libraries and you think that the benchmark can be improved, please consider submitting a pull request.
- This is meant to be an ongoing project and represent the current state-of-the-art.
- We make everything easy to replicate, including the installation of the libraries and the data pre-processing.

Diverse use cases:

- We showcase challenging datasets from all fields that rely on kernel computations.
- We benchmark both pre-computation and query times.
- We support both CPU and GPU implementations.
- We try many different values of parameters for each library -- and ignore the points that are not on the [precision-performance frontier](https://en.wikipedia.org/wiki/Pareto_front).

## Authors

- [Jean Feydy](https://www.jeanfeydy.com), [HeKA team](https://team.inria.fr/heka/), [INRIA Paris](https://www.inria.fr/en/centre-inria-de-paris).
- ...

We rely heavily on the template of the [ANN-benchmarks](https://github.com/erikbern/ann-benchmarks) website, built by [Erik Bernhardsson](https://erikbern.com) with significant contributions from [Martin Aumüller](http://itu.dk/people/maau/) and [Alexander Faithfull](https://github.com/ale-f).

## Related Publications

We will document our framework and results with a publication in due time.
Meanwhile, the following paper details design principles behind the ANN-benchmark framework:

- M. Aumüller, E. Bernhardsson, A. Faithfull:
[ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://arxiv.org/abs/1807.05614). Information Systems 2019. DOI: [10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)

## Related Projects

- [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) is the reference website for approximate nearest neighbor search.
- [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) is a benchmarking effort for billion-scale approximate nearest neighbor search as part of the [NeurIPS'21 Competition track](https://neurips.cc/Conferences/2021/CompetitionTrack).
