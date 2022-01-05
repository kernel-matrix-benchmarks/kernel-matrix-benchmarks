import json
import os
import argparse
import subprocess
from multiprocessing import Pool
from kernel_matrix_benchmarks.main import positive_int


def build(library, args):
    """Builds a library using the relevant Dockerfile in the 'install/' folder."""
    print("Building %s..." % library)
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""

    try:
        subprocess.check_call(
            "docker build %s --rm -t kernel-matrix-benchmarks-%s -f"
            " install/Dockerfile.%s ." % (q, library, library),
            shell=True,
        )
        return {library: "success"}
    except subprocess.CalledProcessError:
        return {library: "fail"}


def build_multiprocess(args):
    return build(*args)


if __name__ == "__main__":

    # This script is normally used with "python install.py"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--proc",
        default=1,
        type=positive_int,
        help="the number of process to build docker images",
    )
    parser.add_argument(
        "--algorithm",
        metavar="NAME",
        help="build only the named algorithm image",
        default=None,
    )
    parser.add_argument(
        "--build-arg", help="pass given args to all docker builds", nargs="+"
    )
    args = parser.parse_args()

    # Step 1: the "master" image is 'install/Dockerfile'
    print("Building base image...")
    subprocess.check_call(
        "docker build \
        --rm -t kernel-matrix-benchmarks -f install/Dockerfile .",
        shell=True,
    )

    # Step 2: identify the Dockerfiles that the user is interested in
    if args.algorithm:
        # Case 1: the user is interested in a single algorithm
        tags = [args.algorithm]
    elif os.getenv("LIBRARY"):
        # Case 2: the user has specified the algorithm with an environment variable
        tags = [os.getenv("LIBRARY")]
    else:
        # Default case: all the Dockerfiles in the 'install/' folder
        tags = [
            fn.split(".")[-1]
            for fn in os.listdir("install")
            if fn.startswith("Dockerfile.")
        ]

    # Step 3: setup all the Docker images, possibly in parallel
    print("Building algorithm images... with (%d) processes" % args.proc)

    if args.proc == 1:
        install_status = [build(tag, args.build_arg) for tag in tags]
    else:
        pool = Pool(processes=args.proc)
        install_status = pool.map(
            build_multiprocess, [(tag, args.build_arg) for tag in tags]
        )
        pool.close()
        pool.join()

    print("\n\nInstall Status:\n" + "\n".join(str(algo) for algo in install_status))
