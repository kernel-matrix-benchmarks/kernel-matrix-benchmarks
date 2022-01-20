#!/bin/sh

# Creates the full kernel-matrix-benchmarks website from scratch.
# This code is meant to be used in a fresh AWS instance.

# Install the required dependencies:
sudo apt -y update
sudo apt -y install python3-pip docker.io zip

# Make sure that we can use Docker:
sudo usermod -a -G docker ubuntu
# We have to switch to the docker group,
# and use some heredocs to execute the remainder of the script:
newgrp docker << NEWGRP
# Install the Python requirements:
pip3 install -r requirements.txt
# Also do it for the sudo user:
sudo pip3 install -r requirements.txt

# Install all the docker images:
python3 install.py

# Run our benchmarks on all datasets:
python3 run.py --dataset uniform-sphere-1k-3-absolute-exponential
# python3 run.py --dataset mnist-784-euclidean
# python3 run.py --dataset fashion-mnist-784-euclidean
# python3 run.py --dataset glove-25-angular
# python3 run.py --dataset glove-50-angular
# python3 run.py --dataset glove-100-angular
# python3 run.py --dataset glove-200-angular

# Create the website and compress it in a zip file:
sudo python3 create_website.py --outputdir website
zip -r website.zip website

# Then, just download the website by running
#
# scp -i "kernel-matrix-benchmarks.pem" ubuntu@....amazonaws.com:/home/ubuntu/kernel-matrix-benchmarks/website.zip website.zip
#
# on your local machine, where:
# - "kernel-matrix-benchmarks.pem" is the encryption key to your instance.
# - "....amazonaws.com" is the id of your instance.
NEWGRP