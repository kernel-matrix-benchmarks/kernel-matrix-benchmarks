#!/bin/sh

# Creates the full kernel-matrix-benchmarks website from scratch.

# Product on the sphere:
python3 run.py --local --dataset product-sphere-D3-E1-M1000-N1000-inverse-distance
python3 run.py --local --dataset product-sphere-D3-E1-M2000-N2000-inverse-distance
python3 run.py --local --dataset product-sphere-D3-E1-M5000-N5000-inverse-distance
python3 run.py --local --dataset product-sphere-D3-E1-M10000-N10000-inverse-distance

# Solver on the sphere:
python3 run.py --local --dataset solver-sphere-D3-E1-M1000-N1000-inverse-distance
python3 run.py --local --dataset solver-sphere-D3-E1-M2000-N2000-inverse-distance
python3 run.py --local --dataset solver-sphere-D3-E1-M5000-N5000-inverse-distance
python3 run.py --local --dataset solver-sphere-D3-E1-M10000-N10000-inverse-distance

# Create the website and compress it in a zip file:
python3 create_website.py --latex --scatter --outputdir website