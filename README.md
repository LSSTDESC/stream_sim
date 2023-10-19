# Simple Stellar Stream Simulator

This package provides some tools for simple stellar stream simulation. The goal is to provide a modular codebase that can be easily configured and run. The user should be able to generate a variety of stream morphologies by changing the configuration files. The output will be the "true" properties of stream stars (e.g., location in stream coordinates, heliocentric distance, magnitude, and velocity) that can be used for image-level or catalog-level injection.

## Installation

Installation consists of git cloning `stream_sim`, adding the python module to your `PYTHONPATH`, and adding the `bin` directory to your `PATH`:

```bash
git clone https://github.com/LSSTDESC/stream_sim.git
export PYTHONPATH=${PWD}/stream_sim:${PYTHONPATH}
export PATH=${PWD}/stream_sim/bin:${PATH}
```

The code has some common dependencies (numpy, scipy, pandas, matplotlib) that should be present in your environment.

Eventually stream_sim will be installable through common package managers (i.e., pip and/or conda).

## Examples

Several simple configuration files are provided as examples in the `config` directory. To generate streams from these config files, use the `generate_streams.py` executable found in the `bin` directory. From the `stream_sim` directory, this code can be executed as shown below.

```bash
# Simple linear stream
./bin/generate_stream.py config/toy1_config.yaml -o toy1.csv --plot
# More complex sinusoidal stream
./bin/generate_stream.py config/toy2_config.yaml -o toy2.csv --plot
# Stream generated from Pal 5 interpolation
./bin/generate_stream.py config/pal5_config.yaml -o pal5.csv --plot
# Stream generated from Spline interpolation
./bin/generate_spline_stream.py config/atlas_spline_config.yaml -o atlas_spline.csv --plot
```
