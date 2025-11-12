# Installation & Dependencies

This guide provides complete instructions for installing `stream_sim` and its dependencies.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/LSSTDESC/stream_sim.git
cd stream_sim

# 2. Set environment variables
export PYTHONPATH=${PWD}:${PYTHONPATH}
export PATH=${PWD}/bin:${PATH}

# 3. Download required data files
python bin/download_data.py
```

## Data Download

`stream_sim` needs external data files (maglim maps, dust map, completeness, photometric errors). Use the downloader and refer to the Data page for details.

```bash
# Download required data (default location: data/)
python bin/download_data.py

# Useful options
python bin/download_data.py --list          # Show what's installed
python bin/download_data.py --force         # Re-download/overwrite
python bin/download_data.py --data-dir DIR  # Custom install location
```

For troubleshooting and data structure, see :doc:`data`.

## Dependences

*To be completed*

Required Python packages:

- ugali
- numpy
- scipy
- pandas
- matplotlib
- astropy
- gala
- healpy
- healsparse
- *(complete list to be added)*


## Optional Dependencies

*To be completed*

## Installing with pip/conda

```{note}
Package installation via pip/conda is planned for future releases.
```
