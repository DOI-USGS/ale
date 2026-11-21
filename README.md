<p align="center">
  <img src="docs/ALE_Logo.svg" alt="ALE" width=200> 
</p>

# Abstraction Layer for Ephemerides (ALE)
[![Build Status](https://travis-ci.org/USGS-Astrogeology/ale.svg?branch=master)](https://travis-ci.org/USGS-Astrogeology/ale)
[![Coverage Status](https://coveralls.io/repos/github/USGS-Astrogeology/ale/badge.svg?branch=master)](https://coveralls.io/github/USGS-Astrogeology/ale?branch=master)
[![Docs](https://readthedocs.org/projects/ale/badge/?version=latest)](https://ale.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://img.shields.io/badge/DOI-10.5066/P906D84L-blue)](https://www1.usgs.gov/csas/doi/#/form/doi:10.5066%2FP906D84L)


This library allows for the position, rotation, velocity and rotational velocity tracking of
multiple bodies in space, especially in relation to one another. It makes extensive use of NAIF's
SPICE data for such calculations.


## Quick Links

  - [Generating an ISD with `isd_generate`](https://astrogeology.usgs.gov/docs/getting-started/using-ale/isd-generate/)
      - How to use the `isd_generate` script, and set up NAIF SPICE Data

  - [ALE Quickstart](https://astrogeology.usgs.gov/docs/getting-started/using-ale/isd-generate/)
      - Brief overview of how to install ALE and use `load`/`loads` in python

  - [Tutorial: Generating an ISD, Creating a CSM Model, and Converting Coordinates](https://astrogeology.usgs.gov/docs/getting-started/csm-stack/image-to-ground-tutorial/#tutorial-instantiating-a-csm-camera-model-from-image)
      - A tutorial on using ALE and Knoten in python

  - [ALE Driver Architecture](https://astrogeology.usgs.gov/docs/concepts/ale/ale-driver-architecture/)
      - How ALE and its drivers work


## Prerequisite: Conda

Conda is a prerequisite for ALE.  If you need it, 
[download and install conda through miniforge](https://conda-forge.org/download/).

## Installing ALE
```sh
# Create an environment ("y" to confirm)
conda create -n ale

# Run this to activate your environment whenever you need to use ALE
conda activate ale

# Install ALE from conda (in your current environment)
conda install -c conda-forge ale
```


## Adding the ALESPICEROOT environment variable
If your ale driver uses NAIF SPICE data, you need to [download NAIF SPICE DATA (see ASC software docs)](https://astrogeology.usgs.gov/docs/getting-started/using-ale/isd-generate/#setting-up-naif-data) and set the ALESPICEROOT variable in one of these two ways:

```sh
# from your shell:
export ALESPICEROOT=/path/to/ale/spice

# from inside a conda env:
conda env config vars set ALESPICEROOT=/path/to/ale/spice
```


## Using ALE to generate ISDs

To generate an ISD for an image, use the load(s) function. Pass the path to your image/label file and ALE will attempt to find a suitable driver and return an ISD. You can use load to generate the ISD as a dictionary or loads to generate the ISD as a JSON encoded string.

```
isd_dict = load(path_to_label)
isd_string = loads(path_to_label)
```

You can get more verbose output from load(s) by passing verbose=True. If you are having difficulty generating an ISD enable the verbose flag to view the actual errors encountered in drivers.


## Developing ALE

### Installing ALE with git and conda

Clone ALE from git and create a conda environment with the necessary dependencies.
```sh
git clone --recurse-submodules [paste link from "<> Code" button above]
cd ale
conda env create -n ale -f environment.yml  # "y" to confirm
```

### Conda Environment

Activate the environment whenever you need to use ALE.
```sh
conda activate ale
```
> *You can add `conda activate ale` to the end of your .bashrc or .zshrc if you want the `ale` environment to be active in every new terminal.*


### Building ALE
Keep in mind that you will need to clone the repository with the `--recursive` flag in order to
retrieve the gtest submodule for testing. If you have already cloned without the `--recursive` flag,
running the following command inside of a cloned fork of the repository will retrieve the gtest submodule manually:
```bash
git submodule update --init --recursive
```

After you've set up and activated your conda environment and cloned the necessary repos, you may then build ALE. Inside
of a cloned fork of the repository, follow these steps:

```bash
python setup.py build_ext install 
```

This will create a build directory containing the built `ale` python package under something like `build/lib.macosx-11.0-arm64-cpython-314`. 
The full folder name will likely look different depending on what system the package was built by. You can check that everything 
is installed under `$CONDA_PREFIX/lib/python3.14/site-packages/ale/`, you're python version may differ.

Now we can build and install the C++ library. If you are just working on the python side of ALE you can skip the next 
steps:

```bash
cd build
cmake .. -DALE_BUILD_DOCS=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make install
```

The C++ library should be installed under `$CONDA_PREFIX/lib/` as `libale`.

## Adding ALE as a dependency

You can add ALE as a dependency of your CMake based C++ project by linking the exported CMake target, `ale::ale`

```c
add_library(my_library some_source.cpp)
find_package(ale REQUIRED)
target_link_libraries(my_library ale::ale)
```


## Running Tests

To test the c++ part of ALE, run `ctest` from the build directory. 

To test the python part of ALE, run `pytest tests/pytests` from the root of the repository.
