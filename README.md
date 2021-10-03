<p align="center">
  <img src="docs/ALE_Logo.svg" alt="ALE" width=200> 
</p>

# Abstraction Layer for Ephemerides (ALE)
[![Build Status](https://travis-ci.org/USGS-Astrogeology/ale.svg?branch=master)](https://travis-ci.org/USGS-Astrogeology/ale)
[![Coverage Status](https://coveralls.io/repos/github/USGS-Astrogeology/ale/badge.svg?branch=master)](https://coveralls.io/github/USGS-Astrogeology/ale?branch=master)
[![Docs](https://readthedocs.org/projects/ale/badge/?version=latest)](https://ale.readthedocs.io/en/latest/?badge=latest)



This library allows for the position, rotation, velocity and rotational velocity tracking of
multiple bodies in space, especially in relation to one another. It makes extensive use of NAIF's
SPICE data for such calculations.

## Using ALE to generate ISDs

To generate an ISD for an image, use the load(s) function. Pass the path to your image/label file and ALE will attempt to find a suitable driver and return an ISD. You can use load to generate the ISD as a dictionary or loads to generate the ISD as a JSON encoded string.

```
isd_dict = load(path_to_label)
isd_string = loads(path_to_label)
```

You can get more verbose output from load(s) by passing verbose=True. If you are having difficulty generating an ISD enable the verbose flag to view the actual errors encountered in drivers.

## Setting up dependencies with conda (RECOMMENDED)

Install conda (either [Anaconda](https://www.anaconda.com/download/#linux) or
[Miniconda](https://conda.io/miniconda.html)) if you do not already have it. Installation
instructions may be found [here](https://conda.io/docs/user-guide/install/index.html).

### Creating an isolated conda environment
Run the following commands to create a self-contained dev environment for ALE (type `y` to confirm creation):
```bash
conda env create -n ale -f environment.yml
```
> *For more information: [conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html)*

### Activating the environment
After creating the `ale` environment, we need to activate it. The activation command depends on your shell.
* **tcsh**: `conda activate ale`
> *You can add these to the end of your $HOME/.bashrc or $HOME/.cshrc if you want the `ale` environment to be active in every new terminal.*

## Building ALE
After you've set up and activated your conda environment, you may then build ALE. Inside
of a cloned fork of the repository, follow these steps:

```bash
python setup.py install
cd build
cmake ..
make
```

Keep in mind that you will need to clone the repository with the `--recursive` flag in order to
retrieve the gtest submodule for testing. If you have already cloned without the `--recusive` flag,
running the following command will retrieve the gtest submodule manually:
```bash
git submodule update --init --recursive
```

## Adding ALE as a dependency

You can add ALE as a dependency of your CMake based C++ project by linking the exported CMake target, `ale::ale`.

For example:

```
add_library(my_library some_source.cpp)
find_package(ale REQUIRED)
target_link_libraries(my_library ale::ale)
```

## Running Tests

To test the c++ part of ALE, run:

```
ctest
```
from the build directory. 

To test the python part of ALE, run:

```
pytest tests/pytests
```
