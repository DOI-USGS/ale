# Abstraction Layer for Ephemerides (ALE)
[![Build Status](https://travis-ci.org/USGS-Astrogeology/ale.svg?branch=master)](https://travis-ci.org/USGS-Astrogeology/ale)
[![Coverage Status](https://coveralls.io/repos/github/USGS-Astrogeology/ale/badge.svg?branch=master)](https://coveralls.io/github/USGS-Astrogeology/ale?branch=master)
[![Docs](https://readthedocs.org/projects/ale/badge/?version=latest)](https://ale.readthedocs.io/en/latest/?badge=latest)



This library allows for the position, rotation, velocity and rotational velocity tracking of
multiple bodies in space, especially in relation to one another. It makes extensive use of NAIF's
SPICE data for such calculations.

## Setting up dependencies with conda (RECOMMENDED)

Install conda (either [Anaconda](https://www.anaconda.com/download/#linux) or
[Miniconda](https://conda.io/miniconda.html)) if you do not already have it. Installation
instructions may be found [here](https://conda.io/docs/user-guide/install/index.html).

### Creating an isolated conda environment
(TODO This command will need to be updated)
Run the following commands to create a self-contained dev environment for ale (type `y` to confirm creation):
```bash
conda env create -n ale -f environment.yml
```
> *For more information: [conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html)*

### Activating the environment
After creating the `ale` environment, we need to activate it. The activation command depends on your shell.
* **bash**: `source activate ale`
* **tcsh**: `conda activate ale`
> *You can add these to the end of your $HOME/.bashrc or $HOME/.cshrc if you want the `ale` environment to be active in every new terminal.*

## Building ALE
After you've set up and activated your conda environment, you may then build ale. Inside
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

## Running Tests

To run ctests to test c++ part of ale, run:

```
ctest
```
from the build directory. 
