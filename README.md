# Ephemeris Abstraction Layer (EAL)
[![Build Status](https://travis-ci.org/USGS-Astrogeology/eal.svg?branch=master)](https://travis-ci.org/USGS-Astrogeology/eal)
[![Coverage Status](https://coveralls.io/repos/github/USGS-Astrogeology/eal/badge.svg?branch=master)](https://coveralls.io/github/USGS-Astrogeology/eal?branch=master)

This library allows for the position, rotation, velocity and rotational velocity tracking of
multiple bodies in space, especially in relation to one another. It makes extensive use of NAIF's
SPICE data for such calculations.

## Setting up dependencies with conda (RECOMMENDED)

Install conda (either [Anaconda](https://www.anaconda.com/download/#linux) or
[Miniconda](https://conda.io/miniconda.html)) if you do not already have it. Installation
instructions may be found [here](https://conda.io/docs/user-guide/install/index.html).

### Creating an isolated conda environment
(TODO This command will need to be updated)
Run the following commands to create a self-contained dev environment for EAL (type `y` to confirm creation):
```bash
conda create -n eal
```
> *For more information: [conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html)*

### Activating the environment
After creating the `eal` environment, we need to activate it. The activation command depends on your shell.
* **bash**: `source activate eal`
* **tcsh**: `conda activate eal`
> *You can add these to the end of your $HOME/.bashrc or $HOME/.cshrc if you want the `eal` environment to be active in every new terminal.*

## Building EAL
After you've set up and activated your conda environment, you may then build EAL. Inside
of a cloned fork of the repository, follow these steps:

```bash
mkdir build && cd build
cmake ..
make
```

Keep in mind that you will need to clone the repository with the `--recursive` flag in order to
retrieve the gtest submodule for testing. If you have already cloned without the `--recusive` flag,
running the following command will retrieve the gtest submodule manually:
```bash
git submodule update --init --recursive
```
