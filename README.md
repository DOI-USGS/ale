# SpiceRefactor

This library allows for the position, rotation, velocity and rotational velocity tracking of
multiple bodies in space, especially in relation to one another. It makes extensive use of NAIF's
SPICE data for such calculations. 

## Setting up dependencies with conda (RECOMMENDED)

Install conda (either [Anaconda](https://www.anaconda.com/download/#linux) or 
[Miniconda](https://conda.io/miniconda.html)) if you do not already have it. Installation
instructions may be found [here](https://conda.io/docs/user-guide/install/index.html).

### Creating an isolated conda environment
(TODO This command will need to be updated) 
Run the following commands to create a self-contained dev environment for SpiceRefactor (type `y` to confirm creation):
```bash
conda create -n spiceRefactor 
```
> *For more information: [conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html)*

### Activating the environment
After creating the `spiceRefactor` environment, we need to activate it. The activation command depends on your shell.
* **bash**: `source activate spiceRefactor`
* **tcsh**: `conda activate spiceRefactor`
> *You can add these to the end of your $HOME/.bashrc or $HOME/.cshrc if you want the `spiceRefactor` environment to be active in every new terminal.*

## Building SpiceRefactor
After you've set up and activated your conda environment, you may then build SpiceRefactor. Inside
of a cloned fork of the repository, follow these steps:

1. TODO

Keep in mind that you will need to clone the repository with the `--recursive` flag in order to
retrieve the gtest submodule for testing. If you have already cloned without the `--recusive` flag,
running the following command will retrieve the gtest submodule manually:
```bash
git submodule update --init --recursive
```
