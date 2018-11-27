# SpiceRefactor

This library allows for the position, rotation, velocity and rotational velocity tracking of
multiple bodies in space, especially in relation to one another. It makes extensive use of NAIF's
SPICE data for such calculations. 

## Setting up dependencies with conda (RECOMMENDED)

Install conda if you do not already have it.
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b
```
> You can add a `-p <install-prefix>` to choose where to install miniconda. By default, it will install it to `$HOME/miniconda3`.

### Setting up conda for bash
Copy and paste the following into a terminal running the `bash` shell:
```bash
echo -e "\n\n# Adding miniconda3 to PATH" >> $HOME/.bashrc && \
echo -e "export PATH=$HOME/miniconda3/bin:\$PATH" >> $HOME/.bashrc && \
source $HOME/.bashrc && \
which conda
```
> *For more information: [bash installation](https://conda.io/docs/user-guide/install/linux.html "Reference to bash conda install")*

### Setting up conda for tcsh
Copy and paste the following into a terminal running the `tcsh` shell:
```tcsh
echo  "\n\n# Setting up miniconda3 for tcsh" >> $HOME/.cshrc && \
echo  "source $HOME/miniconda3/etc/profile.d/conda.csh > /dev/null" >> $HOME/.cshrc && \
source $HOME/.cshrc && \
which conda
```
> *For more information: [tcsh installation](https://github.com/ESMValGroup/ESMValTool/issues/301 "Reference to tcsh conda install")*

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
After you've set up conda, you can build SpiceRefactor:

1. Fork `USGS-Astrogeology/SpiceRefactor` if you don't already have a fork.
2. Clone your fork of `SpiceRefactor` *with `--recursive` option to get the gtest submodule*.
```bash
git clone --recursive git@github.com:<your-username>/SpiceRefactor.git
cd SpiceRefactor
git remote add upstream git@github.com:USGS-Astrogeology/SpiceRefactor.git
```
3. Sync your fork with `upstream` and ensure the gtest submodule is init'd if your fork is old.
```bash
git pull upstream master
git submodule update --init --recursive
git push -u origin master
```
4. `mkdir build && cd build`
5. `cmake .. && make` (TODO This command will need to be updated)
6. `ctest`

---

## Building without a package manager
TODO Do we still need this section at all...?
