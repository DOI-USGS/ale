===============
ALE Quick Start
===============

This document provides a set of steps to get setup for generating Image Support
Data (ISD) for an image.

Installation
============

The easiest way to setup ALE is using Anaconda. Once you have
`Anaconda <https://www.anaconda.com/products/individual>`_ or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ installed install
ALE from conda-forge by running

.. code-block::

  conda install -c conda-forge ale

.. note::
  It is highly recommended that you use
  `environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
  to manage the packages you install with Anaconda.

Data
====

Planetary imagery is not archived with sufficient data to generate an ISD
from only the image and its label. ALE currently supports two supplementary data
sources: ISIS cubes with attached SPICE, and NAIF SPICE Kernels.


If you are working with ISIS cubes that have attached SPICE (the
`spiceinit <https://isis.astrogeology.usgs.gov/Application/presentation/Tabbed/spiceinit/spiceinit.html>`_
application has been run on them) then ALE will use the data embedded in the
cube file.


If you are working with PDS3 images or ISIS cubes that do not have attached
SPICE, then you will need to download the required NAIF SPICE Kernels for your
image. It is recommended that you use the metakernels provided in the
`PDS kernel archives <https://naif.jpl.nasa.gov/naif/data_archived.html>`_.
You can specify the path for ALE to search for metakernels via the
``ALESPICEROOT`` environment variable. This should be set to the directory where
you have the PDS kernel archives downloaded. An example structure would be

* $ALESPICEROOT

  * mro-m-spice-6-v1.0
  * dawn-m_a-spice-6-v1.0
  * mess-e_v_h-spice-6-v1.0

See :py:attr:`ale.base.data_naif.NaifSpice.kernels` for more information about how to
specify NAIF SPICE kernels.

Load/Loads
==========

The :py:meth:`ale.drivers.load` and :py:meth:`ale.drivers.loads` functions are
the main interface for generating ISDs. Simply pass them the path to your image
file/label and they will attempt to generate an ISD for it.

.. code-block:: python

  import ale

  image_label_path = "/path/to/my/image.lbl"
  isd_string = ale.loads(image_label_path)
