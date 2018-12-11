# coding: utf-8

import sys
from setuptools import setup, find_packages

NAME = "minipf"
VERSION = "0.0.1"

# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

setup(
    name=NAME,
    version=VERSION,
    description="Barebones version of pfeffernusse",
    author_email="jlaura@usgs.gov",
    url="",
    keywords=["Pfeffernusse"],
    packages=find_packages(),
    long_description="""\
    A SpiceAPI for extracting NAIF Spice Data
    """
)
