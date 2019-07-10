# coding: utf-8

import sys
from setuptools import setup, find_packages

NAME = "Ale"
VERSION = "0.2.0"

# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

setup(
    name=NAME,
    version=VERSION,
    description="Abstraction Layer for Ephemerides",
    author_email="jlaura@usgs.gov",
    url="",
    keywords=[""],
    packages=find_packages(),
    long_description="""\
    An Abstraction library for reading, writing and computing ephemeris data
    """
)
