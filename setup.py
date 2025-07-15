# coding: utf-8

import sys
from setuptools import setup, find_packages

NAME = "Ale"
VERSION = "1.0.0"

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
    author="USGS ASC Development Team",
    author_email="",
    url="",
    keywords=[""],
    packages=find_packages(),
    long_description="""\
    An Abstraction library for reading, writing and computing ephemeris data
    """,
    package_data={'': ['config.yml']},
    entry_points={
        "console_scripts": [
            "isd_generate=ale.isd_generate:main"
        ],
    },
)
