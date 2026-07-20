# coding: utf-8

import os
import sys
from setuptools import setup, find_packages, Extension

NAME = "Ale"
VERSION = "1.2.0"

# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

conda_path = os.environ.get("CONDA_PREFIX", None)
if not conda_path:
    raise Exception("A conda environment is expected to build this package.")

ale_c_module = Extension(
    name='ale/_ale_c',
    sources=['ale/ale_c.i', 'src/States.cpp', 'src/Orientations.cpp', 'src/InterpUtils.cpp', 'src/Rotation.cpp', 'src/Vectors.cpp'],
    depends=['include/ale/States.h', 'include/ale/Orientations.h', 'include/ale/InterpUtils.h', 'include/ale/Rotation.h', 'include/ale/Vectors.h'],
    language="c++",
    swig_opts=['-c++'],
    extra_compile_args=["-std=c++17"],
    include_dirs=[os.path.join(conda_path, "include/eigen3"), "include/"]
)

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
            "isd_generate=ale.isd_generate:main",
            "isd_to_kernel=ale.isd_to_kernel:main"
        ],
    },
    ext_modules=[ale_c_module],
)
