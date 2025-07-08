import warnings
warnings.filterwarnings("ignore")

import os
import sys

import logging

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python <3.8
    try:
        from importlib_metadata import version, PackageNotFoundError
    except ImportError:
        version = None
        PackageNotFoundError = Exception
try:
    if version is not None:
        __version__ = version("ale")
    else:
        __version__ = 'Unknown version (importlib.metadata not available)'
except PackageNotFoundError:
    __version__ = 'Please install this project with setup.py'

logger = None
if not logger:
    logger = logging.getLogger("ale")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [ALE %(filename)s:%(lineno)d] %(levelname)s: %(message)s ')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    log_level = os.environ.get("ALE_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

logger.debug(f"ALE version: {__version__}")
logger.debug(f"ALE python version: {sys.version.split(' ')[0]}")
logger.debug(f"ALE Log Level {log_level}")

try:
    spice_root = os.environ['ALESPICEROOT']
except:
    spice_root = None

# bring ale stuff into main ale module
from . import drivers
from . import formatters
from . drivers import load, loads
