import os
import warnings
warnings.filterwarnings("ignore")

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

try:
    spice_root = os.environ['ALESPICEROOT']
except:
    spice_root = None

# bring ale stuff into main ale module
from . import drivers
from . import formatters
from . drivers import load, loads
