import os
import warnings
from pkg_resources import get_distribution, DistributionNotFound


try:
    _dist = get_distribution('ale')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'ale')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version

try:
    spice_root = os.environ['ALESPICEROOT']
except:
    spice_root = None

# bring ale stuff into main ale module
from . import drivers
from . import formatters
from . drivers import load, loads
