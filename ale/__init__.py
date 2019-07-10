import os
import pathlib
from shutil import copyfile
import yaml
from pkg_resources import get_distribution, DistributionNotFound

class DotDict(dict):
    """dot.notation access to dictionary attributes"""""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __str__(self):
        return yaml.dump(dict(self))

home_path = os.path.expanduser('~')
config_dir = pathlib.Path(os.path.join(home_path, '.ale'))
config_file_path = pathlib.Path(os.path.join(home_path, '.ale', 'config.yml'))

config_dir.mkdir(parents=True, exist_ok=True)

if not config_file_path.is_file():
  copyfile(os.path.join(os.path.dirname(__file__), 'config.yml'), config_file_path)

config = DotDict(yaml.load(open(config_file_path), Loader=yaml.FullLoader))

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

# bring ale stuff into main ale module
from . import drivers
from . import formatters
from . drivers import load, loads
