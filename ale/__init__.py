import os
import pathlib
from shutil import copyfile

import yaml

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

config = DotDict(yaml.load(open(config_file_path)))

from . import drivers
from . import formatters
from .drivers import load, loads

