import os
import pathlib
from shutil import copyfile

from . import drivers
from . import formatters
from .drivers import load, loads

home_path = os.path.expanduser('~')
config_dir = pathlib.Path(os.path.join(home_path, '.ale'))
config_file_path = pathlib.Path(os.path.join(home_path, '.ale', 'config.yml'))

config_dir.mkdir(parents=True, exist_ok=True) 

if not config_file_path.isfile():
  copyfile('config.yml', config_file_path)

