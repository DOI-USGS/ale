import pvl
import zlib

import importlib
import inspect
import itertools
from itertools import chain
import os
from glob import glob
import json
import numpy as np
import datetime
from datetime import datetime, date
import traceback

from ale.formatters.usgscsm_formatter import to_usgscsm
from ale.formatters.isis_formatter import to_isis

from abc import ABC

# dynamically load drivers
__all__ = [os.path.splitext(os.path.basename(d))[0] for d in glob(os.path.join(os.path.dirname(__file__), '*_drivers.py'))]
__driver_modules__ = [importlib.import_module('.'+m, package='ale.drivers') for m in __all__]

__formatters__ = {'usgscsm': to_usgscsm,
                  'isis': to_isis}

drivers = dict(chain.from_iterable(inspect.getmembers(dmod, lambda x: inspect.isclass(x) and "_driver" in x.__module__) for dmod in __driver_modules__))

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, datetime.datetime):
            return obj.__str__()
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        if isinstance(obj, pvl.PVLModule):
            return pvl.dumps(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.nan):
            return None
        return json.JSONEncoder.default(self, obj)


def load(label, formatter='usgscsm'):
    """
    Attempt to load a given label from all possible drivers

    Parameters
    ----------
    label : str
               String path to the given label file
    """
    if isinstance(formatter, str):
        formatter = __formatters__[formatter]

    for name, driver in drivers.items():
        print(f'Trying {name}')
        try:
            res = driver(label)
            with res as driver:
                return formatter(driver)
        except Exception as e:
            print(f'Failed: {e}\n')
            traceback.print_exc()
    raise Exception('No Such Driver for Label')


def loads(label, formatter='usgscsm'):
    res = load(label, formatter)
    return json.dumps(res, cls=JsonEncoder)
