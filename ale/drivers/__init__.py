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
from datetime import datetime, date

from abc import ABC

import datetime

# dynamically load drivers
__all__ = [os.path.splitext(os.path.basename(d))[0] for d in glob(os.path.join(os.path.dirname(__file__), '*_driver.py'))]
__driver_modules__ = [importlib.import_module('.'+m, package='ale.drivers') for m in __all__]

drivers = dict(chain.from_iterable(inspect.getmembers(dmod, lambda x: inspect.isclass(x) and "_driver" in x.__module__) for dmod in __driver_modules__))



def load(label):
    """
    Attempt to load a given label from all possible drivers

    Parameters
    ----------
    label : str
               String path to the given label file
    """
    for name, driver in drivers.items():
            print("Trying:", name)
            try:
                res = driver(label)
                if res.is_valid():
                    with res as r:
                            return res.to_dict()
            except Exception as e:
                import traceback
                print("Driver Failed:", e)
                traceback.print_exc()
    raise Exception('No Such Driver for Label')


def loads(label):
    class JsonEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, datetime.datetime):
                return obj.__str__()
            return json.JSONEncoder.default(self, obj)

    res = load(label)
    return json.dumps(res, cls=JsonEncoder)
