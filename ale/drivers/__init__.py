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
from collections import OrderedDict

from ale.formatters.usgscsm_formatter import to_usgscsm
from ale.formatters.isis_formatter import to_isis
from ale.base.data_isis import IsisSpice

from abc import ABC

# dynamically load drivers
__all__ = [os.path.splitext(os.path.basename(d))[0] for d in glob(os.path.join(os.path.dirname(__file__), '*_drivers.py'))]
__driver_modules__ = [importlib.import_module('.'+m, package='ale.drivers') for m in __all__]

__formatters__ = {'usgscsm': to_usgscsm,
                  'isis': to_isis}

def sort_drivers(drivers=[]):
    return list(sorted(drivers, key=lambda x:IsisSpice in x.__bases__, reverse=True))

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


def load(label, props={}, formatter='usgscsm', verbose=False):
    """
    Attempt to load a given label from all possible drivers

    Parameters
    ----------
    label : str
               String path to the given label file
    """
    if isinstance(formatter, str):
        formatter = __formatters__[formatter]

    if props and isinstance(props, str):
        props = json.loads(props)

    drivers = chain.from_iterable(inspect.getmembers(dmod, lambda x: inspect.isclass(x) and "_driver" in x.__module__) for dmod in __driver_modules__)
    drivers = sort_drivers([d[1] for d in drivers])

    for driver in drivers:
        if verbose:
            print(f'Trying {driver}')
        try:
            res = driver(label, props=props)
            with res as driver:
                return formatter(driver)
        except Exception as e:
            if verbose:
                print(f'Failed: {e}\n')
                traceback.print_exc()
    raise Exception('No Such Driver for Label')


def loads(label, props='', formatter='usgscsm'):
    res = load(label, props, formatter)
    return json.dumps(res, cls=JsonEncoder)
