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

# dynamically load drivers
__all__ = [os.path.splitext(os.path.basename(d))[0] for d in glob(os.path.join(os.path.dirname(__file__), '*_driver.py'))]
__driver_modules__ = [importlib.import_module('.'+m, package='ale.drivers') for m in __all__]

drivers = dict(chain.from_iterable(inspect.getmembers(dmod, lambda x: inspect.isclass(x) and "_driver" in x.__module__) for dmod in __driver_modules__))

def load(label):
    for name, driver in drivers.items():
        try:
            print("TRYING:", driver)
            res = driver(label)
            if res.is_valid():
                with res as r:
                    return res

        except Exception as e:
            import traceback
            traceback.print_exc()
    raise Exception('No Such Driver for Label')

def loads(label):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

    with load(label) as o:
        s = json.dumps(o.to_dict(), cls=NumpyEncoder, default=json_serial)
    return s
