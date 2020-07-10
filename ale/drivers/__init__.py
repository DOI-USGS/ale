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
from ale.formatters.formatter import to_isd
from ale.base.data_isis import IsisSpice

from abc import ABC

# Explicit list of disabled drivers
__disabled_drivers__ = ["ody_drivers",
                        "hayabusa2_drivers",
                        "juno_drivers",
                        "tgo_drivers"]

# dynamically load drivers
__all__ = [os.path.splitext(os.path.basename(d))[0] for d in glob(os.path.join(os.path.dirname(__file__), '*_drivers.py'))]
__all__ = [driver for driver in __all__ if driver not in __disabled_drivers__]
__driver_modules__ = [importlib.import_module('.'+m, package='ale.drivers') for m in __all__]

__formatters__ = {'usgscsm': to_usgscsm,
                  'isis': to_isis,
                  'ale' : to_isd}

def sort_drivers(drivers=[]):
    return list(sorted(drivers, key=lambda x:IsisSpice in x.__bases__, reverse=False))

class AleJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

def load(label, props={}, formatter='ale', verbose=False):
    """
    Attempt to load a given label from all possible drivers.

    This function opens up the label file and attempts to produce an ISD in the
    format specified using the supplied properties. Drivers are tried sequentially
    until an ISD is successfully created. Drivers that use external ephemeris
    data are tested before drivers that use attached epehemeris data.

    Parameters
    ----------
    label : str
            String path to the given label file

    props : dict
            A dictionary of optional keywords/parameters for use in driver
            loading. Each driver specifies its own set of properties to use.
            For example, Drivers that use the NaifSpice mix-in use the 'kernels'
            property to specify an explicit set of kernels and load order.

    formatter : {'ale', 'isis', 'usgscsm'}
                Output format for the ISD. As of 0.8.0, it is recommended that
                the `ale` formatter is used. The `isis` and `usgscsm` formatters
                are retrained for backwards compatability.

    verbose : bool
              If True, displays debug output specifying which drivers were
              attempted and why they failed.

    Returns
    -------
    dict
         The ISD as a dictionary
    """
    if isinstance(formatter, str):
        formatter = __formatters__[formatter]

    drivers = chain.from_iterable(inspect.getmembers(dmod, lambda x: inspect.isclass(x) and "_driver" in x.__module__) for dmod in __driver_modules__)
    drivers = sort_drivers([d[1] for d in drivers])

    for driver in drivers:
        if verbose:
            print(f'Trying {driver}')
        try:
            res = driver(label, props=props)
            # get instrument_id to force early failure
            res.instrument_id

            with res as driver:
                isd = formatter(driver)
                if verbose:
                    print("Success with: ", driver)
                    print("ISD:\n", json.dumps(isd, indent=2, cls=AleJsonEncoder))
                return isd
        except Exception as e:
            if verbose:
                print(f'Failed: {e}\n')
                traceback.print_exc()
    raise Exception('No Such Driver for Label')

def loads(label, props='', formatter='ale', verbose=False):
    """
    Attempt to load a given label from all possible drivers.

    This function is the same as load, except it returns a JSON formatted string.

    Returns
    -------
    str
        The ISD as a JSON formatted string

    See Also
    --------
    load
    """
    res = load(label, props, formatter, verbose=verbose)
    return json.dumps(res, cls=AleJsonEncoder)
