import pvl

import importlib
import inspect
from itertools import chain, compress
import os
from glob import glob
import json
import numpy as np
import datetime
from datetime import datetime
import traceback

from ale.formatters.usgscsm_formatter import to_usgscsm
from ale.formatters.isis_formatter import to_isis
from ale.formatters.formatter import to_isd
from ale.base.data_isis import IsisSpice
from ale.base.label_isis import IsisLabel
from ale.base.label_pds3 import Pds3Label
from ale.base.data_naif import NaifSpice


from abc import ABC

# Explicit list of disabled drivers
__disabled_drivers__ = ["ody_drivers",
                        "hayabusa2_drivers",
                        "juno_drivers",
                        "tgo_drivers",
                        "msi_drivers"]

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

def load(label, props={}, formatter='ale', verbose=False, only_isis_spice=False, only_naif_spice=False):
    """
    Attempt to load a given label from possible drivers.

    This function opens up the label file and attempts to produce an ISD in the
    format specified using the supplied properties. Drivers are tried sequentially
    until an ISD is successfully created. Drivers that use external ephemeris
    data are tested before drivers that use attached ephemeris data.

    Using the only_* flags will limit the drivers used to construct ISDs. If you
    are not sure what input data you have, just leave the only_* parameters as False.
    Leaving/Setting all only_* parameters to False should satisfy most situations.

    Only parameters explained and there uses:
    * ``only_isis_spice=True`` Used for spiceinit'd ISIS cubes, used, for example, 
    when one has updated the ephemeris information on an ISIS cube.
    * ``only_naif_spice=True`` Used for example, when one has a data product or 
    an ISIS cube, but not yet obtained ephemeris information.

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

    only_isis_spice : bool
                      Explicitly searches for drivers constructed from the IsisSpice
                      component class

    only_naif_spice : bool
                      Explicitly searches for drivers constructed from the NaifSpice
                      component class

    Returns
    -------
    dict
         The ISD as a dictionary
    """
    if isinstance(formatter, str):
        formatter = __formatters__[formatter]
    
    driver_mask = [only_isis_spice, only_naif_spice]
    class_list = [IsisSpice, NaifSpice]
    class_list = list(compress(class_list, driver_mask))
    # predicat logic: make sure x is a class, who contains the word "driver" (clipper_drivers) and 
    # the componenet classes 
    predicat = lambda x: inspect.isclass(x) and "_driver" in x.__module__ and [i for i in class_list if i in inspect.getmro(x)] == class_list
    driver_list = [inspect.getmembers(dmod, predicat) for dmod in __driver_modules__]
    drivers = chain.from_iterable(driver_list)
    drivers = sort_drivers([d[1] for d in drivers])

    if verbose:
        print("Attempting to pre-parse label file")
    try:
        # Try default grammar for pds3 label
        parsed_label = parse_label(label)
    except Exception as e:
        if verbose:
            print("First parse attempt failed with")
            print(e)
        # If pds3 label fails, try isis grammar
        try:
            parsed_label = parse_label(label, pvl.grammar.ISISGrammar())
        except Exception as e:
            if verbose:
                print("Second parse attempt failed with")
                print(e)
            # If both fail, then don't parse the label, and just pass the driver a file.
            parsed_label = None

    if verbose:
        if parsed_label:
            print("Successfully pre-parsed label file")
        else:
            print("Failed to pre-parse label file. Individual drivers will try again.")

    for driver in drivers:
        if verbose:
            print(f'Trying {driver}')
        try:
            res = driver(label, props=props, parsed_label=parsed_label)
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

def loads(label, props='', formatter='ale', indent = 2, verbose=False, only_isis_spice=False, only_naif_spice=False):
    """
    Attempt to load a given label from all possible drivers.

    This function is the same as load, except it returns a JSON formatted string.

    See load for shared parameter documentation.
    
    Parameters
    ----------
    indent : int
             The number of spaces to indent each nested component of the JSON string.
             See json.dumps.

    Returns
    -------
    str
        The ISD as a JSON formatted string

    See Also
    --------
    load
    """
    res = load(label, props, formatter, verbose, only_isis_spice, only_naif_spice)
    return json.dumps(res, indent=indent, cls=AleJsonEncoder)

def parse_label(label, grammar=pvl.grammar.PVLGrammar()):
    """
    Attempt to parse a PVL label.

    Parameters
    ----------
    label
        The label as a pvl string or pvl file.

    grammar
        The pvl grammar with which to parse the label. If None, default to PVLGrammar


    Returns
    -------
    pvl.collections.PVLModule
        The PVL label deserialized to a Python object

    See Also
    --------
    load
    loads
    """
    try:
        parsed_label = pvl.loads(label, grammar=grammar)
    except Exception:
        parsed_label = pvl.load(label, grammar=grammar)
    except:
        raise ValueError("{} is not a valid label for grammar {}".format(label, grammar.__name__))

    return parsed_label
