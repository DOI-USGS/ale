from collections import OrderedDict
from glob import glob
from itertools import groupby, chain
import os
from os import path
import re
import warnings

import numpy as np
import pvl

from ale import spice_root
from ale.util import get_isis_preferences
from ale.util import get_isis_mission_translations
from ale.util import read_pvl
from ale.util import search_isis_db
from ale.util import dict_merge
from ale.util import dict_to_lower
from ale.util import expandvars

def get_metakernels(spice_dir=spice_root, missions=set(), years=set(), versions=set()):
    """
    Given a root directory, get any subdirectory containing metakernels,
    assume spice directory structure.

    Mostly doing filtering here, might be worth using Pandas?

    Parameters
    ----------
    spice_dir : str
                Path containing Spice directories downloaded from NAIF's website

    missions : set, str
               Mission or set of missions to search for

    years : set, str, int
            year or set of years to search for

    versions : set, str
               version or set of versions to search for
    """
    if not missions or missions == "all":
        missions = set()
    if not years or years == "all":
        years = set()
    if not versions or versions == "all":
        versions = set()

    if isinstance(missions, str):
        missions = {missions}

    if isinstance(years, str) or isinstance(years, int):
        years = {str(years)}
    else:
        years = {str(year) for year in years}

    avail = {
        'count': 0,
        'data': []
    }

    missions = [m.lower() for m in missions]
    if spice_dir is not None:
        mission_dirs = list(filter(path.isdir, glob(path.join(spice_dir, '*'))))
    else:
        warnings.warn("Unable to search mission directories without" +
                      "ALESPICEROOT being set. Defaulting to empty list")
        mission_dirs = []

    for md in mission_dirs:
        # Assuming spice root has the same name as the original on NAIF website"
        mission = os.path.basename(md).split('-')[0].split('_')[0]
        if missions and all([m not in mission.lower() for m in missions]):
            continue

        metakernel_keys = ['mission', 'year', 'version', 'path']

        # recursive glob to make metakernel search more robust to subtle directory structure differences
        metakernel_paths = sorted(glob(os.path.join(md, '**','*.[Tt][Mm]'), recursive=True))

        metakernels = []
        for k in metakernel_paths:
            components = path.splitext(path.basename(k))[0].split('_') + [k]
            if len(components) == 3:
                components.insert(1, 'N/A')

            metakernels.append(dict(zip(metakernel_keys, components)))

        # naive filter, do we really need anything else?
        if years:
            metakernels = list(filter(lambda x:x['year'] in years or x['year'] == 'N/A', metakernels))
        if versions:
            if versions == 'latest':
                latest = []
                # Panda's groupby is overrated
                for k, g in groupby(metakernels, lambda x:x['year']):
                    items = list(g)
                    latest.append(max(items, key=lambda x:x['version']))
                metakernels = latest
            else:
                metakernels = list(filter(lambda x:x['version'] in versions, metakernels))

        avail['data'].extend(metakernels)

    avail['count'] = len(avail['data'])

    return avail


def generate_kernels_from_cube(cube,  expand=False, format_as='list'):
    """
    Parses a cube label to obtain the kernels from the Kernels group.

    Parameters
    ----------
    cube : cube
        Path to the cube to pull the kernels from.
    expand : bool, optional
        Whether or not to expand variables within kernel paths based on your IsisPreferences file.
        See :func:`get_isis_preferences` for how the IsisPreferences file is found.
    format_as : str, optional {'list', 'dict'}
        How to return the kernels: either as a one-dimensional ordered list, or as a dictionary
        of kernel lists.

    Returns
    -------
    : list
        One-dimensional ordered list of all kernels from the Kernels group in the cube.
    : Dictionary
        Dictionary of lists of kernels with the keys being the Keywords from the Kernels group of
        cube itself, and the values being the values associated with that Keyword in the cube.
    """
    # enforce key order
    mk_paths = OrderedDict.fromkeys(
        ['TargetPosition', 'InstrumentPosition',
         'InstrumentPointing', 'Frame', 'TargetAttitudeShape',
         'Instrument', 'InstrumentAddendum', 'LeapSecond',
         'SpacecraftClock', 'Extra'])

    # just work with full path
    cube = os.path.abspath(cube)
    cubelabel = pvl.load(cube)

    try:
        kernel_group = cubelabel['IsisCube']
    except KeyError:
        raise KeyError(f'{cubelabel}, Could not find kernels group, input cube [{cube}] may not be spiceinited')

    return get_kernels_from_isis_pvl(kernel_group, expand, format_as)


def get_kernels_from_isis_pvl(kernel_group, expand=True, format_as="list"):
    # enforce key order
    mk_paths = OrderedDict.fromkeys(
        ['TargetPosition', 'InstrumentPosition',
         'InstrumentPointing', 'Frame', 'TargetAttitudeShape',
         'Instrument', 'InstrumentAddendum', 'LeapSecond',
         'SpacecraftClock', 'Extra'])


    if isinstance(kernel_group, str):
        kernel_group = pvl.loads(kernel_group)

    kernel_group = kernel_group["Kernels"]

    def load_table_data(key):
        mk_paths[key] = kernel_group.get(key, None)
        if isinstance(mk_paths[key], str):
            mk_paths[key] = [mk_paths[key]]
        while 'Table' in mk_paths[key]: mk_paths[key].remove('Table')
        while 'Nadir' in mk_paths[key]: mk_paths[key].remove('Nadir')

    load_table_data('TargetPosition')
    load_table_data('InstrumentPosition')
    load_table_data('InstrumentPointing')
    load_table_data('TargetAttitudeShape')
    # the rest
    mk_paths['Frame'] = [kernel_group.get('Frame', None)]
    mk_paths['Instrument'] = [kernel_group.get('Instrument', None)]
    mk_paths['InstrumentAddendum'] = [kernel_group.get('InstrumentAddendum', None)]
    mk_paths['SpacecraftClock'] = [kernel_group.get('SpacecraftClock', None)]
    mk_paths['LeapSecond'] = [kernel_group.get('LeapSecond', None)]
    mk_paths['Clock'] = [kernel_group.get('Clock', None)]
    mk_paths['Extra'] = [kernel_group.get('Extra', None)]

    # handles issue with OsirisRex instrument kernels being in a 2d list
    if isinstance(mk_paths['Instrument'][0], list):
        mk_paths['Instrument'] = np.concatenate(mk_paths['Instrument']).flat

    if (format_as == 'list'):
        # get kernels as 1-d string list
        kernels = []
        for kernel in chain.from_iterable(mk_paths.values()):
            if isinstance(kernel, str):
                kernels.append(kernel)
            elif isinstance(kernel, list):
                kernels.extend(kernel)
        if expand:
            isisprefs = get_isis_preferences()
            if not "DataDirectory" in isisprefs:
              warnings.warn("No IsisPreferences file found, is your ISISROOT env var set?")

            kernels = [expandvars(k, isisprefs['DataDirectory'], case_sensitive=False) for k in kernels]
        # Ensure that the ISIS Addendum kernel is last in case it overrides
        # some values from the default Instrument kernel
        # Sorts planetary constants kernel first so it can be overridden by more specific kernels
        kernels = sorted(kernels, key=lambda x: "Addendum" in x)
        kernels = sorted(kernels, key=lambda x: "pck00" in x, reverse=True)
        return kernels
    elif (format_as == 'dict'):
        # return created dict
        if expand:
            isisprefs = get_isis_preferences()
            for kern_list in mk_paths:
                for index, kern in enumerate(mk_paths[kern_list]):
                    if kern is not None:
                        mk_paths[kern_list][index] = expandvars(kern, isisprefs['DataDirectory'], case_sensitive=False)
        return mk_paths
    else:
        raise Exception(f'{format_as} is not a valid return format')


def find_kernels(cube, isis_data, format_as=dict):
    """
    Find all kernels for a cube and return a json object with categorized kernels.

    Parameters
    ----------

    cube : str
           Path to an ISIS cube

    isis_data : str
                path to $ISISDATA

    format_as : obj
                What type to return the kernels as, ISIS3-like dict/PVL or flat list

    Returns
    -------
    : obj
      Container with kernels
    """
    def remove_dups(listofElements):
        # Create an empty list to store unique elements
        uniqueList = []

        # Iterate over the original list and for each element
        # add it to uniqueList, if its not already there.
        for elem in listofElements:
            if elem not in uniqueList:
                uniqueList.append(elem)

        # Return the list of unique elements
        return uniqueList

    cube_label = pvl.load(cube)
    mission_lookup_table = get_isis_mission_translations(isis_data)

    mission_dir = mission_lookup_table[cube_label["IsisCube"]["Instrument"]["SpacecraftName"]]
    mission_dir = path.join(isis_data, mission_dir.lower())

    kernel_dir = path.join(mission_dir, "kernels")
    base_kernel_dir = path.join(isis_data, "base", "kernels")

    kernel_types = [ name for name in os.listdir(kernel_dir) if os.path.isdir(os.path.join(kernel_dir, name)) ]
    kernel_types.extend(name for name in os.listdir(base_kernel_dir) if os.path.isdir(os.path.join(base_kernel_dir, name)))
    kernel_types = set(kernel_types)

    db_files = []
    for typ in kernel_types:
        files = sorted(glob(path.join(kernel_dir, typ, "*.db")))
        base_files = sorted(glob(path.join(base_kernel_dir, typ, "*.db")))
        files = [list(it) for k,it in groupby(files, key=lambda f:os.path.basename(f).split(".")[0])]
        base_files = [list(it) for k,it in groupby(base_files, key=lambda f:os.path.basename(f).split(".")[0])]

        for instrument_dbs in files:
            db_files.append(read_pvl(sorted(instrument_dbs)[-1], True))
        for base_dbs in base_files:
            db_files.append(read_pvl(sorted(base_dbs)[-1], True))


    kernels = {}
    for f in db_files:
        #TODO: Error checking
        typ = f[0][0]
        kernel_search_results = search_isis_db(f[0][1], cube_label, isis_data)

        if not kernel_search_results:
            kernels[typ] = None
        else:
            try:
                kernels[typ]["kernels"].extend(kernel_search_results["kernels"])
                if any(kernel_search_results.get("types", [None])):
                    kernels[typ]["types"].extend(kernel_search_results["types"])
            except:
                kernels[typ] = {}
                kernels[typ]["kernels"] = kernel_search_results["kernels"]
                if any(kernel_search_results.get("types", [None])):
                    kernels[typ]["types"] = kernel_search_results["types"]

    for k,v in kernels.items():
        if v:
            kernels[k]["kernels"] = remove_dups(v["kernels"])

    if format_as == dict:
        return kernels
    elif format_as == list:
        kernel_list = []
        for _,kernels in kernels.items():
            if kernels:
                kernel_list.extend(kernels["kernels"])
        return kernel_list
    else:
        warnings.warn(f"{format_as} is not a valid format, returning as dict")
        return kernels


def write_metakernel_from_cube(cube, mkpath=None):
    # add ISISPREF paths as path_symbols and path_values to avoid custom expand logic
    pvlprefs = get_isis_preferences()

    kernels = generate_kernels_from_cube(cube)

    # make sure kernels are mk strings
    kernels = ["'"+k+"'" for k in kernels]

    paths = OrderedDict(pvlprefs['DataDirectory'])
    path_values = ["'"+os.path.expandvars(path)+"'" for path in paths.values()]
    path_symbols = ["'"+symbol.lower()+"'" for symbol in paths.keys()]

    body = '\n\n'.join([
        'KPL/MK',
        f'Metakernel Generated from an ISIS cube: {cube}',
        '\\begindata',
        'PATH_VALUES = (',
        '\n'.join(path_values),
        ')',
        'PATH_SYMBOLS = (',
        '\n'.join(path_symbols),
        ')',
        'KERNELS_TO_LOAD = (',
        '\n'.join(kernels),
        ')',
        '\\begintext'
    ])

    if mkpath is not None:
        with open(mkpath, 'w') as f:
            f.write(body)

    return body
