from collections import OrderedDict
from glob import glob
from itertools import groupby, chain
import os
from os import path
from tempfile import NamedTemporaryFile
import re
import warnings
from collections.abc import Iterable

import numpy as np
import pvl
import json
import pyspiceql

from ale import spice_root
from ale import logger
from ale.util import get_isis_preferences
from ale.util import get_isis_mission_translations
from ale.util import read_pvl
from ale.util import search_isis_db
from ale.util import dict_merge
from ale.util import dict_to_lower
from ale.util import expandvars

def get_kernels_from_metakernel(metakernel, new_root=spice_root, old_root='/usgs/cpkgs/isis3/data'):
    """
    Given a metakernel:
    1. Replacing the old root with the new root.
    2. Check if its kernels are visibile at the new root.
    3. load the new metakernel (with updated root) from a temp file.
    
    Parameters
    ----------
    kernel : str
             Path of the metakernel 
             (Presumably, containing default PATH_VALUES that have failed)

    new_root : str
               The new root to use (Defaults to ALESPICEROOT/ISISROOT)

    old_root : str
               The old root to replace. (Defaults to /usgs/cpkgs/isis3/data)
    """

    # For a section of an MK, returns all strings that match (are between quotes)
    def read_section(text_lines, opener, closers, match="'(.*?)'"):
        matches = []
        in_section = False
        for line in text_lines:
            in_section = in_section or re.search(opener, line)
            for closer in closers:
                if re.search(closer, line):
                    in_section = False
            if in_section:
                matches.extend(re.findall(match, line))
        return matches

    # Path Symbols/Metakernel reading
    path_values     = []        # base paths from kernel
    path_symbols    = []

    default_paths   = {}        # dict of base paths
    spiceroot_paths = {}        # dict of spice_root-based paths
    
    # Kernel Lists
    listed_kernels    = []      # kernels with path symbols as listed in metakernel
    default_kernels   = []      # kernels with subbed mk path
    spiceroot_kernels = []      # kernels with subbed spice root path

    missing_default_kernels   = False
    missing_spiceroot_kernels = False
    
    check_spiceroot = True

    if isinstance(new_root, str):
        if new_root.endswith('/'):
            new_root = new_root[:-1]
    else:
        check_spiceroot = False
        logger.debug("new_root is not a string.  Likely, ALESPICEROOT is not set.\nCannot check ALESPICEROOT for corrected metakernel paths.")

    # Check if mk is a file, show its contents
    extension = os.path.splitext(metakernel)[1]
    if extension.lower() != '.tm':
        raise ValueError(f'File {metakernel} is not a metakernel (does not have the .tm file extension).')

    # Read mk into memory
    logger.debug("Reading vaules from MK file")
    with open(metakernel, 'r') as mk:
        mklines = mk.readlines()

    # Read Path Values, Path Symbols, and Kernels to Load
    path_values = read_section(mklines, r'PATH_VALUES\s*=', 
                               [r'PATH_SYMBOLS\s*=', r'KERNELS_TO_LOAD\s*=', r'\\begintext'])

    path_symbols = read_section(mklines, r'PATH_SYMBOLS\s*=', 
                               [r'PATH_VALUES\s*=', r'KERNELS_TO_LOAD\s*=', r'\\begintext'])
    
    listed_kernels = read_section(mklines, r'KERNELS_TO_LOAD\s*=', 
                               [r'PATH_VALUES\s*=', r'PATH_SYMBOLS\s*=', r'\\begintext'])

    logger.debug(f"Path values: {path_values}")
    logger.debug(f"Path symbols: {path_symbols}")
    logger.debug(f"Kernels to Load from metakernel: {listed_kernels}") 

    if len(listed_kernels) == 0:
        raise ValueError(f"No kernels were found listed in this metakernel: {metakernel}")

    # Create Path Dicts
    if(len(path_symbols) != len(path_values)):
        msg = f"The number of PATH_SYMBOLS ({len(path_symbols)})"
        msg = msg + f"and PATH_VALUES ({len(path_values)}) found"
        msg = msg + f"were not the same in metakernel {metakernel}."
        raise ValueError(msg)
    else:
        for index, key in enumerate(path_symbols):
            default_paths[key] = path_values[index]
            if check_spiceroot:
                spiceroot_paths[key] = re.sub(old_root, new_root, path_values[index])

    # Check default mk paths for kernels
    mk_dir = os.path.dirname(os.path.abspath(metakernel))
    for kernel in listed_kernels:
        default_kernel = kernel
        for symbol, path in default_paths.items():
            default_kernel = default_kernel.replace('$' + symbol, path)

        # PATH_VALUES may be relative (e.g. '..' in ESA-style metakernels). SPICE
        # resolves these against the metakernel's own directory, not the current
        # working directory. Anchor relative paths to the metakernel dir so they
        # resolve regardless of where ALE is invoked from.
        if not os.path.isabs(default_kernel):
            default_kernel = os.path.normpath(os.path.join(mk_dir, default_kernel))

        if os.path.isfile(default_kernel):
            default_kernels.append(default_kernel)
        else:
            logger.warning(f"Could not find kernel in paths from metakernal: {default_kernel}.  Looking under ALESPICEROOT if set...")
            missing_default_kernels = True
            break

    # If kernels missing from default, Check spice_root paths for kernels
    if missing_default_kernels and check_spiceroot:

        for kernel in listed_kernels:
            spiceroot_kernel = kernel
            for symbol, path in spiceroot_paths.items():
                spiceroot_kernel = spiceroot_kernel.replace('$' + symbol, path)

            if os.path.isfile(spiceroot_kernel):
                spiceroot_kernels.append(spiceroot_kernel)
            else:
                missing_spiceroot_kernels = True
                raise FileNotFoundError(f"""One or more kernel was missing from the default path in the metakernel,
                                        so ALE looked under the ALESPICEROOT path for kernels, 
                                        but could not find this kernel there: {spiceroot_kernel}""")
    
    # Found kernels under default mk path, return those
    if not missing_default_kernels and len(default_kernels) > 0:
        return default_kernels

    # Found kernels under spiceroot, return those
    elif not missing_spiceroot_kernels and len(spiceroot_kernels) > 0:
        return spiceroot_kernels
    
    # Kernels missing, Error message
    errmsg = f"""One or more kernels from this metakernel ({metakernel}) were not found... 
                {len(default_kernels)} found under paths from metakernel: {path_values}; 
                Missing kernels under metakernel path? {missing_default_kernels}; 
                {len(spiceroot_kernels)} found under ALESPICEROOT path: {new_root}; 
                Missing kernels under ALESPICEROOT path? {missing_spiceroot_kernels}; 
                (If one kernel was missing, ALE did not search for more kernels under the same path.)
                """
    raise FileNotFoundError(errmsg)

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
            # Extract year and version by PATTERN, not by fixed position.
            # Metakernel names vary in how many segments precede year/version:
            #   lro_2013, lro_2013_v01, ch2_v01,
            #   orx_noola_2020_v06             (mission_instrument_year_version),
            #   em16_cassis_v533_20250325_002  (ESA: name_version_date_build).
            # A positional split mis-parses any name with more than
            # mission_year_version segments (e.g. orx_noola_2020_v06 -> year
            # 'noola', version '2020'). Year = a 4- or 8-digit segment (YYYY or
            # YYYYMMDD); version = a v<N> segment; either may be absent ('N/A').
            segments = path.splitext(path.basename(k))[0].split('_')
            # Skip forecast/planning metakernels (predicted, planning, plan, flip):
            # they list predicted kernels and are never the right source for an ISD
            # of archived data. Otherwise such a name (e.g. CH1_PREDICTED_V00 or
            # em16_plan) can tie with, and be chosen over, the observation metakernel.
            if any(s.lower() in ('predicted', 'planning', 'plan', 'flip')
                   for s in segments):
                continue
            year = next((s for s in segments
                         if re.fullmatch(r'\d{4}(\d{4})?', s)), 'N/A')
            version = next((s for s in reversed(segments)
                            if re.fullmatch(r'[vV]\d+', s)), 'N/A')
            mission = segments[0] if segments else 'N/A'
            metakernels.append({'mission': mission, 'year': year,
                                'version': version, 'path': k})

        # naive filter, do we really need anything else?
        if years:
            metakernels = list(filter(lambda x:x['year'] in years or x['year'] == 'N/A', metakernels))
        if versions:
            if versions == 'latest':
                latest = []
                # groupby only groups consecutive equal keys, so sort by year first
                metakernels = sorted(metakernels, key=lambda x: x['year'])
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
    # just work with full path
    cube = os.path.abspath(cube)
    cubelabel = None
    try:
        cubelabel = pvl.load(cube)
    except:
        cubelabel = None
    
    if (cubelabel == None):
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            geodata = gdal.Open(cube)
            cubelabel = json.loads(geodata.GetMetadata("json:ISIS3")[0])
        except Exception as e:
            cubelabel = None
    
    if (cubelabel == None):
        raise RuntimeError(f"Could not parse {cube} for pvl or json label")

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
         'SpacecraftClock', 'Extra', 'Clock', 'ShapeModel'])

    if isinstance(kernel_group, str):
        kernel_group = pvl.loads(kernel_group)

    kernel_group = kernel_group["Kernels"]

    def read_kernels(key):
        mk_paths[key] = kernel_group.get(key, None)
        if (mk_paths[key] == "Null"):
            mk_paths[key] = None
        if isinstance(mk_paths[key], str) or mk_paths[key] == None:
            mk_paths[key] = [mk_paths[key]]
        while 'Table' in mk_paths[key]: mk_paths[key].remove('Table')
        while 'Nadir' in mk_paths[key]: mk_paths[key].remove('Nadir')

    for key in mk_paths.keys():
        read_kernels(key)

    if (mk_paths['ShapeModel'][0]):
        if (os.path.splitext(mk_paths['ShapeModel'][0])[-1] != "bds"):
            mk_paths['ShapeModel'] = [None]

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
    elif (format_as == 'spiceql'):
        mk_paths.pop("Clock")
        mk_paths.pop("ShapeModel")
        mk_paths["ck"] = [k.replace("$", "") for k in mk_paths.pop("InstrumentPointing") if k]
        mk_paths["spk"] = [k.replace("$", "") for k in mk_paths.pop("InstrumentPosition") if k]
        mk_paths["pck"] = [k.replace("$", "") for k in mk_paths.pop("TargetAttitudeShape") if k]
        mk_paths["tspk"] = [k.replace("$", "") for k in mk_paths.pop("TargetPosition") if k]
        mk_paths["fk"] = [k.replace("$", "") for k in mk_paths.pop("Frame") if k]
        mk_paths["ik"] = [k.replace("$", "") for k in mk_paths.pop("Instrument") if k]
        mk_paths["iak"] = [k.replace("$", "") for k in mk_paths.pop("InstrumentAddendum") if k]
        mk_paths["sclk"] = [k.replace("$", "") for k in mk_paths.pop("SpacecraftClock") if k]
        mk_paths["lsk"] = [k.replace("$", "") for k in mk_paths.pop("LeapSecond") if k]
        mk_paths["extra"] = [k.replace("$", "") for k in mk_paths.pop("Extra") if k]
        return dict(mk_paths)
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
