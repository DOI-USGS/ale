import os
from os import path

from glob import glob
from itertools import filterfalse, groupby

import pvl

import collections
from collections import OrderedDict
from itertools import chain

from ale import spice_root

import re

def get_metakernels(spice_dir=spice_root, missions=set(), years=set(), versions=set()):
    """
    Given a root directory, get any subdirectory containing metakernels,
    assume spice directory structure.

    Mostly doing filtering here, might be worth using Pandas?

    Parameters
    ----------
    spice_dir : str
                Path containing Spice directories downlaoded from NAIF's website

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
    mission_dirs = list(filter(path.isdir, glob(path.join(spice_dir, '*'))))

    for md in mission_dirs:
        # Assuming spice root has the same name as the original on NAIF website"
        mission = os.path.basename(md).split('-')[0]
        if missions and (mission.lower() not in missions):
            continue

        metakernel_keys = ['mission', 'year', 'version', 'path']

        # recursive glob to make metakernel search more robust to subtle directory structure differences
        metakernel_paths = sorted(glob(os.path.join(md, '**','*.tm'), recursive=True))

        metakernels = [dict(zip(metakernel_keys, [mission]+path.splitext(path.basename(k))[0].split('_')[1:3] + [k])) for k in metakernel_paths]

        # naive filter, do we really need anything else?
        if years:
            metakernels = list(filter(lambda x:x['year'] in years, metakernels))
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


def find_latest_metakernel(path, year):
    metakernel = None
    mks = sorted(glob(os.path.join(path,'*.[Tt][Mm]')))
    if not mks:
        raise Exception(f'No metakernels found in {path}.')
    for mk in mks:
        if str(year) in os.path.basename(mk):
            metakernel = mk
    if not metakernel:
        raise Exception(f'No metakernels found in {path} for {year}.')
    return metakernel



def dict_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

    return dct


def get_isis_preferences(isis_preferences=None):
    """
    Returns ISIS Preference file as a pvl object
    """
    def read_pref(path):
        with open(path) as f:
            preftext = f.read().replace('EndGroup', 'End_Group')
            pvlprefs = pvl.loads(preftext)
        return pvlprefs

    argprefs = {}
    if isis_preferences:
        if isinstance(isis_preferences, dict):
            argprefs = isis_preferences
        else:
            argprefs = read_pref(isis_preferences)

    try:
        homeprefs = read_pref(os.path.join(os.path.expanduser("~"), '.Isis', 'IsisPreferences'))
    except FileNotFoundError as e:
        homeprefs = {}

    try:
        isisrootprefs_path = os.path.join(os.environ["ISISROOT"], 'IsisPreferences')
        isisroot = os.environ['ISISROOT']
        isisrootprefs = read_pref(isisrootprefs_path)
    except (FileNotFoundError, KeyError) as e:
        isisrootprefs = {}

    finalprefs = dict_merge(dict_merge(isisrootprefs, homeprefs), argprefs)

    return finalprefs


def dict_to_lower(d):
    return {k.lower():v if not isinstance(v, dict) else dict_to_lower(v) for k,v in d.items()}


def expandvars(path, env_dict=os.environ, default=None, case_sensative=True):
    user_dict = env_dict if case_sensative else dict_to_lower(env_dict)

    def replace_var(m):
        group0 = m.group(0) if case_sensative else m.group(0).lower()
        group1 = m.group(1) if case_sensative else m.group(1).lower()

        return user_dict.get(m.group(2) or group1, group0 if default is None else default)
    reVar = r'\$(\w+|\{([^}]*)\})'
    return re.sub(reVar, replace_var, path)


def generate_kernels_from_cube(cube,  expand=False, format_as='list'):
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
        kernel_group = cubelabel['IsisCube']['Kernels']
    except KeyError:
        raise KeyError(f'{cubelabel}, Could not find kernels group, input cube [{cube}] may not be spiceinited')

    # Start loading. Stuff that might have multiple entries first
    def load_table_data(key):
        mk_paths[key] = kernel_group.get(key, None)
        if isinstance(mk_paths[key], str):
            mk_paths[key] = [mk_paths[key]]
        while 'Table' in mk_paths[key]: mk_paths[key].remove('Table')

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

    if (format_as == 'list'):
        # get kernels as 1-d string list
        kernels = [kernel for kernel in chain.from_iterable(mk_paths.values()) if isinstance(kernel, str)]
        if expand:
            isisprefs = get_isis_preferences()
            kernels = [expandvars(expandvars(k, dict_to_lower(isisprefs['DataDirectory']))) for k in kernels]
        return kernels   
    elif (format_as == 'dict'):
        # return created dict
        return mk_paths
    else:
        raise Exception(f'{format_as} is not a valid return format')


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


def write_metakernel_from_kernel_list(kernels):
    """
    Parameters
    ----------
    kernels : str
              list of kernel paths

    Returns
    -------
    : str
      Returns string representation of a Naif Metakernel file
    """

    kernels = [os.path.abspath(k) for k in kernels]
    common_prefix = os.path.commonprefix(kernels)

    kernels = ["'"+"$PREFIX"+k[len(common_prefix):]+"'" for k in kernels]
    body = '\n\n'.join([
            'KPL/MK',
            f'Metakernel Generated from a kernel list by Ale',
            '\\begindata',
            'PATH_VALUES = (',
            "'"+common_prefix+"'",
            ')',
            'PATH_SYMBOLS = (',
            "'PREFIX'",
            ')',
            'KERNELS_TO_LOAD = (',
            '\n'.join(kernels),
            ')',
            '\\begintext'
        ])

    return body
