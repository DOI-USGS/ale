import os
from os import path

from glob import glob
from itertools import filterfalse, groupby

from ale import config

import pvl
from collections import OrderedDict
from itertools import chain

import pysis

def get_metakernels(spice_dir=config.spice_root, missions=set(), years=set(), versions=set()):
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

    mission_dirs = list(filter(path.isdir, glob(path.join(spice_dir, '*'))))

    for md in mission_dirs:
        # Assuming spice root has the same name as the original on NAIF website"
        mission = os.path.basename(md).split('-')[0]
        if missions and mission not in missions:
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
    if not avail:
        avail = {
            'count' : 0,
            'data' : 'ERROR: NONE OF {} ARE VALID MISSIONS'.format(missions)
        }

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

def get_isis_preferences(isis_preferences=None):
    """
    Returns ISIS Preference file as a pvl object
    """
    if not isis_preferences:
        isis_preferences = os.path.join(os.path.expanduser("~"), '.Isis', 'IsisPreferences')

    if not os.path.isfile(isis_preferences):
        isis_preferences = os.path.join(pysis.env.ISIS_ROOT, 'IsisPreferences')
    try:
        with open(isis_preferences) as f:
            preftext = f.read().replace('EndGroup', 'End_Group')
            pvlprefs = pvl.loads(preftext)
    except Exception as e:
        raise Exception(f'Failed to load IsisPreferences file [{isis_preferences}]: {e}')

    return pvlprefs

def generate_kernels_from_cube(cube):
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

    # get kernels as 1-d string list
    kernels = [kernel for kernel in chain.from_iterable(mk_paths.values()) if isinstance(kernel, str)]

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
