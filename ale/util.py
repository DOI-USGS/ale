import os
from os import path

from glob import glob
from itertools import filterfalse, groupby
import warnings

import pvl

import collections
from collections import OrderedDict
from itertools import chain

import subprocess
import re
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

import spiceypy as spice

from ale import spice_root

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
        mission = os.path.basename(md).split('-')[0].split('_')[0]
        if missions and all([m not in mission.lower() for m in missions]):
            continue

        metakernel_keys = ['mission', 'year', 'version', 'path']

        # recursive glob to make metakernel search more robust to subtle directory structure differences
        metakernel_paths = sorted(glob(os.path.join(md, '**','*.tm'), recursive=True))

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
    """
    Parses a cube label to obtain the kernels from the Kernels group.

    Parameters
    ----------
    cube : cube
        Path to the cube to pull the kernels from.
    expand : bool, optional
        Whether or not to expand variables within kernel paths based on your IsisPreferences file.
    format_as : str, optional {'list', 'dict'}
        How to return the kernels: either as a one-demensional ordered list, or as a dictionary
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
        if expand:
            isisprefs = get_isis_preferences()
            for kern_list in mk_paths:
                for index, kern in enumerate(mk_paths[kern_list]):
                    if kern is not None:
                        mk_paths[kern_list][index] = expandvars(expandvars(kern, dict_to_lower(isisprefs['DataDirectory'])))
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

def get_ck_frames(kernel):
    """
    Get all of the reference frames defined in a kernel.

    Parameters
    ----------
    kernel : str
             The path to the kernel

    Returns
    -------
    ids : list
          The set of reference frames IDs defined in the kernel
    """
    ckbrief = subprocess.run(["ckbrief", "-t {}".format(kernel)],
                             capture_output=True,
                             check=True,
                             text=True)
    ids = set()
    for id in re.findall(r'^(-?[0-9]+)', ckbrief.stdout, flags=re.MULTILINE):
        ids.add(int(id))
    # Sort the output list for testability
    return sorted(list(ids))

def create_spk_dependency_tree(kernels):
    """
    construct the dependency tree for the body states in a set of kernels.

    Parameters
    ----------
    kernels : list
              The list of kernels to evaluate the dependencies in. If two
              kernels in this list contain the same information for the same
              pair of bodies, then the later kernel in the list will be
              identified in the kernel property for that edge in dep_tree.

    Returns
    -------
    dep_tree : nx.DiGraph
               The dependency tree for the kernels. Nodes are bodies. There is
               an edge from one node to another if the state of the body of the
               source node is defined relative to the state of the body of the
               destination node. The kernel edge property identifies what kernel
               the information for the edge is defined in.
    """
    dep_tree = nx.DiGraph()

    for kernel in kernels:
        brief = subprocess.run(["brief", "-c {}".format(kernel)],
                               capture_output=True,
                               check=True,
                               text=True)
        for body, rel_body in re.findall(r'\((.*)\).*w\.r\.t\..*\((.*)\)', brief.stdout):
            dep_tree.add_edge(int(body), int(rel_body), kernel=kernel)

    return dep_tree

def spkmerge_config_string(dep_tree, output_spk, bodies, lsk, start, stop):
    """
    Create the contents of an spkmerge config file that will produce a spk that
    completely defines the state of a list of bodies for a time range.

    Parameters
    ----------
    dep_tree : nx.DiGraph
               Dependency tree from create_kernel_dependency_tree that contains
               information about what the state of different bodies are relative
               to and where that information is stored.
    output_spk : str
                 The path to the SPK that will be output by spkmerge
    bodies : list
             The list of body ID codes that need to be defined in the kernel
             created by spkmerge
    lsk : str
          The absolute path to the leap second kernel to use
    start : str
            The UTC start time for the kernel created by spkmerge
    stop : str
           The UTC stop time for the kernel created by spkmerge

    Returns
    -------
     : str
       The contents of an spkmerge config file that will produce a kernel that
       defines the state of the input bodies for the input time range.
    """
    input_kernels = set()
    all_bodies = set(bodies)
    for body in bodies:
        # Everything is ultimately defined relative to
        # SOLAR SYSTEM BARYCENTER (0) so find the path to it
        dep_path = shortest_path(dep_tree, body, 0)
        all_bodies.update(dep_path)
        for i in range(len(dep_path) - 1):
            input_kernels.add(dep_tree[dep_path[i]][dep_path[i+1]]['kernel'])
    config =  f"LEAPSECONDS_KERNEL     = {lsk}\n"
    config += f"SPK_KERNEL             = {output_spk}\n"
    config += f"   BODIES              = {', '.join([str(b) for b in all_bodies])}\n"
    config += f"   BEGIN_TIME          = {start}\n"
    config += f"   END_TIME            = {stop}\n"
    for kernel in input_kernels:
        config += f"   SOURCE_SPK_KERNEL   = {kernel}\n"
        config += f"      INCLUDE_COMMENTS = no\n"
    return config

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



def duckpool(naifvar, start=0, length=10, default=None):
    """
    Duck typing friendly version of spiceypy kernel pool functions.

    Parameters
    ----------
    naifvar : str
              naif var string to query pool for

    start : int
            Index of first value

    length : int
             max number of values returned

    default : obj
              Default value to return if key is not found in kernel pool

    Returns
    -------
    : obj
      Spice value returned from spiceypy if found, default value otherwise

    """
    for f in [spice.gdpool, spice.gcpool, spice.gipool]:
        try:
            val = f(naifvar, start, length)
            return val[0] if  len(val) == 1 else val
        except:
            continue
    return default


def query_kernel_pool(matchstr="*"):
    """
    Collect multiple keywords from the naif kernel pool based on a
    template string

    Parameters
    ----------
    matchstr : str
               matchi_c formatted str

    Returns
    -------
    : dict
      python dictionary of naif keywords in {keyword:value} format.
    """

    try:
        svars = spice.gnpool(matchstr, 0, 100)
    except Exception as e:
        warnings.warn(f"kernel search for {matchstr} failed with {e}")
        svars = []

    svals = [duckpool(v) for v in svars]
    return dict(zip(svars, svals))

