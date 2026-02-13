import os
from os import path

from glob import glob
from itertools import chain, filterfalse, groupby
import warnings

import pvl

from collections import OrderedDict
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from datetime import datetime
import pytz

import subprocess
import re
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

from ale import logger

class CachedDict():
    """
    A subclass of dict that tracks the accessed keys.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CachedDict object.

        Parameters
        ----------
        *args : positional arguments
            Variable length argument list.
        **kwargs : keyword arguments
            Arbitrary keyword arguments.
        """
        self.data = dict(**kwargs)
        self.accessed_keys = set()
    

    def __str__(self): 
        """
        to string function 

        Returns
        -------
        str
           str representation of dictionary with filtered items
        """
        return str(self.to_dict)
 

    def __getitem__(self, key):
        """
        Get the value corresponding to the given key.

        Parameters
        ----------
        key
            The key to retrieve the value for.

        Returns
        -------
        value
            The value corresponding to the key.
        """
        self.accessed_keys.add(key)
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        """
        Set the value for the given key.

        Parameters
        ----------
        key
            The key to set the value for.
        value
            The value to be set.
        """
        return self.data.__setitem__(key, value)

    def __delitem__(self, key):
        """
        Delete the value corresponding to the given key.

        Parameters
        ----------
        key
            The key to delete.
        """
        self.accessed_keys.remove(key)
        return self.data.__delitem__(key)

    def keys(self):
        """
        Get the list of keys that have been accessed.

        Returns
        -------
        list
            A list of keys.
        """
        return list(self.accessed_keys)

    def values(self):
        """
        Get the list of values corresponding to the accessed keys.

        Returns
        -------
        list
            A list of values.
        """
        return [self[key] for key in self.accessed_keys if key in self]

    def items(self):
        """
        Get the list of key-value pairs corresponding to the accessed keys.

        Returns
        -------
        list
            A list of key-value pairs.
        """
        return [(key, self.data[key]) for key in self.accessed_keys if key in self]

    def is_key_accessed(self, key):
        """
        Check if a key has been accessed or not.

        Parameters
        ----------
        key
            The key to check.

        Returns
        -------
        bool
            True if the key has been accessed, False otherwise.
        """
        return key in self.accessed_keys


    def to_dict(self): 
        """
        returns a dictionary of only keys accessed
        
        Returns 
        -------
        dict 
            Dictionary of just the keys accessed     
        """
        return dict(zip(self.keys(), self.values()))


def find_latest_metakernel(path, year):
    """
    Find the latest version of a metakernel.

    Parameters
    ----------
    path : str
        The string path of the directory in which to search for metakernels.
    year : str|int
        The year of the desired metakernel.

    Returns
    -------
    str
        The string filename of the latest metakernel.

    Raises
    ------
    Exception
        No metakernels were found in the given path.
    Exception
        No metakernels matching the specified year were found in the provided path.
    """
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
    """
    Merge the contents of two dictionaries.

    Parameters
    ----------
    dct : dict
        The first dictionary to merge.
    merge_dct : dict
        The second dictionary to merge.

    Returns
    -------
    dict
        The merged dictionary containing the contents of the two given dictionaries.
    """
    new_dct = dct.copy()
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            new_dct[k] = dict_merge(dct[k], merge_dct[k])
        else:
            new_dct[k] = merge_dct[k]

    return new_dct

def dict_to_lower(d):
    return {k.lower():v if not isinstance(v, dict) else dict_to_lower(v) for k,v in d.items()}

def get_isis_preferences(isis_preferences=None):
    """
    Returns ISIS Preference file as a pvl object.

    This will search the following locations, in order, for an IsisPreferences file:

    #. The .Isis directory in your home directory
    #. The directory pointed to by the ISISROOT environment variable
    """
    argprefs = {}
    if isis_preferences:
        if isinstance(isis_preferences, dict):
            argprefs = isis_preferences
        else:
            argprefs = read_pvl(isis_preferences)

    try:
        homeprefs = read_pvl(os.path.join(os.path.expanduser("~"), '.Isis', 'IsisPreferences'))
    except FileNotFoundError as e:
        homeprefs = {}

    try:
        isisrootprefs_path = os.path.join(os.environ["ISISROOT"], 'IsisPreferences')
        isisrootprefs = read_pvl(isisrootprefs_path)
    except (FileNotFoundError, KeyError) as e:
        isisrootprefs = {}

    finalprefs = dict_merge(dict_merge(isisrootprefs, homeprefs), argprefs)

    return finalprefs


def dict_to_lower(d):
    """
    Recursively convert all keys in the dictionary to lower-case.

    Parameters
    ----------
    d : dict
        The dictionary for which to convert the keys to lower-case.

    Returns
    -------
    dict
        The original dictionary with keys converted to lower-case.
    """
    return {k.lower():v if not isinstance(v, dict) else dict_to_lower(v) for k,v in d.items()}


def expandvars(path, env_dict=os.environ, default=None, case_sensitive=True):
    """
    Expand the environment variables in a given string.

    Parameters
    ----------
    path : str
        The string containing variables to be expanded.
    env_dict : dict, optional
        A dictionary containing environment variable definitions, by default os.environ
    default : obj, optional
        A default value for undefined environment variables, by default None
    case_sensitive : bool, optional
        Whether or not the output should match the case of the input, by default True

    Returns
    -------
    str
        The original string with expanded environment variables.

    Raises
    ------
    KeyError
        Raised if the value is not in the dictionary
    """
    if env_dict != os.environ:
        env_dict = dict_merge(env_dict, os.environ)

    while "$" in path:
        user_dict = env_dict if case_sensitive else dict_to_lower(env_dict)

        def replace_var(m):
            group1 = m.group(1) if case_sensitive else m.group(1).lower()
            val = user_dict.get(m.group(2) or group1 if default is None else default)
            if not val:
                raise KeyError(f"Failed to evaluate {m.group(0)} from env_dict. " + 
                               f"Should {m.group(0)} be an environment variable?")

            return val
        reVar = r'\$(\w+|\{([^}]*)\})'
        path = re.sub(reVar, replace_var, path)
    return path


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
    Construct the dependency tree for the body states in a set of kernels.

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

    kernels = ["'"+"$PREFIX/"+k[len(common_prefix):]+"'" for k in kernels]
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


def read_pvl(path, use_jank=False):
    """
    Syntax sugar, used to load a pvl object file from path

    Parameters
    ----------

    path : str
           Path to Pvl file

    use_jank : bool
               If true, uses faster but less reliable JBFPvlParser, else uses standard PVL parser.

    """
    with open(path) as f:
        preftext = f.read().replace('EndGroup', 'End_Group').replace("EndObject", "End_Object")
        if use_jank:
            pvlprefs = JBFPvlParser(open(path).read())
        else:
            pvlprefs = pvl.loads(preftext)

    return pvlprefs


def get_isis_mission_translations(isis_data):
    """
    Use ISIS translation files and return a lookup table.

    Parameters
    ----------

    isis_data : str
                 path to $ISIS3DATA

    Returns
    -------

    : dict
      Dictionary mapping label mission strings to ISIS3 mission strings

    """
    mission_translation_file = read_pvl(os.path.join(isis_data, "base", "translations", "MissionName2DataDir.trn"))
    # For some reason this file takes the form [value, key] for mission name -> data dir
    lookup = [l[::-1] for l in mission_translation_file["MissionName"].getall("Translation")]
    return dict(lookup)


def JBFPvlParser(lines):
    """
    Janky But Faster PVL Parser(TM)

    Only really supports ISIS's Kernel DB files. This is because KernelDB files are sometimes very large for smithed kernels.
    This should bring the parsing time for those DB files from minutes to seconds.
    Still needs nested object/group support and it should be able to read most PVL files.

    Parameters
    ----------

    lines : str
          string body of PVL file.

    Returns : PVLModule
              object representing the parsed PVL


    """
    def JBFKeywordParser(lines):
        keyword = lines[0].split("=")[0]
        value = lines[0].split("=")[1]+"".join(l.strip() for l in lines[1:])

        if "(" in value and ")" in value:
            value = value.replace("(", "").replace(")", "").split(",")
            value = tuple([v.replace("\"", "") for v in value])
        else:
            value = value.strip()

        return keyword.strip(), value


    if isinstance(lines, str):
        lines = lines.split("\n")

    items = []
    lines = [l.strip() for l in lines if l.strip()]
    metadata = []
    for i,l in enumerate(lines):
        if "group = " in l.lower():
            metadata.append([i, "group_start"])
        elif "object = " in l.lower():
            metadata.append([i, "object_start"])
        elif "=" in l:
            metadata.append([i, "keyword"])
        elif "end_group" in l.lower() or "endgroup" in l.lower():
            metadata.append([i, "group_end"])
        elif "end_object" in l.lower() or "endobject" in l.lower():
            metadata.append([i, "object_end"])

    imeta = 0
    while imeta < len(metadata):
        element_start_line, element_type = metadata[imeta]

        if element_type == "keyword":
            next_element_start = metadata[imeta+1][0] if imeta+1<len(metadata) else len(lines)+1
            element_lines = lines[element_start_line:next_element_start]
            items.append(JBFKeywordParser(element_lines))
            imeta+=1
        elif element_type == "group_start":
            group_name = lines[element_start_line].split('=')[1].strip()

            next_meta = [(i,m) for i,m in enumerate(metadata[imeta:]) if m[1] == "group_end"][0]
            next_group_start = next_meta[1][0]

            group_lines = lines[element_start_line+1:next_group_start]
            items.append((group_name, JBFPvlParser(group_lines)))
            imeta += next_meta[0]
        elif element_type == "object_start":
            # duplicate code but whatever
            group_name = lines[element_start_line].split('=')[1].strip()

            next_meta = [(i,m) for i,m in enumerate(metadata[imeta:]) if m[1] == "object_end"][0]
            next_group_start = next_meta[1][0]

            group_lines = lines[element_start_line+1:next_group_start]
            items.append((group_name, JBFPvlParser(group_lines)))
            imeta += next_meta[0]
        elif element_type == "object_end" or element_type == "group_end":
            imeta+=1

    return pvl.PVLModule(items)


def search_isis_db(dbobj, labelobj, isis_data):
    """
    Given an PVL obj of a KernelDB file and an Isis Label for a cube, find the best kernel
    to attach to the cube.

    The Logic here is a bit gross, but it matches ISIS's implementation very closely.


    Parameters
    ----------
    dbobj : PVLModule
            ISIS3 KernelDB file as a loaded PVLModule

    labelobj : PVLModule
               Cube label as loaded PVLModule

    isis_data : str
                 path to $ISISDATA

    Returns
    -------
    : dict
      dictionary containing kernel list and optionally the kernel type if relevant.

    """
    if not dbobj:
        return

    quality = dict(e[::-1] for e  in enumerate(["predicted", "nadir", "reconstructed", "smithed"]))

    utc_start_time = labelobj["IsisCube"]["Instrument"]["StartTime"]
    utc_stop_time = labelobj["IsisCube"]["Instrument"]["StopTime"]

    run_time = None
    dependencies = None
    kernels = []
    typ = None
    types = []

    # flag is set when a kernel is found matching the start time but not stop time
    # and therefore a second pair needs to be found
    partial_match = False

    # Flag is set when kernels encapsulating the entire image time is found
    full_match = False

    for selection in dbobj.getall("Selection"):
        files = selection.getall("File")
        if not files:
            raise Exception(f"No File found in {selection}")

        # selection criteria
        matches = []
        if "Match" in selection:
            matches = selection.getall("Match")

        times = []
        if "Time" in selection:
            times = selection.getall("Time")

        files = [path.join(*file) if isinstance(file, list) else file  for file in files]

        for i,time in enumerate(times):
            isis_time_format = '%Y %b %d %H:%M:%S.%f TDB'
            other_isis_time_format =  '"%Y %b %d %H:%M:%S.%f TDB"'

            try:
                time = [datetime.strptime(time[0].strip(), isis_time_format),
                        datetime.strptime(time[1].strip(), isis_time_format)]
            except Exception as e:
                time = [datetime.strptime(time[0].strip(), other_isis_time_format),
                        datetime.strptime(time[1].strip(), other_isis_time_format)]

            time[0] = pytz.utc.localize(time[0])
            time[1] = pytz.utc.localize(time[1])
            start_time_in_range = utc_start_time >= time[0] and utc_start_time <= time[1]
            stop_time_in_range = utc_stop_time >= time[0] and utc_stop_time <= time[1]
            times[i] = stop_time_in_range, stop_time_in_range

        for i,match in enumerate(matches):
            matches[i] = labelobj["IsisCube"][match[0].strip()][match[1].strip()].lower().strip() == match[2].lower().strip()

        if any(matches if matches else [True]):
            for i,f in enumerate(files):
                if isinstance(f, tuple):
                    f = os.path.join(*[e.strip() for e in f])

                full_path = os.path.join(isis_data, f).replace("$", "").replace("\"", "")
                if "{" in full_path:
                    start = full_path.find("{")
                    stop = full_path.find("}")
                    full_path = full_path[:start] + "?"*(stop-start-1) + full_path[stop+1:]
                if '?' in full_path:
                    full_path = sorted(glob(full_path))[-1]
                files[i] = full_path

            if times:
                have_start_match, have_stop_match = list(map(list, zip(*times)))
                typ = selection.get("Type", None)
                typ = typ.lower().strip() if typ else None

                current_quality = max([quality[t.lower().strip()] for t in types if t]) if any(types) else 0

                if any(have_start_match) and any(have_stop_match):
                    # best case, the image is fully encapsulated in the kernel
                    full_match = True
                    if quality[typ] >= current_quality:
                        kernels = files
                        types = [selection.get("Type", None)]
                elif any(have_start_match):
                    kernels.extend(files)
                    types.append(selection.get("Type", None))
                    partial_match = True
                elif any(have_stop_match):
                    if partial_match:
                        if quality[typ] >= current_quality:
                            kernels.extend(files)
                            types.append(selection.get("Type", None))
                            full_match = True
            else:
                full_match = True
                kernels = files
                types = [selection.get("Type", None)]

    if partial_match:
        # this can only be true if a kernel matching start time was found
        # but not the end time
        raise Exception("Could not find kernels encapsulating the full image time")

    kernels = {"kernels" : kernels}
    if any(types):
        kernels["types"] = types
    return kernels


def merge_kernels(dict1, dict2, strategy='combine'):
    """
    Merge two dictionaries with configurable conflict resolution strategies.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to merge.
    dict2 : dict
        The second dictionary to merge.
    strategy : {'right', 'left', 'combine'}, optional
        The strategy to use when both dictionaries have the same key:
        
        - 'right': Values from `dict2` take precedence over `dict1`.
        - 'left': Values from `dict1` take precedence over `dict2`.
        - 'combine': If a key exists in both, combine the values into a list
          (removing duplicates). If either value is already a list, the result
          will be a flattened list of unique values.

        Default is 'combine'.

    Returns
    -------
    merged : dict
        The merged dictionary according to the specified strategy.

    Examples
    --------
    >>> merge_kernels({'a': 1, 'b': 2}, {'b': 3, 'c': 4}, strategy='right')
    {'a': 1, 'b': 3, 'c': 4}

    >>> merge_kernels({'a': 1, 'b': 2}, {'b': 3, 'c': 4}, strategy='left')
    {'a': 1, 'b': 2, 'c': 4}

    >>> merge_kernels({'a': 1, 'b': 2}, {'b': 3, 'c': 4}, strategy='combine')
    {'a': 1, 'b': [2, 3], 'c': 4}
    """
    logger.debug(f"merge_kernels: dict1: {dict1}, dict2: {dict2}, strategy: {strategy}")
    if not dict1:
        return dict2 
    if not dict2:
        return dict1

    merged = dict1.copy()

    for key, value in dict2.items():
        if key in merged:
            if strategy == 'left':
                continue  # Keep dict1 value
            elif strategy == 'right':
                merged[key] = value  # Use dict2 value
            elif strategy == 'combine':
                # Combine into list
                if not isinstance(merged[key], list):
                    merged[key] = [merged[key]]
                if isinstance(value, list):
                    merged[key] = merged[key] + value
                    merged[key] = list(set(merged[key]))
                elif value not in merged[key]:
                    merged[key].append(value)
        else:
            merged[key] = value
    
    return merged
 