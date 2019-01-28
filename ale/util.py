import os
from glob import glob
from itertools import filterfalse, groupby
from os import path

from ale import config


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
