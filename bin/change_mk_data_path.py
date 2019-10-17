#!/usr/bin/env python

import argparse
from glob import glob
from os import path
import re

parser = argparse.ArgumentParser()
parser.add_argument('mk_path', action='store', help='Path containing metakernel (*.tm) files.')
parser.add_argument('new_data_path', action='store', help='new data path containing kernel, this path will overwrite the current data path in the metakernels')
args = parser.parse_args()

mks = glob(path.join(args.mk_path,'*.[Tt][Mm]'))

for mk in mks:
    with open(mk, 'r') as f:
        lines = f.readlines()

    lines_to_change = []
    for i,l in enumerate(lines):
        match = re.search('PATH_VALUES\s*=', l)
        if match:
            lines_to_change.append(i)

    if not lines_to_change:
        print(f'No path values in {mk}')

    for i in lines_to_change:
        lines[i] = f'      PATH_VALUES     = ( \'{args.new_data_path}\' )\n'
    with open(mk, 'w') as f:
        f.write(''.join(lines))

