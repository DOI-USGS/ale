import subprocess
import os
import re
import warnings
import numpy as np
import json
import pvl
from glob import glob
from pathlib import Path

import ale

from ale.base.data_isis import read_table_data

class SimpleSpice():
    def scs2e(self, *args):
        return 0.1
    def bods2c(self, x):
        return -12345
    def frmnam(self, id):
        return 'Test_Frame'
    def gdpool(self, key, x, length):
        return np.ones(length)
    def gipool(self, key, x, length):
        return np.arange(length)
    def bodvrd(self, key, x, length):
        return (3, np.ones(length,))
    def spkpos(self, *args):
        return (np.ones(3), None)
    def spkezr(self, *args):
        return (np.ones(6), None)
    def furnsh(self, *args):
        return
    def unload(self, *args):
        return
    def pxform(self, *args):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    def m2q(self, *args):
        return np.asarray([1,2,3,4])
    def bodn2c(self, *args):
        return "SPACE"
    def sct2e(self, *args):
        return 0.1
    def cidfrm(self, *args):
        return (2000, "Test_Body_Frame", True)

def get_mockkernels(self, *args):
    return "some_metakernel"

def compare_dicts(ldict, rdict):
    differences = []
    for key in rdict:
        if key not in ldict:
            differences.append(f'Key {key} is present in the right dict, but not the left dict.')
    for key, item in ldict.items():
        if key not in rdict:
            differences.append(f'Key {key} is present in the left dict, but not the right dict.')
        elif isinstance(item, dict):
            differences.extend(compare_dicts(item, rdict[key]))
        elif isinstance(item, np.ndarray) or isinstance(item, list):
            if len(item) != len(rdict[key]):
                differences.append(f'Array sizes of key {key} are not equal {item} : {rdict[key]}.')
            elif not np.allclose(item, rdict[key]):
                differences.append(f'Array values of key {key} are not almost equal {item} : {rdict[key]}.')
        elif isinstance(item, str):
            if item.lower() != rdict[key].lower():
                differences.append(f'Values of key {key} are not equal {item} : {rdict[key]}.')
        elif isinstance(item, float):
            if not np.isclose(item, rdict[key]):
                differences.append(f'Floating point values of key {key} are not close {item} : {rdict[key]}.')
        else:
            if item != rdict[key]:
                differences.append(f'Values of key {key} are not equal {item} : {rdict[key]}.')
    return differences

data_root = os.path.join(Path(__file__).parent.absolute(), 'data')
dirs = next(os.walk(data_root, topdown=True))[1]
dirs = [d for d in dirs if not d.startswith('.')]
image_2_data = {}

for d in dirs:
    tmp = os.path.join(data_root, d)
    image_2_data[d] = [os.path.join(tmp, f) for f in os.listdir(tmp) if not f.startswith('.') and os.path.splitext(f)[1] != '.lbl']
    # force IAKs to the back of the list
    image_2_data[d] = sorted(image_2_data[d], key=lambda x: "Addendum" in x)

def get_image_label(image, label_type='pds3'):
    if not isinstance(image, str):
        try:
            image = str(image)
        except:
            raise KeyError('Cannot coerce requested image name to string')

    label_file = glob(os.path.join(data_root, '*',f'{image}_{label_type}.lbl'))
    if not label_file:
        raise Exception(f'Could not find label file for {image}')

    return label_file[0]

def get_image(image):
    if not isinstance(image, str):
        try:
            image = str(image)
        except:
            raise KeyError('Cannot coerce requested image name to string')

    label_file = glob(os.path.join(data_root, '*',f'{image}.cub'))
    if not label_file:
        raise Exception(f'Could not find label file for {image}')

    return label_file[0]


def get_table_data(image, table_name):
    if not isinstance(image, str):
        try:
            image = str(image)
        except:
            raise KeyError('Cannot coerce requested image name to string')

    table_file = glob(os.path.join(data_root, '*',f'{image}_{table_name}.tbl'))
    if not table_file:
        raise Exception(f'Could not find {table_name} table file for {image}')

    table_label = pvl.load(table_file[0])
    return read_table_data(table_label["Table"], table_file[0])


def get_isd(instrument):
    if not isinstance(instrument, str):
        raise KeyError('instrument name is not a string')

    label_file = glob(os.path.join(data_root, 'isds',f'{instrument}_isd.json'))
    if not label_file:
        raise Exception(f'Could not find label file for {instrument}')

    return json.load(open(label_file[0]))


def get_image_kernels(image):
    """
    Get the kernels to use with a test image.

    Parameters
    ----------
    image : str
            The image name to get kernels for. I.E. 'EN1072174528M'

    Returns
    -------
    : list
      A list containing the absolute paths to the kernels for the images.
      This list contains all of the kernel files available in the test image's
      data directory which may contain transfer files that need to be converted
      to binary files.
    """
    if not isinstance(image, str):
        try:
            image = str(image)
        except:
            raise KeyError('Cannot coerce requested image name to string')
    if image in image_2_data:
        return image_2_data[image]
    else:
        raise KeyError('Could not find test data for' + image)

def convert_kernels(kernels):
    """
    Convert any transfer kernels in a list to binary kernels

    Parameters
    ----------
    kernels : list
              A list of kernels. Only transfer kernels present in the list will
              be converted. Non-transfer kernels will be ignored.

    Returns
    -------
    updated_kernels : list
                      The updated kernel list where all transfer kernels have
                      been replaced with their converted binary kernel. This
                      is designed to be passed directly to furnsh.
    binary_kernels : list
                     The list of binary kernels created.
    """
    binary_kernels = []
    updated_kernels = []
    for kernel in kernels:
        split_kernel = os.path.splitext(kernel)
        if 'x' in split_kernel[1].lower():
            # Get the full path to the kernel then truncate it to the relative path
            path = os.path.join(data_root, kernel)
            path = os.path.relpath(path)
            try:
                bin_output = subprocess.run(['tobin', path],
                                            capture_output=True, check=True)
                matches = re.search(r'To: (.*\.b\w*)', str(bin_output.stdout))
                if not matches:
                    warnings.warn('Failed to convert transfer kernel, ' + kernel + ', skipping...')
                else:
                    kernel = matches.group(1)
                    binary_kernels.append(kernel)
            except:
                warnings.warn(f"Unable to convert {path} to binary kernel")
        updated_kernels.append(kernel)
    return updated_kernels, binary_kernels
