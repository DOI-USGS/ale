import subprocess
import os
import re
import numpy as np
import ale

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

ale_root = os.path.split(ale.__file__)[0]
data_root = os.path.join(ale_root, '../tests/pytests/data')
dirs = next(os.walk(data_root, topdown=True))[1]
dirs = [d for d in dirs if not d.startswith('.')]
image_2_data = {}

for d in dirs:
    tmp = os.path.join(data_root, d)
    image_2_data[d] = [os.path.join(tmp, f) for f in os.listdir(tmp) if not f.startswith('.')]

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
        print('Checking kernel', kernel)
        split_kernel = os.path.splitext(kernel)
        if 'x' in split_kernel[1].lower():
            print('Converting transfer kernel', kernel)
            bin_output = subprocess.run(['tobin', os.path.join(data_root, kernel)],
                                        capture_output=True, check=True)
            matches = re.search(r'To: (.*\.b\w*)', str(bin_output.stdout))
            if not matches:
                print('Failed to convert transfer kernel', kernel, 'skipping...')
            else:
                kernel = matches.group(1)
                binary_kernels.append(kernel)
                print('New binary kernel', kernel)
        updated_kernels.append(kernel)
    return updated_kernels, binary_kernels
