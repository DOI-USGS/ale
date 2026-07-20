import os
import json
import unittest

import pytest
import pyspiceql as psql 

import ale
# from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver, MroCtxIsisLabelIsisSpiceDriver
from conftest import get_image, get_image_kernels, get_isd, convert_kernels, get_image_label, compare_dicts

@pytest.fixture(scope='module')
def test_ctx_kernels():
    kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type, kernel_type", [('pds3', 'naif'), ('isis3', 'naif')])
def test_mro_ctx_load(test_ctx_kernels, label_type, kernel_type):
    label_file = get_image_label('B10_013341_1010_XN_79S172W', label_type)

    if label_type == 'isis3' and kernel_type == 'isis':
        label_file = get_image('B10_013341_1010_XN_79S172W')
        isd_str = ale.loads(label_file, props={'attach_kernels': False})
        compare_isd = get_isd('ctx_isis')
    else:
        isd_str = ale.loads(label_file, props={'kernels': test_ctx_kernels, 'attach_kernels': False, 'reduction': 'hermite'}, verbose=True)
        compare_isd = get_isd('ctx')

    isd_obj = json.loads(isd_str)

    if label_type == 'isis3' and kernel_type == 'naif':
        compare_isd['image_samples'] = 5000
        compare_isd["projection"] = '+proj=sinu +lon_0=148.36859083039 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs'
        compare_isd["geotransform"] = [-219771.1526456, 1455.4380969907, 0.0, 5175537.8728989, 0.0, -1455.4380969907]

    compare_isd["instrument_position"]["ephemeris_times"] = [297088762.24158406, 297088762.61698407, 297088762.9923841]
    compare_isd["instrument_position"]["positions"] = [[-1885.2980675616825, 913.165223601331, -2961.9668280021374], 
                                                       [-1886.0348306902727, 912.111196630215, -2961.8260667990207], 
                                                       [-1886.771356647758, 911.0570545575628, -2961.6849329349425]]
    compare_isd["instrument_position"]["velocities"] = [[-1.962923764670415, -2.8075907222127268, 0.37446657801488403], 
                                                        [-1.9622923277060045, -2.807897516114349, 0.3754593119812068], 
                                                        [-1.9616606456386552, -2.808203957776863, 0.37645199765660525]]

    print(json.dumps(isd_obj))
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []