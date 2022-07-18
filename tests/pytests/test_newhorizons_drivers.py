import pytest
import ale
import os
import json
import pvl

import numpy as np
from ale.drivers import co_drivers
from ale.formatters.isis_formatter import to_isis
from ale.formatters.formatter import to_isd
from ale.base.data_isis import IsisSpice
import unittest
from unittest.mock import patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

from ale.drivers.nh_drivers import NewHorizonsLorriIsisLabelNaifSpiceDriver, NewHorizonsLeisaIsisLabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label, get_isd

image_dict = {
    'lor_0034974380_0x630_sci_1': get_isd("nhlorri"),
    'lsb_0296962438_0x53c_eng': get_isd("nhleisa")
}


@pytest.fixture()
def test_kernels(scope="module"):
    updated_kernels = {}
    binary_kernels = {}
    for image in image_dict.keys():
        kernels = get_image_kernels(image)
        updated_kernels[image], binary_kernels[image] = convert_kernels(kernels)
    yield updated_kernels
    for kern_list in binary_kernels.values():
        for kern in kern_list:
            os.remove(kern)

# Test load of nh lorri labels
@pytest.mark.parametrize("image", ['lor_0034974380_0x630_sci_1'])
def test_nhlorri_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_isd) == []

# Test load of nh leisa labels
@pytest.mark.parametrize("image", ['lsb_0296962438_0x53c_eng'])
def test_nhleisa_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_isd) == []
