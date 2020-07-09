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

from ale.drivers.nh_drivers import NewHorizonsLorriIsisLabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label, get_isd

@pytest.fixture()
def test_kernels(scope="module"):
    kernels = get_image_kernels("lor_0034974380_0x630_sci_1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_newhorizons_load(test_kernels):
    label_file = get_image_label("lor_0034974380_0x630_sci_1", "isis")
    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    isd_dict = json.loads(usgscsm_isd_str)
    isis_compare_dict = get_isd('newhorizons')

    assert compare_dicts(isd_dict, isis_compare_dict) == []
