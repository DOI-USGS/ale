import os
import json
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import spiceypy as spice

import ale
from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver, MroCtxIsisLabelIsisSpiceDriver

from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts

@pytest.fixture()
def isis_compare_dict():
    return {
    }

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('st_2385617364_x')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
def test_load(test_kernels, label_type, formatter, isis_compare_dict):
    label_file = get_image_label('st_2385617364_x', label_type)

    isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter=formatter)
    isd_obj = json.loads(usgscsm_isd_str)

    assert compare_dicts(isd_obj, isis_compare_dict) == []
