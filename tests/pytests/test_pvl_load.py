import pvl
import ale
import os
import json
import unittest
from unittest.mock import patch

from ale.drivers.clementine_drivers import ClementineIsisLabelNaifSpiceDriver

from conftest import get_image_kernels, convert_kernels, get_image_label

class test_pvl_loads(unittest.TestCase):

    def setUp(self):
        lbl = get_image_label("LUA3107H.161", "isis3")
        self.driver = ClementineIsisLabelNaifSpiceDriver(lbl)
        
    def test_load_kernels(self):
        kerns = get_image_kernels('LUA3107H.161')
        updated_kerns, binary_kerns = convert_kernels(kerns)
        yield updated_kerns
        for kern in binary_kerns:
            os.remove(kern)

    def test_pvl_load(self):
        cube_label = get_image_label('LUA3107H.161', "isis3")
        cube_pvl_obj = pvl.load(cube_label)
        isd = ale.loads(cube_pvl_obj, props={'kernels': self.test_load_kernels(), 'exact_ck_times': False}, only_naif_spice=True)
        isd_obj = json.loads(isd)
        return isd_obj
