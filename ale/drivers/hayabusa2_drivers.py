import spiceypy as spice

import ale
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver

class Hayabusa2ONCIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        lookup_table = {'ONC-W2': 'HAYABUSA2_ONC-W2'}
        return lookup_table[super().instrument_id]

    @property
    def sensor_model_version(self):
        return 1

    @property
    def spacecraft_name(self):
        name = super().spacecraft_name.replace('-', '')
        if name.split(' ')[0] != "HAYABUSA2":
            raise Exception(f"{name} for label is not a valid Hayabusa 2 spacecraft name")
        return name
