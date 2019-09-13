import spiceypy as spice

import ale
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.base import Driver

class Voyager2IssnacIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, Driver):

    @property
    def sensor_model_version(self):
        return 1

    @property
    def ikid(self):
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def spacecraft_name(self):
        return super().spacecraft_name.replace('_', ' ')

    @property
    def pixel_size(self):
        return spice.gdpool('INS{}_PIXEL_PITCH'.format(self.ikid), 0, 1)[0]

    @property
    def detector_center_sample(self):
        return 499.5

    @property
    def detector_center_line(self):
        return 499.5

    @property
    def ephemeris_start_time(self):
        inital_time = spice.utc2et(self.utc_start_time.isoformat())
        # To get shutter end (close) time, subtract 2 seconds from the start time
        updated_time = inital_time - 2
        # To get shutter start (open) time, take off the exposure duration from the end time.
        start_time = updated_time - self.exposure_duration
        return start_time

    @property
    def ephemeris_stop_time(self):
        return self.ephemeris_start_time + self.exposure_duration
