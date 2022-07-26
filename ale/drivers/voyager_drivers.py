import spiceypy as spice

import ale
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver

class VoyagerCameraIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        sc_lookup = {
        "VOYAGER_1" : "VG1",
        "VOYAGER_2" : "VG2"
        }
        sensor_lookup = {
        "NARROW_ANGLE_CAMERA" : "ISSNA",
        "WIDE_ANGLE_CAMERA" : "ISSWA"
        }
        return sc_lookup[super().spacecraft_name] + '_' + sensor_lookup[super().instrument_id]

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label['IsisCube']['Instrument']['InstrumentId']

    @property
    def sensor_model_version(self):
        return 1

    @property
    def ikid(self):
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def spacecraft_name(self):
        name = super().spacecraft_name.replace('_', ' ')
        if name.split(' ')[0] != "VOYAGER":
            raise Exception("{name} for label is not a valid Voyager spacecraft name")
        return name

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
        inital_time = spice.utc2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
        # To get shutter end (close) time, subtract 2 seconds from the start time
        updated_time = inital_time - 2
        # To get shutter start (open) time, take off the exposure duration from the end time.
        start_time = updated_time - self.exposure_duration
        return start_time

    @property
    def ephemeris_stop_time(self):
        return self.ephemeris_start_time + self.exposure_duration
