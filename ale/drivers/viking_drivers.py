import spiceypy as spice

import ale
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver

sensor_name_lookup = {
    "VISUAL_IMAGING_SUBSYSTEM_CAMERA_A" : "Visual Imaging Subsystem Camera A",
    "VISUAL_IMAGING_SUBSYSTEM_CAMERA_B" : "Visual Imaging Subsystem Camera B"
}

spacecraft_name_lookup = {
    'VIKING_ORBITER_1': 'VIKING ORBITER 1',
    'VIKING_ORBITER_2': 'VIKING ORBITER 2'
}

alt_id_lookup = {
    'VIKING ORBITER 1': -27999,
    'VIKING ORBITER 2':-30999
}

class VikingIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, Driver):



    @property
    def instrument_id(self):
        """
        Overridden to check that the instrument ID is correct

        Returns
        -------
        : str
          The name of the sensor
        """
        instrument_id = super().instrument_id

        if(instrument_id not in sensor_name_lookup):
            raise Exception (f'Instrument ID [{instrument_id}] is wrong.')

        return instrument_id

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return sensor_name_lookup[super().instrument_id]

    @property
    def spacecraft_name(self):
        """
        Overridden to work with spice calls.

        Returns
        -------
        : str
          Name of the spacecraft.
        """
        return spacecraft_name_lookup[super().spacecraft_name]

    @property
    def alt_ikid(self):
        """
        Viking Orbiter 1 & 2 each have an alternate naif id code as defined in the
        SCLK kernels.
        Expects spacecraft name to be defined.
        Returns -27999 for vo1 and -30999 for vo2

        Returns
        -------
        : integer
        Alternate Naif Integer ID code for the instrument
        """

        return alt_id_lookup[self.spacecraft_name]

    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call.

        Returns
        -------
        : integer
          Naif Integer ID code for the instrument
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def ephemeris_start_time(self):
        """
        Overridden to use the alternate instrument ID. Also computes an offset to match
        what is being done in ISIS code.
        Expects spacecraft_clock_start_count to be defined.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        ephemeris_start_time = spice.scs2e(self.alt_ikid, str(self.spacecraft_clock_start_count))

        if self.exposure_duration <= .420:
            offset1 = 7.0 / 8.0 * 4.48
        else:
            offset1 = 3.0 / 8.0 * 4.48
        offset2 = 1.0 / 64.0 * 4.48

        return ephemeris_start_time + offset1 + offset2

class VikingIsisLabelIsisSpiceDriver(Framer, IsisLabel, IsisSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Overriden to check that the instrument ID is correct

        Returns
        -------
        : str
          The name of the sensor
        """
        instrument_id = super().instrument_id

        if(instrument_id not in sensor_name_lookup):
            raise Exception (f'Instrument ID [{instrument_id}] is wrong.')

        return instrument_id

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return sensor_name_lookup[super().instrument_id]

    @property
    def spacecraft_name(self):
        """
        Overridden to work with spice calls.

        Returns
        -------
        : str
          Name of the spacecraft.
        """
        return spacecraft_name_lookup[super().spacecraft_name]
