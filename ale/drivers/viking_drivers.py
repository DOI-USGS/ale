import spiceypy as spice

import ale
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.base import Driver

class VikingIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, Driver):

    @property
    def spacecraft_name(self):
        """
        Overridden to work with spice calls.

        Returns
        -------
        : str
          Name of the spacecraft.
        """
        return super().spacecraft_name.replace('_', ' ')

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
        Overridden to use the alternate instrument ID '-27999' as defined in the
        SCLK kernel. Also computes an offset to match what is being done in ISIS code.
        Expects spacecraft_clock_start_count to be defined.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        if spacecraft_name == 'VIKING ORBITER 1':
            ephemeris_start_time = spice.scs2e(-27999, str(self.spacecraft_clock_start_count))

        # else use vo2 alt ID
        else:
            ephemeris_start_time = spice.scs2e(-30999, str(self.spacecraft_clock_start_count))

        if self.exposure_duration <= .420:
            offset1 = 7.0 / 8.0 * 4.48
        else:
            offset1 = 3.0 / 8.0 * 4.48
        offset2 = 1.0 / 64.0 * 4.48

        return ephemeris_start_time + offset1 + offset2
