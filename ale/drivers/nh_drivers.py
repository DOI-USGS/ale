from glob import glob
import os

import struct
import pvl
import spiceypy as spice
import numpy as np

from ale.base import Driver
from ale.base.type_distortion import NoDistortion, LegendreDistortion
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer

class NewHorizonsLorriIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, LegendreDistortion, Driver):
    """
    Driver for reading New Horizons LORRI ISIS3 Labels. These are Labels that have been
    ingested into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            "LORRI" : "NH_LORRI"
        }
        return id_lookup[super().instrument_id]


    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        For LORRI, there are two options associated with different binning modes:
        1x1 binning: -98301
        4x4 binning: -98302

        Returns
        -------
        : integer
          Naif Integer ID code for the instrument
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def detector_center_line(self):
        """
        The center of the CCD in detector pixels
        Expects ikid to be defined. this should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[0])

    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels
        Expects ikid to be defined. this should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[1])

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']


class NewHorizonsMvicIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, LegendreDistortion, Driver):
    """
    Driver for reading New Horizons MVIC ISIS3 Labels. These are Labels that have been
    ingested into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """

    @property
    def parent_id(self):
        """
        The base naif id of the spacecraft.  For New Horizons, this is -98000.

        Required for distortion coefficients, which are not unique to instruments,
        but are instead shared by all instruments on the spacecraft + residuals.

        Returns
        -------
        : int
          Naif id of the spacecraft
        """
        return round(self.ikid, -2)


    @property
    def sensor_model_version(self):
        """
        Returns instrument model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1


    @property
    def instrument_name(self):
        """
        The name of the instrument.  This is not included in the .fit label, but is
        present in the .lbl file, so it is not present in ISIS conversion, and it
        must be hard-coded.

        Returns
        -------
        : str
          Name of the instrument
        """
        return "MULTISPECTRAL VISIBLE IMAGING CAMERA"


    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            "MVIC_TDI" : "NH_MVIC"
        }
        return id_lookup[super().instrument_id]


    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        Returns
        -------
        : integer
          Naif Integer ID code for the instrument
        """
        print(self.label['IsisCube']['Kernels']['NaifFrameCode'][0])
        return self.label['IsisCube']['Kernels']['NaifFrameCode'][0]


    @property
    def detector_center_line(self):
        """ Returns detector center line.  This information is found in ik/nh_ralph_v100.ti, which
        is not loaded as an ik."""
        return -1


    @property
    def detector_center_sample(self):
        """ Returns detector center line.  This information is found in ik/nh_ralph_v100.ti, which
        is not loaded as an ik."""
        return 0


    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']


    @property
    def odtx(self):
        """
        Returns the x coefficient for the optical distortion model
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.parent_id),0, 20).tolist()


    @property
    def odty(self):
        """
        Returns the y coefficient for the optical distortion model.
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.parent_id), 0, 20).tolist()


    @property
    def naif_keywords(self):
        """
        Adds base NH instrument distortion, which is shared among all instruments on NH.

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        return {**super().naif_keywords,
                f"INS{self.parent_id}_DISTORTION_COEF_X": self.odtx,
                f"INS{self.parent_id}_DISTORTION_COEF_Y": self.odty}
