from glob import glob
import os

import numpy as np
import pvl
import spiceypy as spice

from ale import config
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import LineScanner


class MroCtxIsisLabelIsisSpiceDriver(Driver, IsisSpice, LineScanner, RadialDistortion):

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
        return "N/A"

    @property
    def spacecraft_id(self):
        return "N/A"

    @property
    def ikid(self):
        return int(self.label["IsisCube"]["Kernels"]["NaifFrameCode"])

    @property
    def line_exposure_duration(self):
        return self.label["IsisCube"]["Instrument"]["LineExposureDuration"].value * 0.001 # Scale to seconds


class MroCtxIsisLabelNaifSpiceDriver(IsisLabel, Driver, NaifSpice, LineScanner, RadialDistortion):
    """
    Driver for reading CTX ISIS labels.
    """

    @property
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        metakernel_dir = config.mro
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            self._metakernel = None
            for mk in mks:
                if str(self.start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel

    @property
    def instrument_id(self):
        return "MRO_CTX"

    @property
    def ephemeris_start_time(self):
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.label['IsisCube']['Instrument']['SpacecraftClockCount']
            self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, sclock)
        return self._ephemeris_start_time

    @property
    def line_exposure_duration(self):
        if not hasattr(self, '_line_exposure_duration'):
            self._line_exposure_duration = self.label['IsisCube']['Instrument']['LineExposureDuration'].value * 0.001
        return self._line_exposure_duration

    @property
    def spacecraft_name(self):
        return "MRO"


class MroCtxPds3LabelNaifSpiceDriver(Pds3Label, NaifSpice, LineScanner, RadialDistortion, Driver):
    """
    Driver for reading CTX PDS3 labels. Requires a Spice mixin to acquire addtional
    ephemeris and instrument data located exclusively in spice kernels.
    """

    @property
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        metakernel_dir = config.mro
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            self._metakernel = None
            for mk in mks:
                if str(self.utc_start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel

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
            'CONTEXT CAMERA':'MRO_CTX'
        }

        return id_lookup[super().instrument_id]

    @property
    def spacecraft_name(self):
        """
        Returns the spacecraft name used in various Spice calls to acquire
        ephemeris data.
        
        Returns
        -------
        : str
          spacecraft name
        """
        name_lookup = {
            'MARS_RECONNAISSANCE_ORBITER': 'MRO'
        }
        return name_lookup[super().spacecraft_name]

    @property
    def detector_start_line(self):
        return 1

    @property
    def detector_start_sample(self):
        return self.label.get('SAMPLE_FIRST_PIXEL', 0)
    
    @property
    def sensor_model_version(self):
        return 1
