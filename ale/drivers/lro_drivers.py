import os
import numpy as np
import pvl
import spiceypy as spice
from glob import glob

from ale import config
from ale.util import get_metakernels
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import LineScanner


class LroLrocPds3LabelNaifSpiceDriver(NaifSpice, Pds3Label, LineScanner, Driver):
    """
    Driver for reading LROC NACL, NACR (not WAC, it is a push frame) labels. Requires a Spice mixin to
    acquire addtional ephemeris and instrument data located exclusively in SPICE kernels, A PDS3 label,
    and the LineScanner and Driver bases.
    """

    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """

        instrument = super().instrument_id

        frame_id = self.label.get("FRAME_ID")

        if instrument == "LROC" and frame_id == "LEFT":
            return "LRO_LROCNACL"
        elif instrument == "LROC" and frame_id == "RIGHT":
            return "LRO_LROCNACR"


    @property
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : str
          Path to latest metakernel file
        """
        metakernel_dir = config.lro

        mks = sorted(glob(os.path.join(metakernel_dir, '*.tm')))
        if not hasattr(self, '_metakernel'):
            self._metakernel = None
            for mk in mks:
                if str(self.utc_start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel


    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. LROC NAC img PDS3 labels do not the have SPACECRAFT_NAME keyword, so we
        override it here to use the label_pds3 property for instrument_host_id

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_id

    @property
    def sensor_model_version(self):
        """
        Returns ISIS instrument sensor model version number

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def detector_start_sample(self):
        """
        Returns the starting sample contained in the image

        Returns
        -------
        : int
          Returns the starting sample
        """
        return 1

    @property
    def detector_start_line(self):
        """
        Returns the starting line contained in the image

        Returns
        -------
        : int
          Returns the starting line
        """
        return 1

    @property
    def usgscsm_distortion_model(self):
        """
        The distortion model name with its coefficients

        LRO LROC NAC does not use the default distortion model so we need to overwrite the
        method packing the distortion model into the ISD.

        Returns
        -------
        : dict
          Returns a dict with the model name : dict of the coefficients
        """

        return {"lrolrocnac":
                {"coefficients": self.odtk}}

    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients. There is only one coefficient for LROC NAC l/r
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid), 0, 1).tolist()

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correciton and abberation correction to
        use in NAIF calls.

        LROC is specifically set to not use light time correction because it is
        so close to the surface of the moon that light time correction to the
        center of the body is incorrect.

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'
