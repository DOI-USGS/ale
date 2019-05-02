import os
import numpy as np
import pvl
import spiceypy as spice
from glob import glob

from ale import config
from ale.util import get_metakernels
from ale.drivers.base import LineScanner, NaifSpice, Pds3Label, Driver

class LroLrocPds3LabelNaifSpiceDriver(Driver, NaifSpice, Pds3Label, LineScanner):
  """
  Driver for reading Lroc NACL, NACR, WAC labels. Requires a Spice mixin to
  acquire addtional ephemeris and instrument data located exclusively in
  spice kernels.
  """

  @property
  def instrument_id(self):
    """
    Returns an instrument id for uniquely identifying the instrument, but often
    also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
    are the same ID that spice lib expects as input to bods2c calls.

    Ignores Wide Angle for now

    Returns
    -------
    : str
      instrument id
    """

    instrument = self.label.get("INSTRUMENT_ID")

    # NACL, NACR, or WAC
    frame_id = self.label.get("FRAME_ID")

    if instrument == "LROC" and frame_id == "LEFT":
      return "LRO_LROCNACL"
    elif instrument == "LROC" and frame_id == "RIGHT":
      return "LRO_LROCNACR"
        # WAC not implemented yet

  @property
  def metakernel(self):
    """
    Returns latest instrument metakernels

    Returns
    -------
    : string
      Path to latest metakernel file
    """
    metakernel_dir = config.lro

    mks = sorted(glob(os.path.join(metakernel_dir, '*.tm')))
    if not hasattr(self, '_metakernel'):
      for mk in mks:
        if str(self.start_time.year) in os.path.basename(mk):
          self._metakernel = mk
    return self._metakernel


  @property
  def spacecraft_name(self):
    """
    Spacecraft name used in various SPICE calls to acquire
    ephemeris data.

    Returns
    -------
    : str
      Spacecraft name
    """
    return "LRO"


  @property
  def _odtk(self):
    """
    Returns
    -------
    : list
      Radial distortion coefficients
    """
    return spice.gdpool('INS{}_OD_K'.format(self.ikid), 0, 1).tolist()

  @property
  def optical_distortion(self):
    """
    LRO LROC NAC does not use the default distortion model so we need to overwrite the
    method packing the distortion model into the ISD.

    """
    return {
      "lrolrocnac": {
        "coefficients": self._odtk,
      }
    }

