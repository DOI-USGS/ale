import re
import spiceypy as spice
import os
import math
import numpy as np

from glob import glob

import ale
from ale.base import Driver
from ale.base.label_isis import IsisLabel
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer, LineScanner
from ale.base.type_distortion import NoDistortion
from ale.base.data_isis import read_table_data
from ale.base.data_isis import parse_table
from ale.transformation import ConstantRotation, FrameChain

ID_LOOKUP = {
    "FC1" : "DAWN_FC1",
    "FC2" : "DAWN_FC2"
}

degs_per_rad = 57.2957795

class DawnFcPds3NaifSpiceDriver(Framer, Pds3Label, NaifSpice, Driver):
    """
    Dawn driver for generating an ISD from a Dawn PDS3 image.
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore the
        the same ID that Spice expects in bods2c calls. Expects instrument_id to be
        defined from the PDS3Label mixin. This should be a string containing the short
        name of the instrument. Expects filter_number to be defined. This should be an
        integer containing the filter number from the PDS3 Label.

        Returns
        -------
        : str
          instrument id
        """
        instrument_id = super().instrument_id
        filter_number = self.filter_number

        return "{}_FILTER_{}".format(ID_LOOKUP[instrument_id], filter_number)

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data. Dawn does not have a SPACECRAFT_NAME keyword, therefore
        we are overwriting this method using the instrument_host_id keyword instead.
        Expects instrument_host_id to be defined. This should be a string containing
        the name of the spacecraft that the instrument is mounted on.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_id

    @property
    def target_name(self):
        """
        Returns an target name for uniquely identifying the instrument, but often
        piped into Spice Kernels to acquire Ephemeris data from Spice. Therefore they
        the same ID the Spice expects in bodvrd calls. In this case, vesta images
        have a number in front of them like "4 VESTA" which needs to be simplified
        to "VESTA" for spice. Expects target_name to be defined in the Pds3Label mixin.
        This should be a string containing the name of the target body.

        Returns
        -------
        : str
          target name
        """
        target = super().target_name
        target = target.split(' ')[-1]
        return target

    @property
    def ephemeris_start_time(self):
        """
        Compute the center ephemeris time for a Dawn Frame camera. This is done
        via a spice call but 193 ms needs to be added to
        account for the CCD being discharged or cleared.
        """
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.spacecraft_clock_start_count
            self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, sclock)
            self._ephemeris_start_time += 193.0 / 1000.0
        return self._ephemeris_start_time

    @property
    def usgscsm_distortion_model(self):
        """
        The Dawn framing camera uses a unique radial distortion model so we need
        to overwrite the method packing the distortion model into the ISD.
        Expects odtk to be defined. This should be a list containing the radial
        distortion coefficients

        Returns
        -------
        : dict
          Dictionary containing the distortion model
        """
        return {
            "dawnfc": {
                "coefficients" : self.odtk
                }
            }

    @property
    def odtk(self):
        """
        The coefficients for the distortion model
        Expects ikid to be defined. This should be an integer containing the
        Naif ID code for the instrument.

        Returns
        -------
        : list
          Radial distortion coefficients
        """
        return spice.gdpool('INS{}_RAD_DIST_COEFF'.format(self.ikid),0, 1).tolist()

    # TODO: Update focal2pixel samples and lines to reflect the rectangular
    #       nature of dawn pixels
    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This should be an integer containing the
        Naif ID code for the instrument.

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        # Microns to mm
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 1/pixel_size, 0.0]

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This should be an integer containing the
        Naif ID code for the instrument.

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        # Microns to mm
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 0.0, 1/pixel_size]

    @property
    def sensor_model_version(self):
        """
        Returns instrument model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def detector_center_sample(self):
        """
        Returns center detector sample acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        We have to add 0.5 to the CCD Center because the Dawn IK defines the
        detector pixels as 0.0 being the center of the first pixel so they are
        -0.5 based.

        Returns
        -------
        : float
          center detector sample
        """
        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[0]) + 0.5

    @property
    def detector_center_line(self):
        """
        Returns center detector line acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        We have to add 0.5 to the CCD Center because the Dawn IK defines the
        detector pixels as 0.0 being the center of the first pixel so they are
        -0.5 based.

        Returns
        -------
        : float
          center detector line
        """
        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[1]) + 0.5


class DawnVirIsisNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    @property
    def instrument_id(self):
        """
        Returns the ID of the instrument

        Returns
        -------
        : str
          Name of the instrument
        """
        lookup_table = {'VIR': 'Visual and Infrared Spectrometer'}
        return lookup_table[super().instrument_id]
    
    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label["IsisCube"]["Instrument"]["InstrumentId"]
    
    @property
    def exposure_duration(self):
      """
      The exposure duration of the image, in seconds

      Returns
      -------
      : float
        Exposure duration in seconds
      """
      return self.label["IsisCube"]["Instrument"]["FrameParameter"][0]
    
    @property
    def ikid(self):
        """
        Returns the Naif ID code for the instrument
        Expects the instrument_id to be defined. This must be a string containing
        the short name of the instrument.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        lookup_table = {
          'VIS': -203211,
          "IR": -203213
        }
        return lookup_table[self.label["IsisCube"]["Instrument"]["ChannelId"]]
    
    @property
    def housekeeping_table(self):
        """
        This table named, "VIRHouseKeeping", contains four fields: ScetTimeClock, ShutterStatus,
        MirrorSin, and MirrorCos.  These fields contain the scan line time in SCLK, status of 
        shutter - open, closed (dark), sine and cosine of the scan mirror, respectively.

        Returns
        -------
        : dict
          Dictionary with ScetTimeClock, ShutterStatus, MirrorSin, and MirrorCos
        """
        isis_bytes = read_table_data(self.label['Table'], self._file)

        return parse_table(self.label['Table'], isis_bytes)
    
    @property
    def line_scan_rate(self):

        line_times = []
        for line_midtime in self.line_midtimes:
          line_times.append(line_midtime - (self.exposure_duration / 2.0))

        return [len(line_times)], [line_times], [self.exposure_duration]

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
    
    @property
    def optical_angles(self):
      hk_dict = self.housekeeping_table
      
      lines = []
      for index, mirror_sin in enumerate(hk_dict["MirrorSin"]):
          
          mirror_cos = hk_dict["MirrorCos"][index]

          scan_elec_deg = math.atan(mirror_sin/mirror_cos) * degs_per_rad
          opt_ang = ((scan_elec_deg - 3.7996979) * 0.25/0.257812) / 1000

          # if (hk_dict["ShutterStatus"][index].lower() == "closed"):
            # opt_ang = angFit.Evaluate(a+1, NumericalApproximation::NearestEndpoint)
            
          lines.append(opt_ang)

      return lines
    
    @property
    def line_midtimes(self):

        line_times = []
        scet_times = self.housekeeping_table["ScetTimeClock"]
        for scet in scet_times:
          line_midtime = spice.scs2e(self.spacecraft_id, scet)
          line_times.append(line_midtime)

        return line_times
    
    @property
    def is_calibrated(self):
        return self.label['IsisCube']['Instrument']['ProcessingLevelID'] > 2

    @property 
    def has_articulation_kernel(self):
        regex = re.compile('.*dawn_vir_[0-9]{9}_[0-9]{1}.BC')
        return any([re.match(regex, i) for i in self.kernels])
    
    @property
    def frame_chain(self):
        if not hasattr(self, '_frame_chain'):
            if self.has_articulation_kernel and self.is_calibrated:
                self._frame_chain = super().frame_chain
            else:
                self._frame_chain = super().frame_chain
                vir_rotation = ConstantRotation([1,0,0,0], self.sensor_frame_id, self.sensor_frame_id)
                self._frame_chain.add_edge(rotation = vir_rotation)
                self._frame_chain.compute_time_dependent_rotiations([(1, self.sensor_frame_id)], self.ephemeris_time, 0)

                rot = self._frame_chain[1][self.sensor_frame_id]['rotation']
                quats = np.zeros((len(rot.times), 4))
                matrix = rot._rots.as_matrix()
                avs = []
                for opt_ang in self.optical_angles:
                    for i, matrix in enumerate(rot._rots.as_matrix()):
                      s_matrix = spice.rav2xf(matrix, rot.av[i])
                      xform = spice.eul2xf([0, -opt_ang, 0, 0, 0, 0], 1, 2, 3)
                      xform2 = spice.mxmg(xform, s_matrix)
                      rot_mat, av = spice.xf2rav(xform2)
                      avs.append(av)
                      quat_from_rotation = spice.m2q(rot_mat)
                      quats[i,:3] = quat_from_rotation[1:]
                      quats[i,3] = quat_from_rotation[0]
                rot.quats = quats
                rot.av = avs
        return self._frame_chain
