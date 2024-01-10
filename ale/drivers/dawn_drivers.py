import re
import spiceypy as spice
import os
import math
import numpy as np

from scipy.interpolate import CubicSpline

import ale
from ale.base import Driver
from ale.base.label_isis import IsisLabel
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_distortion import NoDistortion
from ale.base.type_sensor import Framer, LineScanner
from ale.base.data_isis import read_table_data
from ale.base.data_isis import parse_table
from ale.transformation import TimeDependentRotation
from ale.transformation import ConstantRotation

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

class DawnFcIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading Dawn ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument,
        but often also used to be piped into Spice Kernels to acquire
        IKIDS. Therefor they are the same ID that Spice expects in bods2c
        calls. Expect instrument_id to be defined in the IsisLabel mixin.
        This should be a string of the form

        Returns
        -------
        : str
          instrument id
        """
        if not hasattr(self, "_instrument_id"):
          instrument_id = super().instrument_id
          filter_number = self.filter_number
          self._instrument_id = "{}_FILTER_{}".format(ID_LOOKUP[instrument_id], filter_number)

        return self._instrument_id
    
    @property
    def filter_number(self):
        """
        Returns the instrument filter number from the ISIS bandbin group.
        This filter number is used in the instrument id to identify
        which filter was used when aquiring the data.

        Returns
        -------
         : int
           The filter number from the instrument
        """
        return self.label["IsisCube"]["BandBin"]["FilterNumber"]

    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft

        Returns
        -------
        : str
          spacecraft name
        """
        return self.label["IsisCube"]["Instrument"]["SpacecraftName"]

    @property
    def sensor_name(self):
        """
        Returns the sensor name

        Returns
        -------
        : str
          sensor name
        """
        return self.instrument_id

    @property
    def sensor_model_version(self):
        """
        Returns ISIS sensor model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label["IsisCube"]["Kernels"]["NaifFrameCode"]

    def filter_name(self):
        """
        Returns the filter used to identify the image

        Returns
        -------
        : str
          filter name
        """
        return self.label["IsisCube"]["BandBin"]["FilterName"]

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

    @property
    def ephemeris_start_time(self):
        """
        Compute the starting ephemeris time for a Dawn Frame camera. This is done
        via a spice call but 193 ms needs to be added to
        account for the CCD being discharged or cleared.

        Returns
        -------
        : float
          ephemeris start time
        """
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.spacecraft_clock_start_count
            self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, sclock)
            self._ephemeris_start_time += 193.0 / 1000.0
        return self._ephemeris_start_time

    @property
    def exposure_duration_ms(self):
        """
        Return the exposure duration in ms for a Dawn Frame camera.

        Returns
        -------
        : float
          exposure duration
        """
        return self.exposure_duration / 1000

    @property
    def ephemeris_stop_time(self):
        """
        Compute the ephemeris stop time for a Dawn Frame camera

        Returns
        -------
        : float
          ephemeris stop time
        """
        return self.ephemeris_start_time + self.exposure_duration_ms

    @property
    def ephemeris_center_time(self):
        """
        Compute the center ephemeris time for a Dawn Frame camera.

        Returns
        -------
        : float
          center ephemeris time
        """
        return self.ephemeris_start_time + (self.exposure_duration_ms / 2.0)
    

class DawnVirIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
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
    def line_exposure_duration(self):
      """
      The exposure duration of the image, in seconds

      Returns
      -------
      : float
        Exposure duration in seconds
      """
      return self.label["IsisCube"]["Instrument"]["FrameParameter"][0]
    
    @property
    def focal_length(self):
      return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])
    
    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return super().detector_center_sample - 0.5

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
    def sensor_frame_id(self):
        """
        Returns the FRAME_DAWN_VIR_{VIS/IR}_ZERO Naif ID code if there are no associated articulation kernels. 
        Otherwise the original Naif ID code is returned.
        Returns
        -------
        : int
          Naif ID code for the sensor frame
        """
        if self.has_articulation_kernel:
          lookup_table = {
            'VIS': -203211,
            "IR": -203213
         }
        else:
          lookup_table = {
            'VIS': -203221,
            "IR": -203223
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
        """
        Returns
        -------
        : list
          Start lines
        : list
          Line times
        : list
          Exposure durations
        """
        line_times = []
        start_lines = []
        exposure_durations = []

        line_no = 0.5

        for line_midtime in self.hk_ephemeris_time:
          if not self.is_calibrated:
            line_times.append(line_midtime - (self.line_exposure_duration / 2.0) - self.center_ephemeris_time)
            start_lines.append(line_no)
            exposure_durations.append(self.label["IsisCube"]["Instrument"]["FrameParameter"][2])
            line_no += 1

        line_times.append(line_times[-1] + self.label["IsisCube"]["Instrument"]["FrameParameter"][2])
        start_lines.append(line_no)
        exposure_durations.append(self.label["IsisCube"]["Instrument"]["FrameParameter"][2])

        return start_lines, line_times, exposure_durations

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
        
        opt_angles = []
        x = np.array([])
        y = np.array([])
        for index, mirror_sin in enumerate(hk_dict["MirrorSin"]):
            shutter_status = hk_dict["ShutterStatus"][index].lower().replace(" ", "")
            is_dark = (shutter_status == "closed")   

            mirror_cos = hk_dict["MirrorCos"][index]

            scan_elec_deg = math.atan(mirror_sin/mirror_cos) * degs_per_rad
            opt_ang = ((scan_elec_deg - 3.7996979) * 0.25/0.257812) / 1000

            if not is_dark:
                x = np.append(x, index + 1)
                y = np.append(y, opt_ang)

            if not self.is_calibrated:
                opt_angles.append(opt_ang)

        cs = CubicSpline(x, y, extrapolate="periodic")

        for i, opt_ang in enumerate(opt_angles):
          shutter_status = hk_dict["ShutterStatus"][i].lower().replace(" ", "")
          is_dark = (shutter_status == "closed")

          if (is_dark):
            if (i == 0):
              opt_angles[i] = opt_angles[i+1]
            elif (i == len(opt_angles) - 1):
              opt_angles[i] = opt_angles[i-1]
            else:
              opt_angles[i] = cs(i+1)

        return opt_angles
    
    @property
    def hk_ephemeris_time(self):

        line_times = []
        scet_times = self.housekeeping_table["ScetTimeClock"]
        for scet in scet_times:
          line_midtime = spice.scs2e(self.spacecraft_id, scet)
          line_times.append(line_midtime)

        return line_times
    
    @property
    def ephemeris_start_time(self):
        """
        Returns the starting ephemeris time of the image using the first et time from
        the housekeeping table minus the line exposure duration / 2. 

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        return self.hk_ephemeris_time[0] - (self.line_exposure_duration / 2)


    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris stop time of the image using the last et time from
        the housekeeping table plus the line exposure duration / 2. 

        Returns
        -------
        : double
          Ephemeris stop time of the image
        """
        return self.hk_ephemeris_time[-1] + (self.line_exposure_duration / 2)

    @property
    def center_ephemeris_time(self):
      return self.hk_ephemeris_time[int(len(self.hk_ephemeris_time)/2)]
    
    @property
    def is_calibrated(self):
        """
        Checks if image is calibrated.

        Returns
        -------
        : bool
        """
        return self.label['IsisCube']['Archive']['ProcessingLevelId'] > 2

    @property 
    def has_articulation_kernel(self):
        """
        Checks if articulation kernel exists in pool of kernels.

        Returns
        -------
        : bool
        """
        try:
          regex = re.compile('.*dawn_vir_[0-9]{9}_[0-9]{1}.BC')
          return any([re.match(regex, i) for i in self.kernels])
        except ValueError:
          return False

    @property
    def frame_chain(self):
      frame_chain = super().frame_chain
      frame_chain.add_edge(rotation=self.inst_pointing_rotation)
      return frame_chain
    
    @property
    def inst_pointing_rotation(self):
        """
        Returns a time dependent instrument pointing rotation for -203221 and -203223 sensor frame ids.

        Returns
        -------
        : TimeDependentRotation
          Instrument pointing rotation
        """
        time_dep_quats = np.zeros((len(self.hk_ephemeris_time), 4))
        avs = []

        for i, time in enumerate(self.hk_ephemeris_time):
          try:
            state_matrix = spice.sxform("J2000", spice.frmnam(self.sensor_frame_id), time)
          except:
            rotation_matrix = spice.pxform("J2000", spice.frmnam(self.sensor_frame_id), time)
            state_matrix = spice.rav2xf(rotation_matrix, [0, 0, 0])

          opt_angle = self.optical_angles[i]
          
          xform = spice.eul2xf([0, -opt_angle, 0, 0, 0, 0], 1, 2, 3)
          xform2 = spice.mxmg(xform, state_matrix)

          rot_mat, av = spice.xf2rav(xform2)
          avs.append(av)

          quat_from_rotation = spice.m2q(rot_mat)
          time_dep_quats[i,:3] = quat_from_rotation[1:]
          time_dep_quats[i, 3] = quat_from_rotation[0]

        time_dep_rot = TimeDependentRotation(time_dep_quats, self.hk_ephemeris_time, 1, self.sensor_frame_id, av=avs)

        return time_dep_rot

    