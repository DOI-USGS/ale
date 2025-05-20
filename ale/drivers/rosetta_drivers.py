import re
import spiceypy as spice
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline


from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import read_table_data, parse_table
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import LineScanner
from ale.transformation import ConstantRotation, FrameChain, TimeDependentRotation


class RosettaVirtisIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, RadialDistortion, Driver):
    @property
    def instrument_id(self):
        """
        Returns the instrument id for Rosetta Virtis
        
        Returns
        -------
        : str
          Frame Reference for Rosetta VIRTIS
        """
        inst_id_lookup = {
            "VIRTIS_M_VIS" : "ROS_VIRTIS-M_VIS",
            "VIRTIS_M_IR" : "ROS_VIRTIS-M_IR",
        }
        return inst_id_lookup[self.label['IsisCube']['Instrument']['ChannelID']] 
    

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for Rosetta Virtis in ISIS.
        
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
        

    @property
    def spacecraft_name(self):
        """
        Return the spacecraft name.  The label calls it "ROSETTA-ORBITER", NAIF calls it "ROSETTA"
        """
        return "ROSETTA"

    @property
    def ephemeris_start_time(self):
        """
        Returns the start ephemeris time for the image.

        Returns
        -------
        : float
          start time
        """
        try:
            # first line's middle et - 1/2 exposure duration = cube start time
            return self.hk_ephemeris_time[0] - (self.line_exposure_duration/2)
        except:
            return self.spiceql_call("strSclkToEt", {"frameCode" : self.spacecraft_id, "sclk" : self.label['IsisCube']['Instrument']['SpacecraftClockStartCount'], "mission" : self.spiceql_mission})


    @property
    def ephemeris_stop_time(self):
        """
        Returns the stop ephemeris time for the image.

        Returns
        -------
        : float
          stop time
        """

        try:
            #  last line's middle et + 1/2 exposure duration = cube start time
            return self.hk_ephemeris_time[-1] + (self.line_exposure_duration/2)
        except:
            return self.spiceql_call("strSclkToEt", {"frameCode" : self.spacecraft_id, "sclk" : self.label['IsisCube']['Instrument']['SpacecraftClockStopCount'], "mission" : self.spiceql_mission})

    @property
    def housekeeping(self):
        """
        Read the housekeeping table from the cub and populate data.

        Returns
        -------
        None
        """
        if not hasattr(self, "_housekeeping"):
            degs_per_rad = 57.2957795 
            binary = read_table_data(self.label['Table'], self._file)
            hk_table = parse_table(self.label['Table'], binary)
            data_scet = hk_table['dataSCET']
            shutter_mode_list = hk_table['Data Type__Shutter state']
            mirror_sin_list = hk_table['M_MIRROR_SIN_HK']
            mirror_cos_list = hk_table['M_MIRROR_COS_HK']

            opt_angles = []
            x = np.array([])
            y = np.array([])
            for index, mirror_sin in enumerate(mirror_sin_list):
                shutter_mode = shutter_mode_list[index]
                is_dark = (shutter_mode == 1)   

                mirror_cos = mirror_cos_list[index]

                scan_elec_deg = math.atan(mirror_sin/mirror_cos) * degs_per_rad
                opt_ang = ((scan_elec_deg - 3.7996979) * 0.25/0.257812) / 1000

                if not is_dark:
                    x = np.append(x, index + 1)
                    y = np.append(y, opt_ang)

                if not self.is_calibrated:
                    opt_angles.append(opt_ang)

            cs = CubicSpline(x, y, extrapolate="periodic")

            for i, opt_ang in enumerate(opt_angles):
                shutter_mode = shutter_mode_list[i]
                is_dark = (shutter_mode == 1)

                if (is_dark):
                    if (i == 0):
                        opt_angles[i] = opt_angles[i+1]
                    elif (i == len(opt_angles) - 1):
                        opt_angles[i] = opt_angles[i-1]
                    else:
                        opt_angles[i] = cs(i+1)

            line_mid_times = [self.spiceql_call("scs2e", {"frameCode" : self.spacecraft_id, "sclk" : str(round(i,5)), "mission": self.spiceql_mission} ) for i in data_scet]
            self._hk_ephemeris_time = line_mid_times
            self._optical_angle = opt_angles
            self._housekeeping= True

    @property
    def hk_ephemeris_time(self):
        """
        Ephemeris times from the housekeeping table.

        Returns
        -------
        : list
          Ephemeris times from the housekeeping table
        """
        if not hasattr(self, "_hk_ephemeris_time"):
            self.housekeeping
        return self._hk_ephemeris_time

    @property
    def optical_angle(self):
        """
        Return optical angles from the housekeeping table.

        Returns
        -------
        : list
          Optical angles from the housekeeping table
        """
        if not hasattr(self, "_optical_angle"):
            self.housekeeping
        return self._optical_angle


    @property
    def line_exposure_duration(self):
        """
        Returns the exposure duration for each line.

        Returns
        -------
        : double
          Exposure duration for a line
        """
        return self.label['IsisCube']['Instrument']['FrameParameter'][0] * 0.001


    @property
    def line_summing(self):
        """
         Returns
         -------
         : int
           Line summing
        """
        return self.label['IsisCube']['Instrument']['FrameParameter'][1]

    @property
    def sample_summing(self):
        """
         Returns
         -------
         : int
           Sample summing
        """
        return self.label['IsisCube']['Instrument']['FrameParameter'][1]

    @property
    def scan_rate(self):
        """
        Returns
        : Double
          Scan rate
        """
        return self.label['IsisCube']['Instrument']['FrameParameter'][2]

    @property
    def odtk(self):
        """
        The coefficients for the distortion model
        Defined in the IAK

        Returns
        -------
        : list
          Radial distortion coefficients
        """
        return [0.0, .038, 0.38]


    @property
    def sensor_frame_id(self):
        """
        Returns the sensor frame id.  Depends on the instrument that was used to
        capture the image. 

        Returns
        -------
        : int
          Sensor frame id
        """

        if self.has_articulation_kernel:
            return super().sensor_frame_id
        else:
             return super().sensor_frame_id - 1


    @property
    def is_calibrated(self):
        """
        Determine if this image is calibrated.

        Returns
        -------
        : bool
          True if calibrated else False
        """
        return self.label['IsisCube']['Instrument']['ProcessingLevelID'] == '3'

    @property 
    def has_articulation_kernel(self):
        """
        Determine if this image has an associated articulation kernel

        Returns
        -------
        : bool
          True if the image has an articulation kernel else False
        """
        regex = re.compile('.*ROS_VIRTIS_M_[0-9]{4}_[0-9]{4}_V[0-9].BC')
        return any([re.match(regex, i) for i in self.kernels])


    @property
    def frame_chain(self):
        """
        Construct the frame chain.  Use super to construct base chain and add edges via inst_pointing_rotation

        Returns
        -------
        : FrameChain
        """
        frame_chain = super().frame_chain
        frame_chain.add_edge(rotation=self.inst_pointing_rotation)
        return frame_chain

    @property
    def inst_pointing_rotation(self):
        """
        Returns a time dependent instrument pointing rotation for virtis frames
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

          opt_angle = self.optical_angle[i]

          xform = spice.eul2xf([0, -opt_angle, 0, 0, 0, 0], 1, 2, 3)
          xform2 = spice.mxmg(xform, state_matrix)

          rot_mat, av = spice.xf2rav(xform2)
          avs.append(av)

          quat_from_rotation = spice.m2q(rot_mat)
          time_dep_quats[i,:3] = -quat_from_rotation[1:]
          time_dep_quats[i, 3] = -quat_from_rotation[0]

        time_dep_rot = TimeDependentRotation(time_dep_quats, self.hk_ephemeris_time, 1, self.sensor_frame_id, av=avs)

        return time_dep_rot