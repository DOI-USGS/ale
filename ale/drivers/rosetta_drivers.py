import re
import spiceypy as spice
import math
import numpy as np
from scipy.spatial.transform import Rotation

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import read_table_data, parse_table
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import LineScanner
from ale.transformation import ConstantRotation, FrameChain



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
        fail
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
        try:
            # first line's middle et - 1/2 exposure duration = cube start time
            return self.ephemeris_time[0] - (self.line_exposure_duration/2)
        except:
            return spice.scs2e(self.spacecraft_id, self.label['IsisCube']['Instrument']['SpacecraftClockStartCount'])


    @property
    def ephemeris_stop_time(self):
        try:
            #  last line's middle et + 1/2 exposure duration = cube start time
            return self.ephemeris_time[-1] + (self.line_exposure_duration/2)
        except:
            return spice.scs2e(self.spacecraft_id, self.label['IsisCube']['Instrument']['SpacecraftClockStopCount'])

    @property
    def scet(self):
        if not hasattr(self, "_scet"):
            binary = read_table_data(self.label['Table'], self._file)
            data_scet = parse_table(self.label['Table'], binary)['dataSCET']
            #self._scet = [(i, spice.scs2e(self.spacecraft_id, str(j - self.line_exposure_duration)), self.line_exposure_duration) for i,j in enumerate(data_scet)]
            self._scet = data_scet
        return self._scet

    @property
    def housekeeping(self):
        # @TODO unsure how to use shutter mode information for ale driver.
        if not hasattr(self, "_housekeeping"):
            binary = read_table_data(self.label['Table'], self._file)
            self._data_scet = parse_table(self.label['Table'], binary)['dataSCET']
            #shutter_mode = parse_table(self.label['Table'], binary)['Data Type__Shutter state']
            mirror_sin = parse_table(self.label['Table'], binary)['M_MIRROR_SIN_HK']
            mirror_cos = parse_table(self.label['Table'], binary)['M_MIRROR_COS_HK']
            scan_elec_deg = [math.atan(i/j) * 57.2957795 for i,j in zip(mirror_sin, mirror_cos)] # 57.2957795 = deg per rad
            opt_ang = [((i - 3.7996979) * 0.25/0.257812)/1000 for i in scan_elec_deg] # Magic numbers from isis RosettaVirtisCamera::readHouseKeeping

            line_mid_times = [spice.scs2e(self.spacecraft_id, str(int(i*(10**5))/(10**5))) for i in self._data_scet]
            #print(line_mid_time)
            # redo with list comprehensions?  Needs these calculations per record
            # zip all of these into a mirrordata list?
            #print(list(zip(line_mid_time, mirror_sin, mirror_cos, opt_ang, shutter_mode)))
            """
            self._housekeeping = [{'scanline_mid_et': i, 'mirror_sin': j, 'mirror_cos': k,
                                'optical_angle': l, 'dark_current': m} for i,j,k,l, m in zip(line_mid_time, mirror_sin, mirror_cos, opt_ang, shutter_mode)]
            """
            self._ephemeris_time = line_mid_times
            self._opt_ang = opt_ang
            self._housekeeping= True

    @property
    def ephemeris_time(self):
        if not hasattr(self, "_ephemeris_time"):
            self.housekeeping
        return self._ephemeris_time

    @property
    def opt_ang(self):
        if not hasattr(self, "_opt_ang"):
            self.housekeeping
        return self._opt_ang


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
        return self.label['IsisCube']['Instrument']['ProcessingLevelID'] == '3'

    @property 
    def has_articulation_kernel(self):
        regex = re.compile('.*ROS_VIRTIS_M_[0-9]{4}_[0-9]{4}_V[0-9].BC')
        return any([re.match(regex, i) for i in self.kernels])


    @property
    def frame_chain(self):
        if not hasattr(self, '_frame_chain'):
            if self.has_articulation_kernel and self.is_calibrated:
                self._frame_chain = super().frame_chain
            else:
                self._frame_chain = super().frame_chain
                virtis_rotation = ConstantRotation([1,0,0,0], self.sensor_frame_id, self.sensor_frame_id)
                self._frame_chain.add_edge(rotation = virtis_rotation)
                self._frame_chain.compute_time_dependent_rotiations([(1, self.sensor_frame_id)], self.ephemeris_time, 0)

                rot = self._frame_chain[1][self.sensor_frame_id]['rotation']
                quats = np.zeros((len(rot.times), 4))
                avs = []
                for i, matrix in enumerate(rot._rots.as_matrix()):
                    s_matrix = spice.rav2xf(matrix, rot.av[i])
                    opt_ang = self.opt_ang[i]
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
