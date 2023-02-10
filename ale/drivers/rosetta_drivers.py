import spiceypy as spice
from scipy.spatial.transform import Rotation

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import LineScanner
from ale.transformation import ConstantRotation



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
            "VIRTIS" : self.label['IsisCube']['Instrument']['ChannelID']
        }
        return inst_id_lookup[super().instrument_id] 
    
    @property
    def ikid(self):
        """
        Returns the ikid/frame code from the ISIS label. This is attached
        via Rosetta Virtis on ingestion into an ISIS cube
        
        Returns
        -------
        : int
          ikid for Rosetta VIRTIS
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']
    
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
        return spice.scs2e(self.spacecraft_id, self.label['IsisCube']['Instrument']['SpacecraftClockStartCount'])


    @property
    def ephemeris_stop_time(self):
        return spice.scs2e(self.spacecraft_id, self.label['IsisCube']['Instrument']['SpacecraftClockStopCount'])


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

        isCalibrated = self.label['IsisCube']['Instrument']['ProcessingLevelID'] == '3'
        if isCalibrated:
            return super().sensor_frame_id
        else:
             return super().sensor_frame_id - 1

    @property
    def frame_chain(self):
        frame_chain = super().frame_chain
        virtis_quats = Rotation.from_euler(self.virtis_rotation_matrix).as_quat()
        virtis_rotation = ConstantRotation(virtis_quats, self.spacecraft_id * 1000, self.sensor_frame_id)
        self._frame_chain.add_edge(rotation = virtis_rotation)
