import spiceypy as spice
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver

class Mariner10IsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    
    @property
    def instrument_id(self):
        """
        Returns the ID of the instrument

        Returns
        -------
        : str
          Name of the instrument
        """
        inst_id_lookup = {
            "M10_VIDICON_A": "M10_SPACECRAFT",
            "M10_VIDICON_B": "M10_SPACECRAFT"
        }
        return inst_id_lookup[super().instrument_id]

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for HiRise in ISIS. This is likely just 1.

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
    
    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return super().instrument_id
    
    @property
    def sensor_frame_id(self):
        """
        Hard coded sensor_frame_id based on the Isis Mariner10 camera model.

        Returns
        -------
        : int
          The sensor frame id
        """
        return -76000
    
    @property
    def ikid(self):
        """
        Returns the ikid/frame code from the ISIS label.
        
        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']
    
    @property
    def ephemeris_start_time(self):
        """
        Returns the start ephemeris time for the image.

        Returns
        -------
        : float
          start time
        """
        return spice.str2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")) - (self.exposure_duration / 2.0)
    
    @property
    def light_time_correction(self):
        """
        Returns the type of light time correction and abberation correction to
        use in NAIF calls.

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'
    