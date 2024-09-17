import spiceypy as spice
import pvl

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import NoDistortion
from ale.base.type_sensor import Framer

class ApolloMetricIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns the instrument id for chandrayaan moon mineralogy mapper
        
        Returns
        -------
        : str
          Frame Reference for chandrayaan moon mineralogy mapper
        """
        inst_id_lookup = {
            "METRIC" : "APOLLO_METRIC"
        }
        return inst_id_lookup[super().instrument_id] 
    
    @property
    def ikid(self):
        """
        Returns the ikid/frame code from the ISIS label. This is attached
        via chan1m3 on ingestion into an ISIS cube
        
        Returns
        -------
        : int
          ikid for chandrayaan moon mineralogy mapper
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']
    
    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for Chandrayaan1M3 in ISIS. This is likely just 1
        
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def sensor_name(self):
        """
        Returns
        -------
        : String
          The name of the sensor
        """
        return self.instrument_id

    @property
    def exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        exposure_duration = self.label['IsisCube']['Code']['ExposureDuration']
        # Check for units on the PVL keyword
        if isinstance(exposure_duration, pvl.collections.Quantity):
            units = exposure_duration.units
            if "ms" in units.lower() or 'milliseconds' in units.lower():
                exposure_duration = exposure_duration.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                exposure_duration = exposure_duration.value
        else:
            # if no units are available, assume the exposure duration is given in milliseconds
            exposure_duration = exposure_duration * 0.001
        return exposure_duration

    @property
    def ephemeris_start_time(self):
        """
        The spacecraft clock start count, frequently used to determine the start time
        of the image.

        Returns
        -------
        : str
          Spacecraft clock start count
        """
        return spice.str2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))

    @property
    def ephemeris_stop_time(self):
        """
        The spacecraft clock start count, frequently used to determine the start time
        of the image.

        Returns
        -------
        : str
          Spacecraft clock start count
        """
        return self.ephemeris_start_time + self.exposure_duration

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
