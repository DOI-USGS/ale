import pvl
import spiceypy as spice

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice, read_table_data, parse_table
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import NoDistortion
from ale.base.type_sensor import Framer, LineScanner

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
        return self.spiceql_call("utcToEt", {"utc" : self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")})

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
        return float(self.naif_keywords['INS{}_BORESIGHT'.format(self.ikid)][0])

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


class ApolloPanIsisLabelIsisSpiceDriver(LineScanner, IsisLabel, IsisSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        """
        The name of the instrument.

        Returns
        -------
        str
          The short text name for the instrument
        """
        id_lookup = {
            "APOLLO_PAN": "APOLLO PANORAMIC CAMERA"
        }

        return id_lookup[super().instrument_id]


    @property
    def instrument_name(self):
        """
        The name of the instrument.

        Returns
        -------
        str
          The short text name for the instrument
        """
        return "APOLLO PANORAMIC CAMERA"

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for ApolloPanCamera in ISIS.
        
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1


    @property
    def focal2pixel_lines(self):
        """
        The line component of the affine transformation
        from focal plane coordinates to centered ccd pixels.

        This information is not contained in the label, so it is
        hard-coded to match apollo kernels.

        Returns
        -------
        list :
            The coefficients of the affine transformation
            formatted as constant, x, y
        """
        return (0.0, 0.0, 200.0)

    @property
    def focal2pixel_samples(self):
        """
        The sample component of the affine transformation
        from focal plane coordinates to centered ccd pixels

        This information is not contained in the label, so it is
        hard-coded to match apollo kernels.

        Returns
        -------
        list :
            The coefficients of the affine transformation
            formatted as constant, x, y
        """
        return (0.0, 200.0, 0.0)

    @property
    def pixel2focal_x(self):
        """
        Returns detector to focal plane x.

        This information is not contained in the label, so it is
        hard-coded to match apollo kernels.

        Returns
        -------
        : list<double>
          detector to focal plane x
        """
        return (0.0, 0.005, 0.0)


    @property
    def pixel2focal_y(self):
        """
        Returns detector to focal plane y.

        This information is not contained in the label, so it is
        hard-coded to match apollo kernels.

        Returns
        -------
        : list<double>
        detector to focal plane y
        """
        return (0.0, 0.0, 0.0)


    @property
    def focal_length(self):
        """
        The focal length of the instrument
        Hard-coded to return the same value as Isis::ApolloPanoramicCamera.cpp

        Returns
        -------
        float :
            The focal length in millimeters
        """
        return 610.0


    @property
    def ephemeris_start_time(self):
        """
        The image start time in ephemeris time
        The only time information written to the label by apollopaninit is UTC time,
        so this pulls from tables.
        
        Returns
        -------
        float :
            The image start ephemeris time
        """
        isis_bytes = read_table_data(self.label['Table'], self._file)
        return parse_table(self.label['Table'], isis_bytes)['ET'][0]


    @property
    def target_body_radii(self):
        
        """
        The triaxial radii of the target body
        This information is not added to the label by apollopaninit, so it
        is pulled from kernels.
        
        Returns
        -------
        list :
            The body radii in kilometers. For most bodies,
            this is formatted as semimajor, semimajor,
            semiminor
        """
        return (1737.4, 1737.4, 1737.4)


    @property
    def detector_center_line(self):
        """
        The center line of the CCD in detector pixels
        This information is not recorded in the label by apollopaninit, so this is
        hard-coded to match the apollo kernels.

        Returns
        -------
        list :
            The center line of the CCD
        """
        if self.spacecraft_name == "APOLLO16":
            return 11503.5
        else:
            return  11450.5 


    @property
    def detector_center_sample(self):
        """
        The center sample of the CCD in detector pixels
        This information is not recorded in the label by apollopaninit, so this is
        hard-coded to match the apollo kernels.

        Returns
        -------
        list :
            The center sample of the CCD
        """
        if self.spacecraft_name == "APOLLO16":
            return 115537.5
        else:
            return 11450.5
        

    @property
    def naif_keywords(self):
        """
        Apollopaninit doesn't create naif keywords section, so populate it manually here.
        Only includes the NAIF keywords that are necessary for the ALE formatter.
        -------
        : dict
          Dictionary of NAIF keywords that are normally attached to the label
        """
        return {"BODY301_RADII": self.target_body_radii,
                "BODY_FRAME_CODE": self.target_frame_id,
                f"INS{self.ikid}_CONSTANT_TIME_OFFSET": 0,
                f"INS{self.ikid}_ADDITIONAL_PREROLL": 0,
                f"INS{self.ikid}_ADDITIVE_LINE_ERROR": 0,
                f"INS{self.ikid}_MULTIPLI_LINE_ERROR": 0,
                f"INS{self.ikid}_TRANSX": self.pixel2focal_x,
                f"INS{self.ikid}_TRANSY": self.pixel2focal_y,
                f"INS{self.ikid}_ITRANSS": self.focal2pixel_samples,
                f"INS{self.ikid}_ITRANSL": self.focal2pixel_lines,
                "BODY_CODE": 301}
