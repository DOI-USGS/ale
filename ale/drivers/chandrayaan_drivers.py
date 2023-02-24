import spiceypy as spice

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import NoDistortion, ChandrayaanMrffrDistortion
from ale.base.type_sensor import LineScanner, Radar

class Chandrayaan1M3IsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    
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
            "M3" : "CHANDRAYAAN-1_M3"
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
        return spice.namfrm(self.instrument_id)
    
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

class Chandrayaan1MRFFRIsisLabelNaifSpiceDriver(Radar, IsisLabel, NaifSpice, ChandrayaanMrffrDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns the instrument id for chandrayaan moon mineralogy mapper
        
        Returns
        -------
        : str
          Frame Reference for chandrayaan Mini RF
        """
        inst_id_lookup = {
            "MRFFR" : "CHANDRAYAAN-1_MRFFR"
        }
        return inst_id_lookup[super().instrument_id] 

    @property
    def spacecraft_name(self):
        return 'CHANDRAYAAN-1'

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
    def ephemeris_start_time(self):
        """
        Returns the start ephemeris time for the image.

        Returns
        -------
        : float
          start time
        """
        return spice.str2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))

    @property
    def ephemeris_stop_time(self):
        """
        Returns the stop ephemeris time for the image.

        Returns
        -------
        : float
          stop time
        """
        return spice.str2et(self.utc_stop_time.strftime("%Y-%m-%d %H:%M:%S.%f"))


    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original IMG file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return -86001

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for Chandrayaan-1 MRFFR in ISIS. This is likely just 1

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def wavelength(self):
        """
        Returns the wavelength in meters used for image acquisition.

        Returns
        -------
        : double
          Wavelength in meters used to create an image
        """

        # Get float value of frequency in GHz
        frequency = self.label['IsisCube']['Instrument']['Frequency'].value
        wavelength = spice.clight() / frequency / 1000.0
        return wavelength


        # Default line_exposure_duration assumes that time is given in milliseconds and coverts
    # in this case, the time is already given in seconds.
    @property
    def line_exposure_duration(self):
        """
        Line exposure duration in seconds. The sum of the burst and the delay for the return.

        Returns
        -------
        : double
          scaled pixel width
        """
        return self.label['IsisCube']['Instrument']['LineExposureDuration']

    @property
    def ground_range_resolution(self):
        return self.label['IsisCube']['Instrument']['RangeResolution']

    @property
    def range_conversion_coefficients(self):
        """
        Range conversion coefficients

        Returns
        -------
        : List
          range conversion coefficients
        """

        range_coefficients_orig = self.label['IsisCube']['Instrument']['RangeCoefficientSet']

        # The first elt of each list is time, which we handle separately in range_conversion_time
        range_coefficients = [elt[1:] for elt in range_coefficients_orig]
        return range_coefficients

    @property
    def range_conversion_times(self):
        """
        Times, in et, associated with range conversion coefficients

        Returns
        -------
        : List
          times for range conversion coefficients
        """
        range_coefficients_utc = self.label['IsisCube']['Instrument']['RangeCoefficientSet']
        range_coefficients_et = [spice.str2et(elt[0]) for elt in range_coefficients_utc]
        return range_coefficients_et


    @property
    def scaled_pixel_width(self):
        """
        Returns the scaled pixel width

        Returns
        -------
        : double
          scaled pixel width
        """
        return self.label['IsisCube']['Instrument']['ScaledPixelWidth']

    @property
    def scaled_pixel_height(self):
        """
        Returns the scaled pixel height

        Returns
        -------
        : double
          scaled pixel height
        """
        return self.label['IsisCube']['Instrument']['ScaledPixelHeight']


    @property
    def look_direction(self):
        """
        Direction of the look (left or right)

        Returns
        -------
        : string
          left or right
        """
        return self.label['IsisCube']['Instrument']['LookDirection'].lower()

    @property
    def naif_keywords(self):
        """
        Adds base NH instrument distortion, which is shared among all instruments on NH.
        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        transx = [-1* self.scaled_pixel_height, self.scaled_pixel_height, 0.0]
        transy = [0,0,0]
        transs = [1.0, 1.0 / self.scaled_pixel_height, 0.0]
        transl = [0.0, 0.0, 0.0]
        return {**super().naif_keywords,
                f"INS{self.ikid}_TRANSX": transx,
                f"INS{self.ikid}_TRANSY": transy,
                f"INS{self.ikid}_ITRANSS": transs,
                f"INS{self.ikid}_ITRANSL": transl}
