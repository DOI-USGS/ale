import spiceypy as spice

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.label_pds3 import Pds3Label
from ale.base.type_distortion import NoDistortion, ChandrayaanMrffrDistortion
from ale.base.type_sensor import LineScanner, Radar
from csv import reader


class Chandrayaan1M3Pds3NaifSpiceDriver(LineScanner, Pds3Label, NaifSpice, NoDistortion, Driver):

    @property
    def ikid(self):
        """
        Returns the Naif ID code for the Moon Mineralogy Mapper

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return -86520

    @property
    def sensor_frame_id(self):
        """
        Returns the Naif ID code for the sensor reference frame

        Returns
        -------
        : int
          Naif ID code for the sensor frame
        """
        return spice.bods2c("CH1")


    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft

        Returns
        -------
        : str
          Full name of the spacecraft
        """
        return self.label['MISSION_ID']


    @property
    def platform_name(self):
        """
        Returns
        -------
        : str
          Name of the platform that the sensor is on
        """
        return self.label['MISSION_NAME']


    @property
    def image_lines(self):
        """
          Returns
        -------
        : int
          Number of lines in the image
        """
        return self.label.get('RDN_FILE').get('RDN_IMAGE').get('LINES')


    @property
    def image_samples(self):
        """
        Returns
        -------
        : int
          Number of samples in the image
        """
        return self.label.get('RDN_FILE').get('RDN_IMAGE').get('LINE_SAMPLES')


    def read_timing_data(self):
        """
        Read data from the timing file.
        Returns
        : list of int
          List of integer line numbers from the timing file
        : list of str
          List of strings describing center UTC Time per line
        """
        if hasattr(self, '_utc_times'):
            return(self._lines, self._utc_times)

        # Read timing file as structured text.  Reads each row as a list.
        with open(self.utc_time_table, 'r') as time_file:
            lists = list(reader(time_file, skipinitialspace=True, delimiter=" "))

        # Transpose such that each column is a list. Unpack and ignore anything that's not lines and times.
        self._lines, self._utc_times, *_ = zip(*lists)


    @property
    def ephemeris_start_time(self):
        """
        Returns
        -------
        : double
          The start time of the image in ephemeris seconds past the J2000 epoch.
        """
        if not hasattr(self, '_ephemeris_start_time'):
            et = spice.utc2et(self.utc_times[0])
            et -= (.5 * self.line_exposure_duration)
            clock_time = spice.sce2s(self.sensor_frame_id, et)
            self._ephemeris_start_time = spice.scs2e(self.sensor_frame_id, clock_time)
        return self._ephemeris_start_time


    @property
    def ephemeris_stop_time(self):
        """
        Returns
        -------
        : double
          The stop time of the image in ephemeris seconds past the J2000 epoch.
        """
        return self.ephemeris_start_time + (self.image_lines * self.exposure_duration)
                        

    @property
    def utc_time_table(self):
        """ 
        Return
        ------
        : str
          The name of the file containing the line timing information
        """
        if not hasattr(self, '_utc_time_table'):
            self._utc_time_table = self.label['UTC_FILE']['^UTC_TIME_TABLE']
        return self._utc_time_table


    @property
    def utc_times(self):
        """ UTC time of the center of each line
        """
        if not hasattr(self, '_utc_times'):
            self.read_timing_data()

        if self._utc_times[0] > self._utc_times[-1]:
            return list(reversed(self._utc_times))
        return self._utc_times
      

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label. For example a return value of 2
        indicates that 2 lines and 2 samples (4 pixels) were summed and divided by 4
        to produce the output pixel value.

        Information found in M3 SIS

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        instrument_mode = self.label['INSTRUMENT_MODE_ID']
        if instrument_mode.upper() == "GLOBAL":
            return 2
        else:
            return 1

                
    @property
    def line_exposure_duration(self):
        """
        Line exposure duration returns the time between the exposures for
        subsequent lines.

        Logic found in ISIS translation file "Chandrayaan1M3Instrument.trn"

        Returns
        -------
        : float
          Returns the line exposure duration in seconds from the PDS3 label.
        """
        instrument_mode = self.label['INSTRUMENT_MODE_ID']
        if instrument_mode.upper() == "GLOBAL":
            return .10176
        elif instrument_mode.upper() == "TARGET":
            return .05088

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          version of the sensor model
        """
        return 1

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
