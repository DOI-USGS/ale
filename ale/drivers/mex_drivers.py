import os
from glob import glob

import numpy as np

import pvl
import struct
import spiceypy as spice
import warnings

from ale.base import Driver
from ale.base.data_isis import read_table_data
from ale.base.data_isis import parse_table
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import LineScanner
from ale.base.type_sensor import Framer
from ale.base.type_distortion import RadialDistortion

FILTER_SPECIFIC_LOOKUP = {
    # This table contains the filter specific information from the ISIS iak kernel. The format is as follows:
    #
    # fikid: [focal_length, ITRANSX, ITRANSY, ITRANSS, ITRANSL]
    -41211: [174.80,
            [-0.026155499841886, -0.006999999957684, 0.0000007696901985785],
            [-59.9926971240526, 0.0000007696901985785, 0.006999999957684],
            [-2.7941368739538, -142.857141993552, 0.015707963236297],
            [8570.3856624766, 0.015707963236297, 142.857141993552]],
    -41212: [174.61,
            [-0.049776299960244, -0.006999999994409, -0.0000002797762790202],
            [-49.7678019894611, -0.0000002797762790202, 0.006999999994409],
            [-7.39506020202098, -142.85714274304, -0.005709719980004],
            [7109.68570860705, -0.005709719980004, 142.85714274304]],
    -41213: [174.74,
            [-0.022301998711020, -0.006999999595423, -0.000002379930922169],
            [-39.5483075824599, -0.000002379930922169, 0.006999999595423],
            [-5.10686167529011, -142.857134600479, -0.048570018819771],
            [5649.75681632013, -0.048570018819771, 142.857134600479]],
    -41214: [175.01,
            [0.066552498797889, -0.006999999873562, 0.000001330464480785],
            [-10.2417126493911, 0.000001330464480785, 0.006999999873562],
            [9.78558608311961, -142.85714027677, 0.027152336342545],
            [1463.0999735726, 0.027152336342545, 142.85714027677]],
    -41215: [175.01,
            [0.033863199938965, -0.006999999987383, -0.0000004202752836277],
            [-0.005497966876288, -0.0000004202752836277, 0.006999999987383],
            [4.83755282624351, -142.857142599663, -0.008577046604648],
            [0.785714284298145, -0.008577046604648, 142.857142599663]],
    -41216: [175.23,
            [0.044402379964146, -0.006999996841365, 0.000006649877982807],
            [10.212957818494, 0.000006649877982807, 0.006999996841365],
            [4.95717543186894, -142.857078395208, 0.135711795567498],
            [-1458.99934165026, 0.135711795567498, 142.857078395208]],
    -41217: [174.80,
            [0.032494699283744, -0.006999999845704, 0.000001469741752306],
            [39.5530931773118, 0.000001469741752306, 0.006999999845704],
            [3.45571545912004, -142.857139708249, 0.029994729638890],
            [-5650.44273259436, 0.029994729638890, 142.857139708249]],
    -41218: [174.82,
            [0.016461898406507, -0.006999999322408, 0.000003079982431615],
            [49.7917927568053, 0.000003079982431615, 0.006999999322408],
            [-0.778052433438109, -142.857129028729, 0.062856784318668],
            [-7113.11359717265, 0.062856784318668, 142.857129028729]],
    -41219: [174.87,
            [0.024021897233075, -0.006999999193716, 0.000003359758681093],
            [59.9868884703161, 0.000003359758681093, 0.006999999193716],
            [-0.681392000547864, -142.857126402363, 0.068566503695773],
            [-8569.5561557859, 0.068566503695773, 142.857126402363]],
}

class MexHrscPds3NaifSpiceDriver(LineScanner, Pds3Label, NaifSpice, RadialDistortion, Driver):
    """
    Driver for a PDS3 Mars Express (Mex) High Resolution Stereo Camera (HRSC) images.

    NOTES
    -----

    * HRSC has 9 different filters. Each has it's own instrument id, as well as
      the main/"HEAD" camera composing those filters. There is also another
      "SRC" instrument, making-up a total of 11 distinct sensors. It is very
      important to understand which code is needed when/where.

    * HRSC is a variable line scanner, and so does not maintain one exposure
      duration, but rather differing exposure durations per line. This
      information is stored within the individual records in the image data
      itself, with the the first 8 bytes making up the double presicion
      ephemeris time that the line exposure was started, and the next 4 bytes
      making up the float containing that line's exposure duration.

    """

    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients. There is only one coefficient for LROC NAC l/r
        """
        return [0.0, 0.0, 0.0]


    @property
    def ikid(self):
        """
        Returns the Naif ID code for the HRSC head instrument

        This would be the Naif ID code for the base (or "head") instrument.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return spice.bods2c("MEX_HRSC_HEAD")


    @property
    def fikid(self):
        """
        Naif ID code of the filter dependent instrument codes.

        Expects filter_number to be defined. This should be an integer containing
        the filter number from the pds3 label.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        return spice.bods2c(self.instrument_id)


    # TODO Since HRSC has different frames based on filters, need to check that
    # this is returning the value needed for all calculations from the base
    # class and therefor does not need to be reimplemented.
    # @property
    # def sensor_frame_id(self):
    #     """
    #     Returns the Naif ID code for the sensor reference frame
    #
    #
    #     This is the frame of the HRSC instrument itself, and is not dependant on filter.
    #
    #     Returns
    #     -------
    #     : int
    #       Naif ID code for the sensor frame
    #     """
    #     return -41210


    @property
    def instrument_id(self):
        """
        Returns the short name of the instrument

        MEX HRSC has nine different filters each with their own name.

         Returns
        -------
        : str
          Short name of the instrument
        """
        if(super().instrument_id != "HRSC"):
            raise Exception ("Instrument ID is wrong.")
        return self.label['DETECTOR_ID']


    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. MEX HRSC img PDS3 labels do not the have SPACECRAFT_NAME
        keyword, so we override it here to use the label_pds3 property for
        instrument_host_id

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_id


    @property
    def focal_length(self):
        """
        Returns the focal length of the filter-specific sensor

        Expects fikid to be defined. This must be the integer Naif id code of
        the filter-specific instrument.

        NOTE: These values are pulled from ISIS iak kernels.

        Returns
        -------
        : float
          focal length
        """
        return FILTER_SPECIFIC_LOOKUP[self.fikid][0]


    @property
    def focal2pixel_lines(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of
        the filter-specific instrument.

        NOTE: These values are pulled from ISIS iak kernels.

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        return FILTER_SPECIFIC_LOOKUP[self.fikid][4]


    @property
    def focal2pixel_samples(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of
        the filter-specific instrument.

        NOTE: These values are pulled from ISIS iak kernels.

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        return FILTER_SPECIFIC_LOOKUP[self.fikid][3]


    @property
    def pixel2focal_x(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of
        the filter-specific instrument.

        NOTE: These values are pulled from ISIS iak kernels.

        Returns
        -------
        : list<double>
        detector to focal plane x
        """
        return FILTER_SPECIFIC_LOOKUP[self.fikid][1]


    @property
    def pixel2focal_y(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of
        the filter-specific instrument.

        NOTE: These values are pulled from ISIS iak kernels.

        Returns
        -------
        : list<double>
        detector to focal plane y
        """
        return FILTER_SPECIFIC_LOOKUP[self.fikid][2]


    @property
    def detector_center_line(self):
        """
        Returns the center detector line.

        For HRSC, we are dealing with a single line, so center line will be 0.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0.0


    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample.

        For HRSC, center sample is consistent regardless of filter. This is
        different from ISIS's center sample because ISIS line scan sensors use
        0.5 based detector samples.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 2592.0


    @property
    def line_scan_rate(self):
        """
        Returns a 2D array of line scan rates.

        For HRSC, the ephemeris times and exposure durations are
        stored in the image data.

        In the image, every line has an entry. This method goes through
        and removes consecutive lines with the same exposure duration.
        There are also potentially missing lines in the image which this
        method accounts for.

        Returns
        -------
        : list
          Line scan rates
        """
        relative_times = [time - self.center_ephemeris_time for time in self.binary_ephemeris_times]
        start_lines = [self.binary_lines[0]]
        start_times = [relative_times[0]]
        exposure_durations = [self.binary_exposure_durations[0]]
        for line, start_time, exposure_duration in zip(self.binary_lines, relative_times, self.binary_exposure_durations):
            # Check for lines missing from the PDS image
            #
            # If more exposures fit into the time since the last entry than
            # there are lines since the last entry, then there are missing lines.
            #
            # If line are missing, add an extra entry for the line immediately
            # following them.
            skipped_lines = int( (start_time - start_times[-1]) / exposure_durations[-1] - (line - start_lines[-1]) + 0.5 ) # add 0.5 to round up
            if exposure_duration != exposure_durations[-1] or skipped_lines > 0:
                start_lines.append(line)
                start_times.append(start_time)
                exposure_durations.append(exposure_duration)
        return (start_lines, start_times, exposure_durations)


    @property
    def binary_exposure_durations(self):
        """
        Returns the exposure durations taken from the binary image data.

        For HRSC, the exposure durations are imbedded in the binary data of the image.
        The exposure durations start at the 9th byte of the line/record and are 4 bytes long.

        Returns
        -------
        : list
          Exposure durations
        """
        if not hasattr(self, '_binary_exposure_durations'):
            self.read_image_data()
        return self._binary_exposure_durations


    @property
    def binary_ephemeris_times(self):
        """
        Returns the ephemeris times taken from the binary image data.

        For HRSC, the ephemeris times are imbedded in the binary data of the image.
        The ephemeris times start at the first byte of the line/records and are 8 bytes long.

        Returns
        -------
        : list
          Ephemeris times
        """
        if not hasattr(self, '_binary_ephemeris_times'):
            self.read_image_data()
        return self._binary_ephemeris_times


    @property
    def binary_lines(self):
        """
        Returns the lines of the binary image data.

        For example, the first entry would be the first line of the image.

        Returns
        -------
        : list
          Image lines
        """
        if not hasattr(self, '_binary_lines'):
            self.read_image_data()
        return self._binary_lines


    def read_image_data(self):
        """
        Reads data off of image and stores in binary_exposure_durations, binary_lines,
        and binary_ephemeris_times.

        For HRSC, the exposure durations and ephemeris times are imbedded in the binary
        data of the image itself. Each line is stored in what is referred to as a
        "record" within the image. The label will have the size of each record,
        the number of records, and the number of records in the label, so the
        beginning of binary data can be calculated.

        For each line/record of the binary data, the first 8 bytes make up the
        double presicion value of the ephemeris time, with the next 4 bytes
        making up the float value of the line exposure duration for the
        associated line. NOTE: The prefix data is always LSB, regardless
        of the overall file format.
        """
        lines = []
        times = []
        durations = []

        with open(self._file, 'rb') as image_file:
            bytes_per_record = self.label['RECORD_BYTES']
            num_records = self.label['FILE_RECORDS']
            img_start_record = self.label['^IMAGE']
            img_start_byte = bytes_per_record * (img_start_record - 1) # Offset by one for zero-based records
            num_img_records = num_records - img_start_record
            image_file.seek(img_start_byte)

            for record in range(num_img_records):
                record_bytes = image_file.read(bytes_per_record)
                eph_time = struct.unpack('<d', record_bytes[:8])[0]
                exp_dur = struct.unpack('<f', record_bytes[8:12])[0] / 1000
                # Offset for zero-based corrections, and then offest for ISIS pixel definition
                lines.append(record+1-0.5)
                times.append(eph_time)
                durations.append(exp_dur)

        self._binary_exposure_durations = durations
        self._binary_lines = lines
        self._binary_ephemeris_times = times


    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris stop time.

        For HRSC, the ephemeris stop time is calculated from the binary image data.

        Returns
        -------
        : float
          Ephemeris stop time
        """
        return self.binary_ephemeris_times[-1] + self.binary_exposure_durations[-1]


    # TODO We need to confirm that returning nothing here does not affect
    # calculations elsewhere in code. Or is there possibly just a better way of
    # doing this?
    @property
    def line_exposure_duration(self):
        """
        Line exposure duration returns the time between the exposures for
        subsequent lines.

        Since HRSC is a variable line scan camera, it does not make sense to
        have one exposure duration value.

        Returns
        -------
        : float
          Returns the line exposure duration in seconds from the PDS3 label.
        """
        return


    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1


class MexHrscIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, RadialDistortion, Driver):
  
  @property
  def instrument_id(self):
      """
      Returns the name of the instrument

      Returns
      -------
      : str
        Name of the instrument
      """
      if(super().instrument_id != "HRSC"):
          raise Exception ("Instrument ID is wrong.")
      return self.label['IsisCube']['Archive']['DetectorId']


  @property
  def sensor_name(self):
      """
      Returns the name of the instrument. Need to over-ride isis_label because
      InstrumentName is not defined in the ISIS label for MEX HSRC cubes.

      Returns
      -------
      : str
        Name of the sensor
      """
      return self.instrument_id

  @property
  def detector_center_line(self):
    """
    Returns the center detector line.

    For HRSC, we are dealing with a single line, so center line will be 0.

    Returns
    -------
    : float
      Detector line of the principal point
    """
    return 0.0


  @property
  def detector_center_sample(self):
    """
    Returns the center detector sample.

    For HRSC, center sample is consistent regardless of filter. This is
    different from ISIS's center sample because ISIS line scan sensors use
    0.5 based detector samples.

    Returns
    -------
    : float
      Detector sample of the principal point
    """
    return 2592.0

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
  def times_table(self):
      """
      Returns EphemerisTime, ExposureTime, and LinesStart informtation which was stored as
      binary information in the ISIS cube.

      Returns
      -------
      : dict
        Dictionary with EphemerisTime, ExposureTime, and LineStart.
      """
      isis_bytes = read_table_data(self.label['Table'], self._file)
      return parse_table(self.label['Table'], isis_bytes)

  @property
  def line_scan_rate(self):
      """
      Returns
      -------
      : tuple
        list of lines, list of ephemeris times, and list of exposure
        times
      """
      return self.times_table['LineStart'], self.times_table['EphemerisTime'], self.times_table['ExposureTime']

  @property
  def ephemeris_start_time(self):
      """
      Returns
      -------
      : float
        starting ephemeris time
      """
      return self.times_table['EphemerisTime'][0]

  @property
  def ephemeris_stop_time(self):
      """
      Returns
      -------
      : float
        ephemeris stop time
      """
      last_line = self.times_table['LineStart'][-1]
      return self.times_table['EphemerisTime'][-1] + ((self.image_lines - last_line + 1) * self.times_table['ExposureTime'][-1])

  @property
  def ikid(self):
      """
      Returns the Naif ID code for the HRSC head instrument

      This would be the Naif ID code for the base (or "head") instrument.

      Returns
      -------
      : int
        Naif ID used to for identifying the instrument in Spice kernels
      """
      return spice.bods2c("MEX_HRSC_HEAD")


  @property
  def fikid(self):
      """
      Naif ID code of the filter dependent instrument codes.

      Expects filter_number to be defined. This should be an integer containing
      the filter number from the pds3 label.
      Expects ikid to be defined. This should be the integer Naif ID code for
      the instrument.

      Returns
      -------
      : int
        Naif ID code used in calculating focal length
      """
      return spice.bods2c(self.instrument_id)

  @property
  def focal2pixel_lines(self):
      """
      NOTE: These values are pulled from ISIS iak kernels.

      Returns
      -------
      : list<double>
        focal plane to detector lines
      """
      return [0.0, 0.0, 111.111111111111]


  @property
  def focal2pixel_samples(self):
      """
      NOTE: These values are pulled from ISIS iak kernels.

      Returns
      -------
      : list<double>
        focal plane to detector samples
      """
      return [0.0, 111.111111111111, 0.0]

class MexSrcPds3NaifSpiceDriver(Framer, Pds3Label, NaifSpice, RadialDistortion, Driver):
    """
    Driver for a PDS3 Mars Express (Mex) High Resolution Stereo Camera (HRSC) - Super Resolution 
    Channel (SRC) image.
    """

    @property
    def odtk(self):
        """
        The coefficients for the distortion model. No distortion model, so pass in all zeroes.

        Returns
        -------
        : list
          Radial distortion coefficients.
        """
        return [0.0, 0.0, 0.0]


    @property
    def ikid(self):
        """
        Returns the Naif ID code for HRSC SRC. 

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return spice.bods2c("MEX_HRSC_SRC")


    @property
    def instrument_id(self):
        """
        Returns the short name of the instrument

        MEX HRSC has nine different filters each with their own name.

         Returns
        -------
        : str
          Short name of the instrument
        """
        if(super().instrument_id != "HRSC"):
            raise Exception ("Instrument ID is wrong.")
        return self.label['DETECTOR_ID']


    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. MEX HRSC img PDS3 labels do not the have SPACECRAFT_NAME
        keyword, so we override it here to use the label_pds3 property for
        instrument_host_id

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_id


    @property
    def focal2pixel_lines(self):
        """
        NOTE: These values are pulled from ISIS iak kernels.

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        return [0.0, 0.0, 111.111111111111]


    @property
    def focal2pixel_samples(self):
        """
        NOTE: These values are pulled from ISIS iak kernels.

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        return [0.0, 111.111111111111, 0.0]


    @property
    def detector_center_line(self):
        """
        Returns the center detector line.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 512.0


    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample.

        This is
        different from ISIS's center sample because ISIS uses
        0.5-based samples.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 512.0


    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
