import os
from glob import glob

import numpy as np

import pvl
import struct
import spiceypy as spice
import warnings

from ale import config
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import LineScanner
from ale.base.type_distortion import RadialDistortion
from ale.util import find_latest_metakernel

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
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        warnings.warn('This driver has currently not been tested! Use at your own risk.')
        self._metakernel_dir = config.mex
        if not hasattr(self, '_metakernel'):
            self._metakernel = find_latest_metakernel(self._metakernel_dir, self.utc_start_time.year)
        return self._metakernel


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
          Naif ID used to for indentifying the instrument in Spice kernels
        """
        return spice.bods2c("MEX_HRSC_HEAD")


    @property
    def fikid(self):
        """
        Naif ID code of the filter dependent instrument codes.

        Expects filter_number to be defined. This should be an integer containing
        the filter number from the pds3 label.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        return spice.bods2c(self.label['DETECTOR_ID'])


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
        the filter-sepcific instrument.

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
        the filter-sepcific instrument.

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

        For HRSC, we are dealing with a single line, so center line will be 1.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0.0


    @property
    def detector_start_sample(self):
        """
        Returns
        -------
        : int
          Detector line corresponding to the first image sample
        """
        return self.label["SAMPLE_FIRST_PIXEL"]


    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample.

        For HRSC, center sample is consistent regardless of filter.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 2592.5


    @property
    def line_scan_rate(self):
        """
        Returns a 2D array of line scan rates.

        For HRSC, this data is actually imbedded in the binary data of the image
        itself. Each line is stored in what is referred to as a "record" within
        the image. The label will have the size of each record, the number of
        records, and the number of records in the label, so the beginning of
        binary data can be calculated.

        For each line/record of the binary data, the first 8 bytes make up the
        double presicion value of the ephemeris time, with the next 4 bytes
        making up the float value of the line exposure duration for the
        associated line. NOTE: The image label specifies MSB_INTEGER as the byte
        order, however, to match ISIS values, we used python struct's little
        endian functionality.

        Returns
        -------
        : list
          Start lines
        : list
          Line times
        : list
          Exposure durations
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
                if record == 0:
                    # Offset for zero-based corrections, and then offest for ISIS pixel definition
                    lines.append(record+1-0.5)
                    times.append(eph_time)
                    durations.append(exp_dur)
                # Only add records if exposure duration has changed since the line before
                elif exp_dur != durations[-1]:
                    # Offset for zero-based corrections, and then offest for ISIS pixel definition
                    lines.append(record+1-0.5)
                    times.append(eph_time)
                    durations.append(exp_dur)

        return lines, times, durations


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
