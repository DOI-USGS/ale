from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import LineScanner, Framer
from ale.base.type_distortion import NoDistortion

############################## HARD CODED VALUES ###############################
# These values are hard coded from the SIS. As the kernels mature, they will
# eventually be available in the IK.

EIS_FOCAL_LENGTHS = {
    "EUROPAM_EIS_WAC" : 46.25,
    "EUROPAM_EIS_NAC" : 1002.7
}

# According to diagrams in the SIS, increasing lines are +Y and increasing samples are -X
# Focal to detector transforms assume 0.014 mm pixel pitch.
EIS_ITRANSL = [0.0, 0.0, 1.0 / 0.014]
EIS_ITRANSS = [0.0, -1.0 / 0.014, 0.0]

EIS_NAC_FILTER_CODES = {
    "CLEAR" : -159121,
    "NUV" : -159125,
    "BLU" : -159126,
    "GRN" : -159127,
    "RED" : -159128,
    "IR1" : -159129,
    "1MU" : -159130
}

EIS_WAC_FILTER_CODES = {
    "CLEAR" : -159141,
    "NUV" : -159145,
    "BLU" : -159146,
    "GRN" : -159147,
    "RED" : -159148,
    "IR1" : -159149,
    "1MU" : -159150
}

# The CMOS can read any line in each region, so for now just take the top line
# These values differ from the SIS because the SIS takes the top left corner as
# (4095, 2047) and we take it as (0, 0). So, these values are 2047 - the value
# in the SIS
EIS_FILTER_START_LINES = {
    "CLEAR" : 580,
    "NUV" : 415,
    "BLU" : 351,
    "GRN" : 287,
    "RED" : 223,
    "IR1" : 159,
    "1MU" : 95
}

# These values differ from the SIS because the SIS takes the top left corner as
# (4095, 2047) and we take it as (0, 0). So, these values are 4095 - the value
# in the SIS
EIS_FILTER_START_SAMPLES = {
    "CLEAR" : 132,
    "NUV" : 32,
    "BLU" : 32,
    "GRN" : 32,
    "RED" : 32,
    "IR1" : 32,
    "1MU" : 32
}

class ClipperEISWACFCIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """
        id_lookup = {
            "EIS WAC FC": "EUROPAM_EIS_WAC"
        }

        return id_lookup[super().instrument_id]

    @property
    def fikid(self):
        """
        The NAIF ID for the filter.

        Returns
        -------
        : int
          The NAIF ID for the filter
        """
        return EIS_WAC_FILTER_CODES[self.filter_name]

    @property
    def filter_name(self):
        """
        The name of the filter used to capture the image

        Returns
        -------
        : string
          The name of the filter
        """
        return self.label['IsisCube']['BandBin']['FilterName'].value

    @property
    def detector_center_sample(self):
        """
        The center sample of the detector in zero based detector pixels
        This isn't in the IK so hardcode it to the absolute center

        Returns
        -------
        float :
            The center sample of the detector
        """
        return 4096 / 2

    @property
    def detector_center_line(self):
        """
        The center line of the detector in zero based detector pixels
        This isn't in the IK so hardcode it to the absolute center

        Returns
        -------
        float :
            The center line of the detector
        """
        return 2048 / 2

    @property
    def detector_start_line(self):
        """
        The first zero based line of the detector read out

        Returns
        -------
        : int
          Zero based Detector line corresponding to the first image line
        """
        return EIS_FILTER_START_LINES[self.filter_name]

    @property
    def detector_start_sample(self):
        """
        The first zero based sample of the detector read out

        Returns
        -------
        : int
          Zero based Detector sample corresponding to the first image sample
        """
        return EIS_FILTER_START_SAMPLES[self.filter_name]

    @property
    def focal_length(self):
        """
        The focal length in millimeters

        Returns
        -------
        : float
          focal length
        """
        return EIS_FOCAL_LENGTHS[self.instrument_id]


class ClipperEISWACPBIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """
        id_lookup = {
            "EIS WAC PB": "EUROPAM_EIS_WAC"
        }

        return id_lookup[super().instrument_id]

    @property
    def fikid(self):
        """
        The NAIF ID for the filter.

        Returns
        -------
        : int
          The NAIF ID for the filter
        """
        return EIS_WAC_FILTER_CODES[self.filter_name]

    @property
    def filter_name(self):
        """
        The name of the filter used to capture the image

        Returns
        -------
        : string
          The name of the filter
        """
        return self.label['IsisCube']['BandBin']['FilterName'].value

    @property
    def detector_center_sample(self):
        """
        The center sample of the detector in zero based detector pixels
        This isn't in the IK so hardcode it to the absolute center

        Returns
        -------
        float :
            The center sample of the detector
        """
        return 4096 / 2

    @property
    def detector_center_line(self):
        """
        The center line of the detector in zero based detector pixels
        This isn't in the IK so hardcode it to the absolute center

        Returns
        -------
        float :
            The center line of the detector
        """
        return 2048 / 2

    @property
    def detector_start_line(self):
        """
        The first zero based line of the detector read out

        Returns
        -------
        : int
          Zero based Detector line corresponding to the first image line
        """
        return EIS_FILTER_START_LINES[self.filter_name]

    @property
    def detector_start_sample(self):
        """
        The first zero based sample of the detector read out

        Returns
        -------
        : int
          Zero based Detector sample corresponding to the first image sample
        """
        return EIS_FILTER_START_SAMPLES[self.filter_name]

    @property
    def focal_length(self):
        """
        The focal length in millimeters

        Returns
        -------
        : float
          focal length
        """
        return EIS_FOCAL_LENGTHS[self.instrument_id]

    @property
    def focal2pixel_lines(self):
        """
        The transformation from focal plan coordinates to detector lines.
        To transform the coordinate (x,y) to detector lines do the following:

        lines = focal2pixel_lines[0] + x * focal2pixel_lines[1] + y * focal2pixel_lines[2]

        Returns
        -------
        : list<double>
          focal plane to detector lines transform
        """
        return EIS_ITRANSL

    @property
    def focal2pixel_samples(self):
        """
        The transformation from focal plan coordinates to detector samples.
        To transform the coordinate (x,y) to detector samples do the following:

        samples = focal2pixel_samples[0] + x * focal2pixel_samples[1] + y * focal2pixel_samples[2]

        Returns
        -------
        : list<double>
          focal plane to detector samples transform
        """
        return EIS_ITRANSS