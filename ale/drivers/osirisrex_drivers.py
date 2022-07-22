import spiceypy as spice

import ale
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.base import Driver
from ale.base.type_distortion import RadialDistortion

class OsirisRexCameraIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, RadialDistortion, Driver):
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Returns
        -------
        : str
          instrument id
        """
        sensor_lookup = {
        "MapCam" : "MAPCAM",
        "PolyCam" : "POLYCAM",
        "SamCam" : "SAMCAM"
        }
        return 'ORX_OCAMS_' + sensor_lookup[super().instrument_id]

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label['IsisCube']['Instrument']['InstrumentId']

    @property
    def exposure_duration(self):
        """
        The exposure duration of the image, in seconds. Unit is "Millisec" in label
        therefore failing in the label_isis exposure duration checks.

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        return self.label['IsisCube']['Instrument']['ExposureDuration'].value * 0.001


    @property
    def sensor_frame_id(self):
        """
        Returns the Naif ID code for the sensor reference frame.
        This is the frame of the OsirisRex instrument itself, and is not dependent on filter.

        Returns
        -------
        : int
          Naif ID code for the sensor frame
        """
        return -64000


    @property
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : int
          The detector line of the principle point
        """
        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[1])

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : int
          The detector sample of the principle point
        """

        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[0])

    @property
    def filter_name(self):
        """
        The name of the filter used to capture the image

        Returns
        -------
        : string
          The name of the filter
        """
        return self.label['IsisCube']['BandBin']['FilterName'].strip().upper()

    @property
    def odtk(self):
        """
        The coefficients for the radial distortion model
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list
          Radial distortion coefficients
        """
        if self.filter_name == "UNKNOWN":
            return spice.gdpool('INS{}_OD_K'.format(self.ikid),0, 3).tolist()
        else:
            return spice.gdpool('INS{ikid}_OD_K_{filter}'.format(ikid = self.ikid, filter = self.filter_name),0, 3).tolist()
