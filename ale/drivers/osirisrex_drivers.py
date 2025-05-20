import ale
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.base import Driver
from ale.base.type_distortion import RadialDistortion

from ale import util



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
    def polyCamFocusPositionNaifId(self):
        """
        Returns the focal length specific Naif ID for USGS Astro
        IAK distortion model look ups.

        Returns
        -------
        : int
          Special focal length specific Naif ID
        """
        if self.instrument_id == "ORX_OCAMS_POLYCAM":
            return self.label['IsisCube']['Instrument']['PolyCamFocusPositionNaifId']
        else:
            return None

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
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : int
          The detector line of the principle point
        """
        return float(self.naif_keywords['INS{}_CCD_CENTER'.format(self.ikid)][1])

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

        return float(self.naif_keywords['INS{}_CCD_CENTER'.format(self.ikid)][0])

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
            return self.naif_keywords['INS{}_OD_K'.format(self.ikid)][0:3]
        else:
            if self.polyCamFocusPositionNaifId != None:
                return self.naif_keywords['INS{focusId}_OD_K_{filter}'.format(focusId = self.polyCamFocusPositionNaifId, filter = self.filter_name)][0:3]
            return self.naif_keywords['INS{ikid}_OD_K_{filter}'.format(ikid = self.ikid, filter = self.filter_name)][0:3]
        
    @property
    def naif_keywords(self):
        """
        Gets all default naif keywords as well as any naif keywords that
        contain the special focal length specific Naif ID

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        keywords = self.spiceql_call("findMissionKeywords", {"key": str(self.polyCamFocusPositionNaifId), "mission" : self.spiceql_mission})
        if keywords: 
          return super().naif_keywords | keywords 
        else: 
          return super().naif_keywords 

    @property
    def sensor_model_version(self):
        """
        Returns the ISIS camera version

        Returns
        -------
        : int
          Camera version number
        """
        return 1
