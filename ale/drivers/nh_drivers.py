import numpy as np
import spiceypy as spice
import pvl

from ale import util

from ale.base import Driver
from ale.base.type_distortion import NoDistortion, LegendreDistortion
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_sensor import LineScanner

class NewHorizonsLorriIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading New Horizons LORRI ISIS3 Labels. These are Labels that have been
    ingested into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
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
        id_lookup = {
            "LORRI" : "NH_LORRI"
        }
        return id_lookup[super().instrument_id]

    @property
    def ephemeris_stop_time(self):
        return super().ephemeris_start_time

    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        For LORRI, there are two options associated with different binning modes:
        1x1 binning: -98301
        4x4 binning: -98302

        Returns
        -------
        : integer
          Naif Integer ID code for the instrument
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

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

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def frame_chain(self):
        self._props['exact_ck_times'] = False
        return super().frame_chain

class NewHorizonsLeisaIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading New Horizons LEISA ISIS3 Labels.
    """

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
        id_lookup = {
            "LEISA" : "NH_RALPH_LEISA"
        }
        return id_lookup[super().instrument_id]

    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode'][0]

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        return spice.scs2e(self.spacecraft_id, self.spacecraft_clock_start_count)

    @property
    def ephemeris_stop_time(self):
        """
        ISIS doesn't preserve the spacecraft stop count that we can use to get
        the ephemeris stop time of the image, so compute the ephemeris stop time
        from the start time and the exposure duration.
        """
        return self.ephemeris_start_time + self.exposure_duration * self.image_lines

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 0

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument. Need to over-ride isis_label because
        InstrumentName is not defined in the ISIS label for NH Leisa cubes.

        Returns
        -------
        : str
        Name of the sensor
        """
        return self.instrument_id

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
    def exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        return self.label['IsisCube']['Instrument']['ExposureDuration']


class NewHorizonsMvicIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, LegendreDistortion, Driver):
    """
    Driver for reading New Horizons MVIC ISIS3 Labels. These are Labels that have been
    ingested into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """

    @property
    def parent_id(self):
        """
        The base naif id of the spacecraft.  For New Horizons, this is -98000.
        Required for distortion coefficients, which are not unique to instruments,
        but are instead shared by all instruments on the spacecraft + residuals.
        Returns
        -------
        : int
          Naif id of the spacecraft
        """
        return round(self.ikid, -2)


    @property
    def sensor_model_version(self):
        """
        Returns instrument model version
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1


    @property
    def instrument_name(self):
        """
        The name of the instrument.  This is not included in the .fit label, but is
        present in the .lbl file, so it is not present in ISIS conversion, and it
        must be hard-coded.
        Returns
        -------
        : str
          Name of the instrument
        """
        return "MULTISPECTRAL VISIBLE IMAGING CAMERA"


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
        id_lookup = {
          "MVIC_FRAMING" : "NH_MVIC"
        }
        return id_lookup[super().instrument_id]


    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.
        Returns
        -------
        : integer
          Naif Integer ID code for the instrument
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode'][0]


    @property
    def detector_center_line(self):
        """ Returns detector center line.  This information is found in ik/nh_ralph_v100.ti, which
        is not loaded as an ik."""
        return -1


    @property
    def detector_center_sample(self):
        """ Returns detector center line.  This information is found in ik/nh_ralph_v100.ti, which
        is not loaded as an ik."""
        return 0


    @property
    def sensor_name(self):
        """
        Returns the name of the instrument
        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']


    @property
    def odtx(self):
        """
        Returns the x coefficient for the optical distortion model
        Expects ikid to be defined. This must be the integer Naif id code of the instrument
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.parent_id),0, 20).tolist()


    @property
    def odty(self):
        """
        Returns the y coefficient for the optical distortion model.
        Expects ikid to be defined. This must be the integer Naif id code of the instrument
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.parent_id), 0, 20).tolist()

    @property
    def band_times(self):
        if not hasattr(self, "_ephem_band_times"):
            band_times = self.label['IsisCube']['BandBin']['UtcTime']
            self._ephem_band_times = []
            for time in band_times:
                if type(time) is pvl.Quantity:
                   time = time.value
                self._ephem_band_times.append(spice.utc2et(time.strftime("%Y-%m-%d %H:%M:%S.%f")))
        return self._ephem_band_times


    @property
    def ephemeris_time(self):
        """
        Returns an array of times between the start/stop ephemeris times
        based on the number of lines in the image.
        Expects ephemeris start/stop times to be defined. These should be
        floating point numbers containing the start and stop times of the
        images.
        Expects image_lines to be defined. This should be an integer containing
        the number of lines in the image.

        Returns
        -------
        : ndarray
          ephemeris times split based on image lines
        """
        if not hasattr(self, "_ephemeris_time"):
          self._ephemeris_time = np.linspace(self.ephemeris_start_time, self.ephemeris_stop_time, self.image_lines + 1)
        return self._ephemeris_time 


    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        return self.band_times[0]


    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris stop time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris stop time of the image
        """
        return self.band_times[-1]


    @property
    def naif_keywords(self):
        """
        Adds base NH instrument distortion, which is shared among all instruments on NH.
        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        return {**super().naif_keywords,
                f"INS{self.parent_id}_DISTORTION_COEF_X": self.odtx,
                f"INS{self.parent_id}_DISTORTION_COEF_Y": self.odty}

class NewHorizonsMvicTdiIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, LegendreDistortion, Driver):
    """
    Driver for reading New Horizons MVIC TDI ISIS3 Labels. These are Labels that have been
    ingested into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
    @property
    def parent_id(self):
        """
        The base naif id of the spacecraft.  For New Horizons, this is -98000.
        Required for distortion coefficients, which are not unique to instruments,
        but are instead shared by all instruments on the spacecraft + residuals.
        Returns
        -------
        : int
          Naif id of the spacecraft
        """
        return round(self.ikid, -2)

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
        id_lookup = {
            "MVIC_TDI" : "ISIS_NH_RALPH_MVIC_METHANE"
        }
        return id_lookup[super().instrument_id]

    @property
    def ikid(self):
        """
        Read the ikid from the cube label
        """
        if not hasattr(self, "_ikid"):
            self._ikid = None
            # Attempt to get the frame code using frame name,
            # If that fails, try to get it directly from the cube label
            try:
                self._ikid = spice.frmname(self.instrument_id)
            except:
                self._ikid = self.label["IsisCube"]["Kernels"]["NaifFrameCode"].value
        return self._ikid

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
        Returns the exposure duration for MVIC TDI cameras

        Returns
        -------
        : float
          The exposure duration
        """
        return (1.0 / self.label["IsisCube"]["Instrument"]["TdiRate"].value)

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. This is currently defaulted to 0
        as it is unused in ISIS

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 0

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. This is currently defaulted to 0
        as it is unused in ISIS

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0

    @property
    def odtx(self):
        """
        Returns the x coefficient for the optical distortion model
        Expects ikid to be defined. This must be the integer Naif id code of the instrument
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.parent_id),0, 20).tolist()

    @property
    def odty(self):
        """
        Returns the y coefficient for the optical distortion model.
        Expects ikid to be defined. This must be the integer Naif id code of the instrument
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.parent_id), 0, 20).tolist()

    @property
    def naif_keywords(self):
        """
        Adds base NH instrument distortion, which is shared among all instruments on NH.

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        return {**super().naif_keywords,
                f"INS{self.parent_id}_DISTORTION_COEF_X": self.odtx,
                f"INS{self.parent_id}_DISTORTION_COEF_Y": self.odty}

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1