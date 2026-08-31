import pyspiceql

from ale.base import Driver, WrongInstrumentException
from ale.base.label_isis import IsisLabel
from ale.base.data_naif import NaifSpice
from ale.base.type_distortion import NoDistortion
from ale.base.type_sensor import Framer


class ClementineIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading UUVIS, HIRES, NIR, and LWIR ISIS3 Labels
    """
    
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument,
        but often also used to be piped into Spice Kernels to acquire
        IKIDS. Therefor they are the same ID that Spice expects in bods2c
        calls. Expect instrument_id to be defined in the IsisLabel mixin.
        This should be a string of the form NEAR EARTH ASTEROID RENDEZVOUS

        Returns
        -------
        : str
          instrument id
        """
        lookup_table = {
        "UVVIS": "ULTRAVIOLET/VISIBLE CAMERA",
        "NIR": "Near Infrared Camera",
        "HIRES": "High Resolution Camera",
        "LWIR": "Long Wave Infrared Camera"
        }
        key = super().instrument_id
        if key not in lookup_table:
            raise WrongInstrumentException(f"Unknown instrument id: {key}.")
        return lookup_table[key]

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          instrument name
        """
        return super().instrument_id

    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft

        Returns
        -------
        : str
          spacecraft name
        """
        return super().spacecraft_name.replace(" ", "_")

    @property
    def sensor_model_version(self):
        """
        Returns ISIS sensor model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def ephemeris_start_time(self):
        if not hasattr(self, "_ephemeris_start_time"):
            self._ephemeris_start_time = pyspiceql.utcToEt(utc=self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"), searchKernels=self.search_kernels, useWeb=self.use_web)[0]
        return self._ephemeris_start_time
        
    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the exposure duration.
        Expects ephemeris start time and exposure duration to be defined. These
        should be double precision numbers containing the ephemeris start and
        exposure duration of the image.
        Returns
        -------
        : double
          Ephemeris stop time for an image
        """
        return self.ephemeris_start_time + self.exposure_duration

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
        return self.label["IsisCube"]["Kernels"]["NaifFrameCode"]
    
    @property
    def focal_length(self):
        """
        NIR manually sets focal length based on filter.

        Returns
        -------
        : float
          focal length
        """
        if (self.instrument_id == "Near Infrared Camera"):
          filter = self.label["IsisCube"]['BandBin']['FilterName']

          lookup_table = {
          "A": 2548.2642,
          "B": 2530.8958,
          "C": 2512.6589,
          "D": 2509.0536,
          "E": 2490.7378,
          "F": 2487.8694
          }

          return lookup_table[filter.upper()] * 0.038

        return super().focal_length

    @property
    def usgscsm_distortion_model(self):
        """
        The four Clementine cameras use different optical distortion models in
        ISIS. Only the NIR camera is modeled here as a single-parameter radial
        distortion: ISIS NirCamera uses RadialDistortionMap(this, -0.0006364),
        i.e. undistorted = distorted * (1 + k1 * r^2) with k1 = -0.0006364. The
        USGSCSM RADIAL model applies undistorted = distorted * (1 - (c0 + c1 r^2
        + c2 r^4)), so matching gives coefficients [0, -k1, 0] = [0, 0.0006364, 0].
        The UVVIS (radial + decentering), HiRes (generic radial) and LWIR
        (zero-coefficient) cameras are left distortion-free here.

        Returns
        -------
        : dict
          Dictionary containing the usgscsm distortion model
        """
        if self.instrument_id == "Near Infrared Camera":
            return {"radial": {"coefficients": [0.0, 0.0006364, 0.0]}}
        return {"radial": {"coefficients": [0.0, 0.0, 0.0]}}

    @property
    def detector_center_line(self):
        # ISIS detector coordinates are 0.5-based (pixel centers at half integers);
        # CSM is 0-based. Subtract 0.5 for the NIR camera, whose distortion is
        # modeled above, as the LRO, MRO, Dawn and Cassini ISS drivers already do.
        # UVVIS and HiRes distortion are not modeled here, so their detector center
        # is left unchanged.
        if self.instrument_id == "Near Infrared Camera":
            return super().detector_center_line - 0.5
        return super().detector_center_line

    @property
    def detector_center_sample(self):
        if self.instrument_id == "Near Infrared Camera":
            return super().detector_center_sample - 0.5
        return super().detector_center_sample