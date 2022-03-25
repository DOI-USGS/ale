from ale.base.data_isis import IsisSpice
from ale.base.label_isis import IsisLabel
from ale.base import Driver
from ale.base.type_sensor import LineScanner
from ale.base.type_distortion import NoDistortion


class IdealLsIsisLabelIsisSpiceDriver(LineScanner, IsisSpice, IsisLabel, NoDistortion, Driver):
    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          name of the instrument.
        """
        return self.instrument_id

    @property
    def instrument_id(self):
        instrument_id = super().instrument_id

        if instrument_id != "IdealCamera":
            raise Exception(f"Instrument ID is {instrument_id} when it should be \"IdealCamera\"")

        return instrument_id

    @property
    def ephemeris_start_time(self):
        """
        The image start time in ephemeris time.

        Returns
        -------
        float :
            The image start ephemeris time
        """

        return self.label.get('IsisCube').get('Instrument').get("EphemerisTime").value


    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the number of lines
        times the exposure duration. Expects ephemeris start time, exposure duration
        and image lines to be defined. These should be double precision numbers
        containing the ephemeris start, exposure duration, and number of lines of
        the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return super().ephemeris_stop_time


    @property
    def spacecraft_name(self):
        """
        Returns the spacecraft name used in various Spice calls to acquire
        ephemeris data.
        Expects the platform_name to be defined. This should be a string of
        the form 'Mars_Reconnaissance_Orbiter'

        Returns
        -------
        : str
          spacecraft name
        """
        return super().platform_name


    @property
    def detector_start_line(self):
        """
        Returns the starting detector line for the image.

        Returns
        -------
        : int
          Starting detector line for the image
        """
        return 0


    @property
    def detector_start_sample(self):
        """
        Returns the starting detector sample for the image.

        Returns
        -------
        : int
          Starting detector sample for the image
        """
        return 0


    @property
    def sensor_model_version(self):
        """
        Returns the ISIS sensor model version.

        Returns
        -------
        : int
          ISIS sensor model version
        """

        return 1


    @property
    def pixel2focal_x(self):
        """
        Returns detector to focal plane x.

        Returns
        -------
        : list<double>
          detector to focal plane x
       """
        return self.naif_keywords.get('IDEAL_TRANSX')


    @property
    def pixel2focal_y(self):
        """
        Returns detector to focal plane y.

        Returns
        -------
        : list<double>
          detector to focal plane y
       """
        return self.naif_keywords.get('IDEAL_TRANSY')


    @property
    def focal2pixel_lines(self):
        """
        Returns focal plane to detector lines.

        Returns
        -------
        : list<double>
          focal plane to detector lines
       """

        return self.naif_keywords.get('IDEAL_TRANSL')


    @property
    def focal2pixel_samples(self):
        """
        Returns focal plane to detector samples.

        Returns
        -------
        : list<double>
          focal plane to detector samples
       """
        return self.naif_keywords.get('IDEAL_TRANSS')

    @property
    def focal_length(self):
        """
        The focal length of the instrument
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keywords from the label.

        Returns
        -------
        float :
            The focal length in millimeters
        """
        return self.naif_keywords.get('IDEAL_FOCAL_LENGTH', None)

    @property
    def detector_center_sample(self):
        """
        The center sample of the CCD in detector pixels

        Returns
        -------
        float :
            The center sample of the CCD
        """
        return self.label['IsisCube']['Instrument']['SampleDetectors'] / 2.0

    @property
    def detector_center_line(self):
        """
        The center line of the CCD in detector pixels

        Returns
        -------
        float :
            The center of line the CCD
        """
        return 0.0
