import numpy as np

class LineScanner():

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL"

    @property
    def line_scan_rate(self):
        """
        Expects ephemeris_start_time to be defined. This should be a float
        containing the start time of the image.
        Expects center_ephemeris_time to be defined. This should be a float
        containing the average of the start and end ephemeris times.

        Returns
        -------
        : list
          2d list of scan rates in the form: [[start_line, line_time, exposure_duration], ...]
        """
        t0_ephemeris = self.ephemeris_start_time - self.center_ephemeris_time
        return [[0.5], [t0_ephemeris], [self.exposure_duration]]

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
        return np.linspace(self.ephemeris_start_time,  self.ephemeris_stop_time, self.image_lines / 64)

    @property
    def center_ephemeris_time(self):
        """
        Returns the sum of the starting ephemeris time and half of the line exposure
        duration. Expects ephemeris start time and line exposure duration to be
        defined. These should be double precision numbers containing the ephemeris start and
        line exposure duration of the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time + self.line_exposure_duration / 2

class Framer():
    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_FRAME_SENSOR_MODEL"

    @property
    def ephemeris_time(self):
        """
        Returns the center ephemeris time for the image which is start time plus
        half of the exposure duration.
        Expects center_ephemeris_time to be defined. This should be a double
        containing the average of the start and stop ephemeris times.

        Returns
        -------
        : double
          Center ephemeris time for the image
        """
        return [self.center_ephemeris_time

    @property
    def center_ephemeris_time(self):
        """
        Returns the sum of the starting ephemeris time and half of the exposure duration.
        Expects ephemeris start time and exposure duration to be defined. These
        should be double precision numbers containing the ephemeris start and
        exposure duration of the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time + self.exposure_duration / 2
