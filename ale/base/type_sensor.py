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
        Returns
        -------
        : list
          2d list of scan rates in the form: [[start_line, line_time, exposure_duration], ...]
        """
        t0_ephemeris = self.starting_ephemeris_time - self.center_ephemeris_time
        return [[float(self.starting_detector_line)], [t0_ephemeris], [self._line_exposure_duration]]

    @property
    def ephemeris_time(self):
        return np.linspace(self.starting_ephemeris_time,  self.ending_ephemeris_time, self.image_lines)

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
        return [self.center_ephemeris_time]
