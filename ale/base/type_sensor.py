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
    def t0_ephemeris(self):
        return self.starting_ephemeris_time - self.center_ephemeris_time

    @property
    def t0_quaternion(self):
        return self.starting_ephemeris_time - self.center_ephemeris_time

    @property
    def dt_ephemeris(self):
        return (self.ending_ephemeris_time - self.starting_ephemeris_time) / self.number_of_ephemerides

    @property
    def dt_quaternion(self):
        return (self.ending_ephemeris_time - self.starting_ephemeris_time) / self.number_of_quaternions

    @property
    def line_scan_rate(self):
        """
        Returns
        -------
        : list
          2d list of scan rates in the form: [[start_line, line_time, exposure_duration], ...]
        """
        return [[float(self.starting_detector_line), self.t0_ephemeris, self.line_exposure_duration]]

    @property
    def number_of_ephemerides(self):
        return self._num_ephem

    @property
    def number_of_quaternions(self):
        #TODO: Not make this hardcoded
        return self._num_quaternions

    @property
    def ending_ephemeris_time(self):
        return (self.image_lines * self.line_exposure_duration) + self.starting_ephemeris_time

    @property
    def center_ephemeris_time(self):
        """
        The center ephemeris time for a fixed rate line scanner.
        """
        if not hasattr(self, '_center_ephemeris_time'):
            halflines = self.image_lines / 2
            center_sclock = self.starting_ephemeris_time + halflines * self.line_exposure_duration
            self._center_ephemeris_time = center_sclock
        return self._center_ephemeris_time

    @property
    def line_exposure_duration(self):
        return self.label['LINE_EXPOSURE_DURATION'].value * 0.001  # Scale to seconds

class Framer():
    @property
    def name_sensor(self):
        return "Generic Framer"

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
    def filter_number(self):
        return self.label.get('FILTER_NUMBER', 0)

    @property
    def number_of_ephemerides(self):
        # always one for framers
        return 1

    @property
    def number_of_quaternions(self):
        # always one for framers
        return 1

    @property
    def center_ephemeris_time(self):
        """
        The center ephemeris time for a framer.
        """
        center_time = self.starting_ephemeris_time + self.exposure_duration / 2
        return center_time

    @property
    def exposure_duration(self):
        return self._exposure_duration
