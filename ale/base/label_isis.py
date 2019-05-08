
class IsisLabel():

    @property
    def start_time(self):
        return self.label['IsisCube']['Instrument']['StartTime']

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def image_lines(self):
        """
        Returns
        -------
        : int
          Number of lines in image
        """
        return self.label['IsisCube']['Core']['Dimensions']['Lines']

    @property
    def image_samples(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        return self.label['IsisCube']['Core']['Dimensions']['Samples']

    @property
    def target_name(self):
        """
        Target name used in various Spice calls to acquire
        target specific ephemeris data.

        Returns
        -------
        : str
          Target name
        """
        return self.label['IsisCube']['Instrument']['TargetName']

    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            sclock = self.label['IsisCube']['Archive']['SpacecraftClockStartCount']
            self._starting_ephemeris_time = spice.scs2e(self.spacecraft_id, sclock).value
        return self._starting_ephemeris_time

    @property
    def _exposure_duration(self):
        return self.label['IsisCube']['Instrument']['ExposureDuration'].value * 0.001
