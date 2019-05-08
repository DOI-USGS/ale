
class IsisLabel():

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
    def sample_summing(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        try:
            summing = self.label['IsisCube']['Instrument']['SummingMode']
        except:
            summing = 1
        return summing

    @property
    def line_summing(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        try:
            summing = self.label['IsisCube']['Instrument']['SummingMode']
        except:
            summing = 1
        return summing

    @property
    def target_name(self):
        """
        Target body name used in various Spice calls to acquire
        target specific ephemeris data.

        Returns
        -------
        : str
          Target name
        """
        return self.label['IsisCube']['Instrument']['TargetName']

    @property
    def spacecraft_clock_start_count(self):
        """
        The spacecraft clock start count, frequently used to determine the start time
        of the image.

        Returns
        -------
        : str
          Spacecraft clock start count
        """
        try:
            start_count = self.label['IsisCube']['Instrument']['SpacecraftClockStartCount']
        except: 
            start_count = self.label['IsisCube']['Archive']['SpacecraftClockStartCount']

        return start_count

    @property
    def  exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        try:
            units = self.label['IsisCube']['Instrument']['ExposureDuration'].units
            if "ms" in units.lower():
                exposure_duration = self.label['IsisCube']['Instrument']['ExposureDuration'].value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                exposure_duration = self.label['IsisCube']['Instrument']['ExposureDuration'].value
        except:
            # if no units are available, assume the exposure duration is given in milliseconds
            exposure_duration = self.label['IsisCube']['Instrument']['ExposureDuration'].value * 0.001
        return exposure_duration
