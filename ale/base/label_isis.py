import pvl

class IsisLabel():

    @property
    def label(self):
        if not hasattr(self, "_label"):
            if isinstance(self._file, pvl.PVLModule):
                self._label = self._file
            try:
                self._label = pvl.loads(self._file)
            except Exception:
                self._label = pvl.load(self._file)
            except:
                raise ValueError("{} is not a valid label".format(self._file))
        return self._label

    @property
    def instrument_id(self):
        """
        Returns
        -------
        : str
          instrument id
        """
        return self.label['IsisCube']['Instrument']['InstrumentId']

    @property
    def platform_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['InstrumentName']

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
          Sample summing
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
          Line summing
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
        if 'SpacecraftClockStartCount' in self.label['IsisCube']['Instrument']:
            return self.label['IsisCube']['Instrument']['SpacecraftClockStartCount']
        elif 'SpacecraftClockCount' in self.label['IsisCube']['Instrument']:
            return self.label['IsisCube']['Instrument']['SpacecraftClockCount']
        elif 'SpacecraftClockStartCount' in self.label['IsisCube']['Archive']:
            return self.label['IsisCube']['Archive']['SpacecraftClockStartCount']
        else:
            return None

    @property
    def spacecraft_clock_stop_count(self):
        """
        The spacecraft clock stop count, frequently used to determine the stop time
        of the image.

        Returns
        -------
        : str
          Spacecraft clock stop count
        """
        if 'SpacecraftClockStopCount' in self.label['IsisCube']['Instrument']:
            return self.label['IsisCube']['Instrument']['SpacecraftClockStopCount']
        elif 'SpacecraftClockStopCount' in self.label['IsisCube']['Archive']:
            return self.label['IsisCube']['Archive']['SpacecraftClockStopCount']
        else:
            return None

    @property
    def utc_start_time(self):
        """
        The UTC start time of the image.
        This is generally less accurate than converting the spacecraft start
        clock count using the spacecraft clock kernel (SCLK).

        Returns
        -------
        : datetime
          Start time of the image in UTC
        """
        return self.label['IsisCube']['Instrument']['StartTime']

    @property
    def utc_stop_time(self):
        """
        The UTC stop time of the image.
        This is generally less accurate than converting the spacecraft stop
        clock count using the spacecraft clock kernel (SCLK).

          Returns
        -------
        : datetime
          Stop time of the image in UTC
        """
        return self.label['IsisCube']['Instrument']['StopTime']

    @property
    def  exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        if 'EXPOSURE_DURATION' in self.label:
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
        else:
            return self.line_exposure_duration

    @property
    def line_exposure_duration(self):
        """
        The line exposure duration of the image, in seconds

        Returns
        -------
        : float
          Line exposure duration in seconds
        """
        return self.label['IsisCube']['Instrument']['LineExposureDuration'].value * 0.001 # scale to seconds
