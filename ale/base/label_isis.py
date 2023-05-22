import pvl

class IsisLabel():
    """
    Mix-in for parsing ISIS Cube labels.
    """

    @property
    def label(self):
        if not hasattr(self, "_label"):
            if isinstance(self._file, pvl.PVLModule):
                self._label = self._file
            grammar = pvl.grammar.ISISGrammar()
            grammar.comments+=(("#", "\n"), )
            try:
                self._label = pvl.loads(self._file, grammar=grammar)
            except Exception:
                self._label = pvl.load(self._file, grammar=grammar)
            except:
                raise ValueError("{} is not a valid label".format(self._file))
        return self._label

    @property
    def instrument_id(self):
        """
        Returns the short name of the instrument

        Returns
        -------
        : str
          instrument id
        """
        return self.label['IsisCube']['Instrument']['InstrumentId']

    @property
    def platform_name(self):
        """
        Returns the name of the platform containing the sensor. This is usually
        the spacecraft name.

        Returns
        -------
        : str
          Name of the platform which the sensor is mounted on
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft
        Returns
        -------
        : str
        Full name of the spacecraft
        """
        return self.platform_name

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label['IsisCube']['Instrument']['InstrumentName']

    @property
    def sensor_model_version(self):
        """
        Returns the ISIS camera version

        Returns
        -------
        : int
          Camera version number
        """
        return self.label["IsisCube"]["Kernels"]["CameraVersion"]

    @property
    def image_lines(self):
        """
        Returns an integer containing the number of lines in the image

        Returns
        -------
        : int
          Number of lines in image
        """
        return self.label['IsisCube']['Core']['Dimensions']['Lines']

    @property
    def image_samples(self):
        """
        Returns an integer containing the number of samples in the image

        Returns
        -------
        : int
          Number of samples in image
        """
        return self.label['IsisCube']['Core']['Dimensions']['Samples']

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label. For example a return value of 2
        indicates that 2 lines and 2 samples (4 pixels) were summed and divided by 4
        to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        try:
            summing = self.label['IsisCube']['Instrument']['SummingMode']
        except:
            summing = 1
        return summing

    @property
    def sample_summing(self):
        """
        Returns the number of detector samples summed to produce each image sample

        Returns
        -------
        : int
          Sample summing
        """
        return self.sampling_factor

    @property
    def line_summing(self):
        """
        the number of detector lines summed to produce each image sample

        Returns
        -------
        : int
          Line summing
        """
        return self.sampling_factor

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
        if not hasattr(self, "_clock_start_count"):
            if 'SpacecraftClockStartCount' in self.label['IsisCube']['Instrument']:
                self._clock_start_count = self.label['IsisCube']['Instrument']['SpacecraftClockStartCount']
            elif 'SpacecraftClockCount' in self.label['IsisCube']['Instrument']:
                self._clock_start_count = self.label['IsisCube']['Instrument']['SpacecraftClockCount']
            elif 'SpacecraftClockStartCount' in self.label['IsisCube']['Archive']:
                self._clock_start_count = self.label['IsisCube']['Instrument']['SpacecraftClockStartCount']
            else:
                self._clock_start_count = None

            if isinstance(self._clock_start_count, pvl.Quantity):
                self._clock_start_count = self._clock_start_count.value

            self._clock_start_count = str(self._clock_start_count)

        return self._clock_start_count

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
        if not hasattr(self, "_clock_stop_count"):
            if 'SpacecraftClockStopCount' in self.label['IsisCube']['Instrument']:
                self._clock_stop_count = self.label['IsisCube']['Instrument']['SpacecraftClockStopCount']
            elif 'SpacecraftClockStopCount' in self.label['IsisCube']['Archive']:
                self._clock_stop_count = self.label['IsisCube']['Archive']['SpacecraftClockStopCount']
            else:
                self._clock_stop_count = None

            if isinstance(self._clock_stop_count, pvl.Quantity):
                self._clock_stop_count = self._clock_stop_count.value

            self._clock_stop_count = str(self._clock_stop_count)

        return self._clock_stop_count

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
    def exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        if 'ExposureDuration' in self.label['IsisCube']['Instrument']:
            exposure_duration = self.label['IsisCube']['Instrument']['ExposureDuration']
            # Check for units on the PVL keyword
            if isinstance(exposure_duration, pvl.collections.Quantity):
                units = exposure_duration.units
                if "ms" in units.lower() or 'milliseconds' in units.lower():
                    exposure_duration = exposure_duration.value * 0.001
                else:
                    # if not milliseconds, the units are probably seconds
                    exposure_duration = exposure_duration.value
            else:
                # if no units are available, assume the exposure duration is given in milliseconds
                exposure_duration = exposure_duration * 0.001
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
        line_exposure_duration = self.label['IsisCube']['Instrument']['LineExposureDuration']
        if isinstance(line_exposure_duration, pvl.collections.Quantity):
            units = line_exposure_duration.units
            if "ms" in units.lower():
                line_exposure_duration = line_exposure_duration.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                line_exposure_duration = line_exposure_duration.value
        else:
            # if no units are available, assume the exposure duration is given in milliseconds
            line_exposure_duration = line_exposure_duration * 0.001
        return line_exposure_duration


    @property
    def interframe_delay(self):
        """
        The interframe delay in seconds

        Returns
        -------
        : float
          interframe delay in seconds
        """
        interframe_delay = self.label['IsisCube']['Instrument']['InterframeDelay']
        if isinstance(interframe_delay, pvl.collections.Quantity):
            units = interframe_delay.units
            if "ms" in units.lower():
                interframe_delay = interframe_delay.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                interframe_delay = interframe_delay.value
        else:
            # if no units are available, assume the interframe delay is given in milliseconds
            interframe_delay = interframe_delay * 0.001

        return interframe_delay
