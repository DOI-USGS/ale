import pvl

class Pds3Label():
    """
    Mix-in for parsing PDS3 PVL labels.
    """

    @property
    def label(self):
        if not hasattr(self, "_label"):
            if isinstance(self._file, pvl.PVLModule):
                self._label = self._file
            else:
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
        Returns the short name of the instrument

         Returns
        -------
        : str
          Short name of the instrument
        """
        return self.label['INSTRUMENT_ID']


    @property
    def instrument_name(self):
        """
        Returns the full name of the instrument

          Returns
        -------
        : str
          Full name of the instrument
        """
        return self.label['INSTRUMENT_NAME']



    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.instrument_name


    @property
    def instrument_host_id(self):
        """
        Returns the short name of the instrument host

          Returns
        -------
        : str
          Short name of the instrument host
        """
        return self.label['INSTRUMENT_HOST_ID']


    @property
    def instrument_host_name(self):
        """
        Returns the full name of the instrument host

          Returns
        -------
        : str
          Full name of the instrument host
        """
        return self.label['INSTRUMENT_HOST_NAME']


    @property
    def platform_name(self):
        """
        Returns the name of the platform which the instrument is mounted on

        Returns
        -------
        : str
          platform name
        """
        return self.instrument_host_name


    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft

          Returns
        -------
        : str
          Full name of the spacecraft
        """
        return self.label['SPACECRAFT_NAME']


    @property
    def utc_start_time(self):
        """
        Returns the start time of the image as a UTC string

          Returns
        -------
        : str
          Start time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        return self.label['START_TIME']


    @property
    def utc_stop_time(self):
        """
        Returns the stop time of the image as a UTC string

          Returns
        -------
        : str
          Stop time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        return self.label['STOP_TIME']


    @property
    def image_lines(self):
        """
          Returns
        -------
        : int
          Number of lines in the image
        """
        return self.label['IMAGE']['LINES']


    @property
    def image_samples(self):
        """
        Returns
        -------
        : int
          Number of samples in the image
        """
        return self.label['IMAGE']['LINE_SAMPLES']


    @property
    def target_name(self):
        """
        Returns a target name uniquely identifying what an observation was capturing.
        This is most often a body name (e.g., Mars, Moon, Europa). This value is often
        use to acquire Ephemeris data from SPICE files; therefore it should be the same
        name spicelib expects in bodvrd calls.

        Returns
        -------
        : str
          Target name
        """
        return self.label['TARGET_NAME']


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
        return self.label.get('SAMPLING_FACTOR', 1)


    @property
    def line_summing(self):
        """
        Expects sampling_factor to be defined. This must be an integer
        containing the number of samples and lines combined from the original data

        Returns
        -------
        : int
           Number of detector lines summed to produce each image line
        """
        return self.sampling_factor


    @property
    def sample_summing(self):
        """
        Expects sampling_factor to be defined. This must be an integer
        containing the number of samples and lines combined from the original data

        Returns
        -------
        : int
           Number of detector lines summed to produce each image line
        """
        return self.sampling_factor


    @property
    def downtrack_summing(self):
        """
        Returns the number of detector pixels (normally in the line direction) that
        have been averaged to produce the output pixel

        Returns
        -------
        : int
          Number of downtrack pixels summed together
        """
        return self.label.get('DOWNTRACK_SUMMING', 1)


    @property
    def crosstrack_summing(self):
        """
        Returns the number of detector pixels (normally in the sample direction) that
        have been averaged to produce the output pixel

        Returns
        -------
        : int
          Number of crosstrack pixels summed together
        """
        return self.label.get('CROSSTRACK_SUMMING', 1)



    @property
    def spacecraft_clock_start_count(self):
        """
        Returns
        -------
        : str
          Returns the start clock count string from the PDS3 label.
        """
        return self.label['SPACECRAFT_CLOCK_START_COUNT']


    @property
    def spacecraft_clock_stop_count(self):
        """
        Returns
        -------
        : str
          Returns the stop clock count string from the PDS3 label.
        """
        count = self.label['SPACECRAFT_CLOCK_STOP_COUNT']
        if count == 'N/A':
            count = None
        return count


    @property
    def exposure_duration(self):
        """
        Returns the exposure duration converted to seconds. If the exposure duration
        is not present in the PDS3 label, then this property returns the
        line exposure duration. Expects line_exposure_duration to be defined. This
        should be a floating point number containing the line exposure duration.

         Returns
         -------
         : float
           Returns the exposure duration in seconds from the PDS3 label.
         """
        # The EXPOSURE_DURATION may either be stored as a (value, unit) or just a value
        if 'EXPOSURE_DURATION' in self.label:
            try:
                unit = self.label['EXPOSURE_DURATION'].units
                unit = unit.lower()
                if unit == "ms" or unit == "msec" or unit == "millisecond":
                  return self.label['EXPOSURE_DURATION'].value * 0.001
                else:
                  return self.label['EXPOSURE_DURATION'].value

            # With no units, assume milliseconds
            except:
                return self.label['EXPOSURE_DURATION'] * 0.001
        else:
            return self.line_exposure_duration


    # Consider expanding this to handle units
    @property
    def line_exposure_duration(self):
        """
        Line exposure duration returns the time between the exposures for
        subsequent lines.

        Returns
        -------
        : float
          Returns the line exposure duration in seconds from the PDS3 label.
        """
        return self.label['LINE_EXPOSURE_DURATION'].value * 0.001  # Scale to seconds

    @property
    def filter_number(self):
        """
        Returns
        -------
        : int
          Returns the filter number from the PDS3 label
        """
        return self.label['FILTER_NUMBER']
