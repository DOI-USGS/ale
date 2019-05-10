import pvl
import spiceypy as spice

class Pds3Label():
    """
    Mixin for reading from PDS3 Labels.

    Attributes
    ----------
    _label : PVLModule
             Dict-like object with PVL keys

    """

    @property
    """
    Returns the number of lines in the image.

    Returns
    -------
    : int
      number of lines
    """
    def image_lines(self):
        return self.label['IMAGE']['LINES']


    @property
    """
    Returns the number of samples in the image.

    Returns
    -------
    : int
      number of samples
    """
    def image_samples(self):
        return self.label['IMAGE']['LINE_SAMPLES']


    @property
    def target_name(self):
        """
        Returns a target name unquely identifying what an observation was capturing. 
        This is most often a body name (e.g., Mars, Moon, Europa). This value is often
        use to acquire Ephermis data from SPICE files; therefore it should be the same
        ID spicelib expects in bodvrd calls.

        Returns
        -------
        : str
          target name
        """

        return self.label['TARGET_NAME']


    @property
    """
    Returns the summing factor from the PDS3 label. For example a return value of 2 
    indicates that 2 lines and 2 samples (4 pixels) were summed and divided by 4
    to produce the output pixel value.

    Returns
    -------
    : int
      number of samples and lines combined from the original data to produce a single pixel in this image
    """
    def detector_summing(self):
        return self.label.get('SAMPLING_FACTOR', 1)


    @property
    """
    Returns
    -------
    : str
      Returns the start clock count string from the PDS3 label.
    """
    def spacecraft_start_clock_count(self):
        return self.label['SPACECRAFT_CLOCK_START_COUNT']


    @property
    """
    Returns
    -------
    : str
      Returns the stop clock count string from the PDS3 label.
    """
    def spacecraft_stop_clock_count(self):
        count = self.label['SPACECRAFT_CLOCK_STOP_COUNT', None]
        if count == 'N/A':
            count = None
        return count


    @property
    """
     Returns
     -------
     : float
       Returns the exposure duration in seconds from the PDS3 label.
     """
    def exposure_duration(self):
        # The EXPOSURE_DURATION may either be stored as a (value, unit) or just a value
        try:
            unit = self.label['EXPOSURE_DURATION'].unit.lower()
            if unit == "ms":
              return self.label['EXPOSURE_DURATION'].value * 0.001
            else:
              return self.label['EXPOSURE_DURATION'].value

        # With no units, assume seconds
        except:
            return self.label['EXPOSURE_DURATION'] * 0.001



