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
    def _focal_plane_tempature(self):
        return self.label['FOCAL_PLANE_TEMPERATURE'].value

    @property
    def instrument_id(self):
        pass

    @property
    def start_time(self):
        return self.label['START_TIME']

    @property
    def image_lines(self):
        return self.label['IMAGE']['LINES']

    @property
    def image_samples(self):
        return self.label['IMAGE']['LINE_SAMPLES']

    @property
    def interpolation_method(self):
        return 'lagrange'

    @property
    def target_name(self):
        """
        Returns an target name for unquely identifying the instrument, but often
        piped into Spice Kernels to acquire Ephermis data from Spice. Therefore they
        the same ID the Spice expects in bodvrd calls.

        Returns
        -------
        : str
          target name
        """

        return self.label['TARGET_NAME']

    @property
    def _target_id(self):
        return spice.bodn2c(self.label['TARGET_NAME'])

    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            sclock = self.label['SPACECRAFT_CLOCK_START_COUNT']
            self._starting_ephemeris_time = spice.scs2e(self.spacecraft_id, sclock)
        return self._starting_ephemeris_time

    @property
    def spacecraft_clock_stop_count(self):
        sc = self.label.get('SPACECRAFT_CLOCK_STOP_COUNT', None)
        if sc == 'N/A':
            sc = None
        return sc

    @property
    def _detector_center_line(self):
        return spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[0]

    @property
    def _detector_center_sample(self):
        return spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[1]

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
        return self.label['MISSION_NAME']

    @property
    def detector_line_summing(self):
        return self.label.get('SAMPLING_FACTOR', 1)

    @property
    def _exposure_duration(self):
        # The EXPOSURE_DURATION may either be stored as a (value, unit) or just a value
        try:
            return self.label['EXPOSURE_DURATION'].value * 0.001
        except:
            return self.label['EXPOSURE_DURATION'] * 0.001
