import ale
from ale.drivers.base import *
from ale import config

class DawnCamera(Driver, Framer, PDS3, Spice, RadialDistortion):
    """
    """

    @property
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        metakernel_dir = config.dawn
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            for mk in mks:
                if str(self.start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore the
        the same ID that Spice expects in bods2c calls.

        Returns
        -------
        : str
          instrument id
        """
        instrument_id = self.label["INSTRUMENT_ID"]
        filter_number = self.label["FILTER_NUMBER"]

        return "DAWN_{}_FILTER_{}".format(instrument_id, filter_number)

    @property
    def _odtk(self):
        """
        Returns
        -------
        : list
          Radial distortion coefficients
        """
        return spice.gdpool('INS{}_RAD_DIST_COEFF'.format(self.ikid),0, 1).tolist()

    @property
    def label(self):
        """
        Loads a PVL from from the _file attribute and
        parses the binary table data.

        Returns
        -------
        PVLModule :
            Dict-like object with PVL keys
        """
        if not hasattr(self, "_label"):
            if isinstance(self._file, pvl.PVLModule):
                self._label = self._file
            try:
                self._file.replace("\\", "/")
                self._label = pvl.loads(self._file)
            except Exception:
                with open(self._file, 'rb') as fp:
                    lines = [line.decode('utf-8', errors='ignore').replace('\\', '/') for line in fp]
                    slabel = ''.join(lines)
                self._label = pvl.loads(slabel)
            except:
                raise ValueError("{} is not a valid label".format(self._file))
        return self._label

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
        return self.label['INSTRUMENT_HOST_NAME']

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
        target = self.label['TARGET_NAME']
        target = target.split(' ')[-1]
        return target

    @property
    def center_ephemeris_time(self):
        """
        The center ephemeris time for a framer.
        """
        center_time = self.starting_ephemeris_time + self.exposure_duration / 2
        return center_time

    @property
    def focal2pixel_samples(self):
        # Microns to mm
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 1/pixel_size, 0.0]

    @property
    def focal2pixel_lines(self):
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 0.0, 1/pixel_size]
