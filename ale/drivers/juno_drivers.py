from ale import util
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver

class JunoJunoCamIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading Juno ISIS labels.
    """

    @property
    def instrument_id(self):
        """
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'JNC'

        Returns
        -------
        : str
          instrument id
        """
        look_up = {'JNC': 'JUNO_JUNOCAM'}
        return look_up[super().instrument_id]

    @property
    def ephemeris_start_time(self):
        """
        Junos camera is split into stacked frames where an image is made
        of sets of RGBM chunks. We need to account for these chunks since
        ISIS produces some number of cubes N where N = M*4.
        Computation obtained from JunoCamera.cpp

        Returns
        -------
        : double
          Starting ephemeris time
        """
        if not hasattr(self, '_ephemeris_start_time'):
            initial_time = super().ephemeris_start_time
            frame_number = self.label['IsisCube']['Instrument']['FrameNumber']
            inter_frame_delay = self.label['IsisCube']['Instrument']['InterFrameDelay'].value
            start_time_bias = self.naif_keywords[f'INS{self.ikid}_START_TIME_BIAS']
            inter_frame_delay_bias = self.naif_keywords[f'INS{self.ikid}_INTERFRAME_DELTA']
            self._ephemeris_start_time = initial_time + start_time_bias + (frame_number - 1) * (inter_frame_delay + inter_frame_delay_bias)
        return self._ephemeris_start_time

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def naif_keywords(self):
        """
        Updated set of naif keywords containing the NaifIkCode for the specific
        Juno filter used when taking the image.

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        filter_code = self.label['IsisCube']['BandBin']['NaifIkCode']
        return {**super().naif_keywords, **util.query_kernel_pool(f"*{filter_code}*")}
