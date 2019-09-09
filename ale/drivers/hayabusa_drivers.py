import spiceypy as spice
import struct

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import Framer

class HayabusaAmicaIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, RadialDistortion, Driver):
    """
    Driver for working with Hayabusa 1 AMICA ISIS cubes and SPICE kernels.
    """

    @property
    def sensor_name(self):
        return self.instrument_id

    @property
    def sensor_model_version(self):
        return 1

    @property
    def ikid(self):
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def spacecraft_clock_start_count(self):
        return str(self.label['IsisCube']['Instrument']['SpacecraftClockStartCount'].value)

    @property
    def pixel_size(self):
        return spice.gdpool('INS{}_PIXEL_PITCH'.format(self.ikid), 0, 1)[0]

    @property
    def isis_naif_keywords(self):
        return {
            'BODY_CODE' : self.target_id,
            'BODY_FRAME_CODE' : self.target_frame_id,
            f'BODY{self.target_id}_RADII' : self.target_body_radii,
            f'INS{self.ikid}_SWAP_OBSERVER_TARGET' : str(self.swap_observer_target).upper(),
            f'INS{self.ikid}_LIGHTTIME_CORRECTION' : str(self.light_time_correction).upper(),
            f'INS{self.ikid}_LT_SURFACE_CORRECT' : str(self.correct_lt_to_surface).upper(),
            f'INS{self.ikid}_FOCAL_LENGTH' : self.focal_length,
            f'INS{self.ikid}_PIXEL_PITCH' : self.pixel_size,
            f'CLOCK_ET_{self.spacecraft_id}_{self.spacecraft_clock_start_count}_COMPUTED' : struct.pack('d', self.ephemeris_start_time).hex(),
            f'INS{self.ikid}_TRANSX' : self.pixel2focal_x,
            f'INS{self.ikid}_TRANSY' : self.pixel2focal_y,
            f'INS{self.ikid}_ITRANSS' : self.focal2pixel_samples,
            f'INS{self.ikid}_ITRANSL' : self.focal2pixel_lines,
            f'INS{self.ikid}_BORESIGHT_LINE' : self.detector_center_line,
            f'INS{self.ikid}_BORESIGHT_SAMPLE' : self.detector_center_sample,
            f'INS{self.ikid}_OD_K' : self.odtk
        }
