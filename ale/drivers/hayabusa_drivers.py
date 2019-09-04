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
