from ale import util
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.base import Driver

class JunoJunoCamIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, Driver):

    @property
    def instrument_id(self):
        look_up = {'JNC': 'JUNO_JUNOCAM'}
        return look_up[super().instrument_id]

    @property
    def sensor_model_version(self):
        return 1

    @property
    def naif_keywords(self):
        filter_code = self.label['IsisCube']['BandBin']['NaifIkCode']
        return {**super().naif_keywords, **util.query_kernel_pool(f"*{filter_code}*")}
