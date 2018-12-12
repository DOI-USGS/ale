from abc import ABC
import spiceypy as spice


class TransverseDistortion(ABC):
    """
    Exposes the properties that are used to describe a transverse distortion model.
    """
    @property
    def odtx(self):
        return spice.gdpool('INS{}_OD_T_X'.format(self.ikid),0, 10)

    @property
    def odty(self):
        return spice.gdpool('INS{}_OD_T_Y'.format(self.ikid), 0, 10)

class RadialDistortion(ABC):
    """
    Exposes the properties that are used to describe a radial distortion model.
    """
    @property
    def odtk(self):
        return spice.gdpool('INS{}_OD_K'.format(self.ikid),0, 3)
