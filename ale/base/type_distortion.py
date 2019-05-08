class RadialDistortion():
    @property
    def optical_distortion(self):
        return {
            "radial": {
                "coefficients" : self._odtk
            }
        }

class TransverseDistortion():
    @property
    def optical_distortion(self):
        return {
            "transverse": {
                "x" : self._odtx,
                "y" : self._odty
            }
        }
