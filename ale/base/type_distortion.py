class RadialDistortion():
    @property
    def usgscsm_distortion_model(self):
        return {
            "radial": {
                "coefficients" : self._odtk
            }
        }
