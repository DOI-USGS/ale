class RadialDistortion():
    @property
    def optical_distortion(self):
        return {
            "radial": {
                "coefficients" : self._odtk
            }
        }    
