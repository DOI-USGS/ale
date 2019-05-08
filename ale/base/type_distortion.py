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

class LROLROCNACDistortion():
    """
    LRO LROC NAC does not use the default distortion model so we need to overwrite the
    method packing the distortion model into the ISD.
    """
    @property
    def optical_distortion(self):
        return {
            "lrolrocnac": {
                "coefficients": self._odtk
                }
            }
        
class KaguyaTCDistortion():
    """
    Kaguya uses a unique radial distortion model so we need to overwrite the
    method packing the distortion model into the ISD.
    from the IK:
    Line-of-sight vector of pixel no. n can be expressed as below.
    Distortion coefficients information:
    INS<INSTID>_DISTORTION_COEF_X  = ( a0, a1, a2, a3)
    INS<INSTID>_DISTORTION_COEF_Y  = ( b0, b1, b2, b3),
    Distance r from the center:
    r = - (n - INS<INSTID>_CENTER) * INS<INSTID>_PIXEL_SIZE.
    Line-of-sight vector v is calculated as
    v[X] = INS<INSTID>BORESIGHT[X] + a0 + a1*r + a2*r^2 + a3*r^3 ,
    v[Y] = INS<INSTID>BORESIGHT[Y] + r+a0 + a1*r +a2*r^2 + a3*r^3 ,
    v[Z] = INS<INSTID>BORESIGHT[Z]
    """
    @property
    def optical_distortion(self):
        return {
            "kaguyatc": {
                "x" : self._odkx,
                "y" : self._odky
            }
        }


class DawnFCDistortion():
    @property
    def optical_distortion(self):
        return {
            "dawnfc": {
                "coefficients" : self._odtk
            }
        }

