class LegendreDistortion():
    """
    Mix-in for sensors that use a legendre distortion model.
    """

    @property
    def usgscsm_distortion_model(self):
        """
        Expects odtx and odty to be defined. These should be lists containing
        the legendre distortion coefficients

        Returns
        -------
        : dict
          Dictionary containing the usgscsm distortion model
        """
        return {
            "legendre":{
                "x_coefficients" : self.odtx,
                "y_coefficients" : self.odty
            }
        }


class RadialDistortion():
    """
    Mix-in for sensors that use a radial distortion model.
    """

    @property
    def usgscsm_distortion_model(self):
        """
        Expects odtk to be defined. This should be a list containing
        the radial distortion coefficients

        Returns
        -------
        : dict
          Dictionary containing the usgscsm distortion model
        """
        return {
            "radial": {
                "coefficients" : self.odtk
            }
        }


class NoDistortion():
    """
    Mix-in for sensors and data sets that do not have a distortion model.
    """

    @property
    def usgscsm_distortion_model(self):
        """
        Returns the specification for no distortion in usgscsm.

        Returns
        -------
        : dict
          Dictionary containing the usgscsm specification for no distortion.
        """
        return {"radial": {"coefficients": [0.0, 0.0, 0.0]}}

class KaguyaSeleneDistortion():
    """
    Mix-in for sensors on the Kaguya/Selene mission. 
    """

    @property
    def usgscsm_distortion_model(self):
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

        Expects odkx and odky to be defined. These should be a list of optical
        distortion x and y coefficients respectively.

        Returns
        -------
        : dict
            radial distortion model

        """
        return {
            "kaguyalism": {
                "x" : self._odkx,
                "y" : self._odky,
                "boresight_x" : self.boresight_x,
                "boresight_y" : self.boresight_y
            }
        }
