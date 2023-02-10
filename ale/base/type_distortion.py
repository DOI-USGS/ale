import numpy as np

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

class CahvorDistortion():
    """
    Mix-in for sensors and data sets that have a CAHVOR distortion model.

    This model is based on info from https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003JE002199
    Expects that cahvor_camera_dict and focal_length to be defined. This should be
    a dictionary defining the CAHVOR distortion parameters.
    """

    @property
    def usgscsm_distortion_model(self):
        """
        Dictionary containing the usgscsm specification for CAHVOR distortion.
        This will be a list of coeffs for radial distortion (x0, x1, x2)
        followed by the optical center (x, y)

        Returns
        -------
        : dict
        """
        R = [0, 0, 0]
        x = 0
        y = 0
        # If our model contains OR in the CAHVOR model
        # then compute the distortion coeffs/offset
        if (len(self.cahvor_camera_dict.keys()) >= 6):
            A = self.cahvor_camera_dict.get("A", [0, 0, 0])
            H = self.cahvor_camera_dict.get("H", [0, 0, 0])
            V = self.cahvor_camera_dict.get("V", [0, 0, 0])
            O = self.cahvor_camera_dict.get("O", [0, 0, 0])
            i = np.dot(O, H) / np.dot(O, A)
            j = np.dot(O, V) / np.dot(O, A)
            x = self.pixel_size * (i - self.compute_h_c())
            y = self.pixel_size * (self.compute_v_c() - j)
            R = self.cahvor_camera_dict.get("R", [0, 0, 0])
            R[1] /= self.focal_length**2
            R[2] /= self.focal_length**4
        return {
            "cahvor": {
                "coefficients": [*R, x, y]
            }
        }


class ChandrayaanMrffrDistortion():
    @property
    def usgscsm_distortion_model(self):
        transx = [-1* self.scaled_pixel_height, self.scaled_pixel_height, 0.0]
        transy = [0,0,0]
        transs = [1.0, 1.0 / self.scaled_pixel_height, 0.0]
        transl = [0.0, 0.0, 0.0]

        return {
            "ChandrayaanMrffr":{
                "x_coefficients" : transx,
                "y_coefficients" : transy
            }
        }