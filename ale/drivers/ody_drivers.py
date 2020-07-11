import spiceypy as spice

import ale
from ale.base.base import Driver
from ale.base.type_distortion import NoDistortion
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import LineScanner

import pvl

class OdyThemisIrIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for Themis IR ISIS cube
    """
    @property
    def instrument_id(self):
        inst_id = super().instrument_id

        if inst_id not in ["THEMIS_IR"]:
            raise Exception(f"{inst_id} is not a valid THEMIS IR instrument name. Expecting THEMIS_IR")

        return inst_id

    @property
    def sensor_model_version(self):
        return 1

    @property
    def spacecraft_name(self):
        name = super().spacecraft_name.replace('_', ' ')
        if name != "MARS ODYSSEY":
            raise Exception("{name} for label is not a valid Mars Odyssey spacecraft name")
        return name

    @property
    def ikid(self):
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def line_exposure_duration(self):
        """
        returns line exposure duration

        Taken from ISIS ThemisIr Camera Model
        """
        return (33.2871/1000 * self.line_summing)

    @property
    def ephemeris_start_time(self):
        og_start_time = super().ephemeris_start_time
        offset = self.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"]
        if isinstance(offset, pvl._collections.Units):
            units = offset.units
            if "ms" in units.lower():
                offset = offset.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                offset = offset.value

        return og_start_time + offset

    @property
    def focal_length(self):
        return 202.059

    @property
    def detector_center_line(self):
        return 0

    @property
    def detector_center_sample(self):
        return 0
    
    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']


class OdyThemisVisIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    """"
    Driver for Themis VIS ISIS cube
    """

    @property
    def instrument_id(self):
        inst_id = super().instrument_id

        if inst_id not in ["THEMIS_VIS"]:
            raise Exception(f"{inst_id} is not a valid THEMIS VIS instrument name. Expecting \"THEMIS_VIS\"")

        return inst_id

    @property
    def sensor_model_version(self):
        return 1

    @property
    def spacecraft_name(self):
        name = super().spacecraft_name.replace('_', ' ')
        if name != "MARS ODYSSEY":
            raise Exception("{name} for label is not a valid Mars Odyssey spacecraft name")
        return name

    @property
    def ikid(self):
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def ephemeris_start_time(self):
        """
        The starting ephemeris time, in seconds

        Formula derived from ISIS3's ThemisVis Camera model

        Returns
        -------
        : double
          Starting ephemeris time in seconds
        """
        og_start_time = super().ephemeris_start_time

        offset = self.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"]
        if isinstance(offset, pvl._collections.Units):
            units = offset.units
            if "ms" in units.lower():
                offset = offset.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                offset = offset.value

        return og_start_time + offset - (self.line_exposure_duration/2)

    @property
    def line_exposure_duration(self):
        """
        The line exposure duration of the image, in seconds

        Returns
        -------
        : float
          Line exposure duration in seconds
        """
        line_exposure_duration = self.label['IsisCube']['Instrument']['ExposureDuration']
        if isinstance(line_exposure_duration, pvl._collections.Units):
            units = line_exposure_duration.units
            if "ms" in units.lower():
                line_exposure_duration = line_exposure_duration.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                line_exposure_duration = line_exposure_duration.value
        else:
            # if no units are available, assume the exposure duration is given in milliseconds
            line_exposure_duration = line_exposure_duration * 0.001
        return line_exposure_duration

    @property
    def focal_length(self):
        return 202.059

    @property
    def detector_center_line(self):
        return 0

    @property
    def detector_center_sample(self):
        return 0

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']
