import spiceypy as spice

import ale
from ale.base.base import Driver
from ale.base.type_distortion import NoDistortion
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
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
        return 160

    @property
    def detector_start_line(self):
        """
        This is a band dependent value and is currently defaulting to band 1
        """
        return 112

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']


class OdyThemisIrIsisLabelIsisSpiceDriver(LineScanner, IsisLabel, IsisSpice, NoDistortion, Driver):
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
    def original_band(self):
        original_band_key = self.label['IsisCube']['BandBin']['OriginalBand']
        # If there is more than 1 band, use the first one
        try:
            original_band = next(iter(original_band_key))
        except TypeError:
            original_band = original_band_key
        return original_band

    @property
    def detector_line(self):
        no_tdi_lines = {
            1  : 9,
            2  : 24,
            3  : 52,
            4  : 77,
            5  : 102,
            6  : 129,
            7  : 155,
            8  : 181,
            9  : 206,
            10 : 232
        }
        tdi_lines = {
            1  : 8.5,
            2  : 24.5,
            3  : 50.5,
            4  : 76.5,
            5  : 102.5,
            6  : 128.5,
            7  : 154.5,
            8  : 180.5,
            9  : 205.5,
            10 : 231.5
        }
        if self.label['IsisCube']['Instrument']['TimeDelayIntegration'].upper() == "ENABLED":
            return tdi_lines[self.original_band]
        else:
            return no_tdi_lines[self.original_band]

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
        band_offset = (self.detector_line - 0.5) * self.line_exposure_duration / self.line_summing

        return og_start_time + offset + band_offset

    @property
    def focal_length(self):
        return 202.059

    @property
    def detector_center_line(self):
        return 0

    @property
    def detector_center_sample(self):
        emp_offsets = {
            1  :  0.021,
            2  :  0.027,
            3  :  0.005,
            4  :  0.005,
            5  :  0.0,
            6  : -0.007,
            7  : -0.012,
            8  : -0.039,
            9  : -0.045,
            10 :  0.0
        }
        return 160 - emp_offsets[self.original_band]

    @property
    def detector_start_line(self):
        emp_offsets = {
            1  : -0.076,
            2  : -0.098,
            3  : -0.089,
            4  : -0.022,
            5  : 0.0,
            6  : -0.020,
            7  : -0.005,
            8  : -0.069,
            9  :  0.025,
            10 : 0.0
        }
        return 120 - self.detector_line + emp_offsets[self.original_band]

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
