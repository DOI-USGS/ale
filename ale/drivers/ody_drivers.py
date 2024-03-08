import spiceypy as spice

import ale
from ale.base.base import Driver
from ale.base.type_distortion import ThemisIrDistortion, NoDistortion
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import LineScanner, PushFrame

import pvl

class OdyThemisIrIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, ThemisIrDistortion, Driver):
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
    def sampling_factor(self):
        """
        Returns the summing factor from the ISIS label. For example a return value of 2
        indicates that 2 lines and 2 samples (4 pixels) were summed and divided by 4
        to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        try:
            summing = self.label['IsisCube']['Instrument']['SpatialSumming']
        except:
            summing = 1
        return summing

    @property
    def line_exposure_duration(self):
        """
        returns line exposure duration

        Taken from ISIS ThemisIr Camera Model
        """
        return (33.2871/1000 * self.line_summing)

    @property
    def start_time(self):
        og_start_time = super().ephemeris_start_time
        offset = self.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"]
        if isinstance(offset, pvl.collections.Quantity):
            units = offset.units
            if "ms" in units.lower():
                offset = offset.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                offset = offset.value

        return og_start_time + offset
    

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        return self.band_times[0]
    

    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the number of lines
        times the exposure duration. Expects ephemeris start time, exposure duration
        and image lines to be defined. These should be double precision numbers
        containing the ephemeris start, exposure duration and number of lines of
        the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.band_times[-1] + (self.image_lines * self.line_exposure_duration)

    @property
    def focal_length(self):
        return 203.9213

    @property
    def detector_start_sample(self):
        """
        Returns the starting detector sample for the image.

        Returns
        -------
        : int
          Starting detector sample for the image
        """
        return 160.5
    
    @property
    def detector_center_line(self):
        return 0

    @property
    def detector_center_sample(self):
        return 0
    
    @property
    def tdi_mode(self):
        return self.label['IsisCube']['Instrument']["TimeDelayIntegration"]
    
    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def band_times(self):
        self._num_bands = self.label["IsisCube"]["Core"]["Dimensions"]["Bands"]
        times = []

        org_bands = self.label["IsisCube"]["BandBin"]["FilterNumber"]

        for vband in range(self._num_bands):
            if 'ReferenceBand' in self.label['IsisCube']['Instrument']:
                band = self.label['IsisCube']['Instrument']['ReferenceBand']
            else:
                if isinstance(org_bands, (list, tuple)):
                    band = org_bands[vband]
                else:
                    band = org_bands

            if (self.tdi_mode == "ENABLED"):
                band_tdi = [8.5, 24.5, 50.5, 76.5, 102.5,
                                   128.5, 154.5, 180.5, 205.5, 231.5]
                detector_line = band_tdi[band-1]
            else:
                band_no_tdi = [9, 24, 52, 77, 102, 129, 155, 181, 206, 232]
                detector_line = band_no_tdi[band-1]

            band_offset = (detector_line - 0.5) * self.line_exposure_duration
            band_offset /= self.sampling_factor

            time = self.start_time + band_offset
            times.append(time)
        return times


class OdyThemisVisIsisLabelNaifSpiceDriver(PushFrame, IsisLabel, NaifSpice, NoDistortion, Driver):
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
    def start_time(self):
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
        if isinstance(offset, pvl.collections.Quantity):
            units = offset.units
            if "ms" in units.lower():
                offset = offset.value * 0.001
            else:
                # if not milliseconds, the units are probably seconds
                offset = offset.value

        return og_start_time + offset - (self.exposure_duration / 2)

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        return self.start_time + self.band_offset[0]
    
    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris start time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        return super().ephemeris_stop_time + self.band_offset[-1] + self.interframe_delay

    @property
    def focal_length(self):
        return 202.059

    @property
    def detector_center_line(self):
        return 512

    @property
    def detector_center_sample(self):
        return 512
    
    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def num_frames(self):
        """
        Number of frames in the image

        Returns
        -------
        : int
          Number of frames in the image
        """
        return self.label['IsisCube']['Instrument']['NumFramelets']

    @property
    def framelet_height(self):
        return self.image_lines / (self.num_frames / self.sampling_factor)
    
    @property
    def sampling_factor(self):
        return self.label['IsisCube']['Instrument']['SpatialSumming']
    
    @property
    def interframe_delay(self):
        return self.label['IsisCube']['Instrument']['InterframeDelay']
    
    @property
    def band_offset(self):
        self._num_bands = self.label["IsisCube"]["Core"]["Dimensions"]["Bands"]
        band_offsets = []

        org_bands = self.label["IsisCube"]["BandBin"]["FilterNumber"]

        for vband in range(self._num_bands):
            if isinstance(org_bands, (list, tuple)):
                timeband = org_bands[vband]
            else:
                timeband = org_bands

            if 'ReferenceBand' in self.label['IsisCube']['Instrument']:
                ref_band = self.label['IsisCube']['Instrument']['ReferenceBand']
                wavelength_to_timeband = [2, 5, 3, 4, 1]
                timeband = wavelength_to_timeband[ref_band - 1]

            band_offset = ((timeband - 1) * self.interframe_delay) - (self.exposure_duration / 2.0)
            
            band_offsets.append(band_offset)
        return band_offsets
    
    @property
    def framelets_flipped(self):
        return True