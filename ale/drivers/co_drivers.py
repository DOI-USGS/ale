import os
from glob import glob

import numpy as np

import pvl
import spiceypy as spice
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion, NoDistortion
from ale.base.type_sensor import Framer
from ale.base.type_sensor import LineScanner

from ale.rotation import ConstantRotation
from ale.transformation import FrameChain
from ale.util import query_kernel_pool
from scipy.spatial.transform import Rotation

vims_id_lookup = {
    "VIMS_VIS" : "CASSINI_VIMS_V",
    "VIMS_IR" : "CASSINI_VIMS_IR"
}

vims_name_lookup = {
    "VIMS" : "Visible and Infrared Mapping Spectrometer",
}

iss_id_lookup = {
    "ISSNA" : "CASSINI_ISS_NAC",
    "ISSWA" : "CASSINI_ISS_WAC"
}

iss_name_lookup = {
    "ISSNA" : "Imaging Science Subsystem Narrow Angle Camera",
    "ISSWA" : "Imaging Science Subsystem Wide Angle Camera"
}

spacecraft_name_lookup = {
    'Cassini-Huygens': 'Cassini'
}

nac_filter_to_focal_length = {
    ("P0","BL2"):2002.19,
    ("P0","CB1"):2002.30,
    ("P0","GRN"):2002.38,
    ("P0","IR1"):2002.35,
    ("P0","MT1"):2002.40,
    ("P0","UV3"):2002.71,
    ("P60","BL2"):2002.13,
    ("P60","CB1"):2002.18,
    ("P60","GRN"):2002.28,
    ("P60","IR1"):2002.36,
    ("P60","MT1"):2002.34,
    ("P60","UV3"):2002.51,
    ("RED","GRN"):2002.61,
    ("RED","IR1"):2002.48,
    ("UV1","CL2"):2003.03,
    ("UV2","CL2"):2002.91,
    ("UV2","UV3"):2002.90,
    ("RED","CL2"):2002.69,
    ("CL1","IR3"):2002.65,
    ("CL1","BL2"):2002.37,
    ("CL1","CB1"):2002.66,
    ("CL1","CB2"):2002.66,
    ("CL1","CB3"):2002.68,
    ("CL1","MT1"):2002.88,
    ("CL1","MT2"):2002.91,
    ("CL1","MT3"):2002.87,
    ("CL1","UV3"):2003.09,
    ("HAL","CL2"):2002.94,
    ("IR2","CL2"):2002.71,
    ("IR2","IR1"):2002.56,
    ("IR2","IR3"):2002.55,
    ("IR4","CL2"):2002.89,
    ("IR4","IR3"):2002.81,
    ("BL1","CL2"):2002.79,
    ("CL1","CL2"):2002.88,
    ("CL1","GRN"):2002.75,
    ("CL1","IR1"):2002.74,
    ("IRP0","CB2"):2002.48,
    ("IRP0","CB3"):2002.74,
    ("IRP0","IR1"):2002.60,
    ("IRP0","IR3"):2002.48,
    ("IRP0","MT2"):2002.72,
    ("IRP0","MT3"):2002.72,
    ("P120","BL2"):2002.11,
    ("P120","CB1"):2002.28,
    ("P120","GRN"):2002.38,
    ("P120","IR1"):2002.39,
    ("P120","MT1"):2002.54,
    ("P120","UV3"):2002.71
}

wac_filter_to_focal_length = {
    ("B2","CL2"):200.85,
    ("B2","IRP90"):200.83,
    ("B2","IRP0"):200.82,
    ("B3","CL2"):201.22,
    ("B3","IRP90"):201.12,
    ("B3","IRP0"):201.11,
    ("L1","BL1"):200.86,
    ("L1","CL2"):200.77,
    ("L1","GRN"):200.71,
    ("L1","HAL"):200.74,
    ("L1","IR1"):200.80,
    ("L1","RED"):200.74,
    ("L1","VIO"):201.09,
    ("R2","CL2"):200.97,
    ("R2","IR"):200.95,
    ("R2","IRP90"):200.95,
    ("R3","CL2"):201.04,
    ("R3","IRP90"):201.03,
    ("R3","IRP0"):201.04,
    ("R4","CL2"):201.22,
    ("R4","IRP90"):201.16,
    ("R4","IRP0"):201.15,
    ("T2","CL2"):200.82,
    ("T2","IRP0"):200.81,
    ("T2","IRP90"):200.82,
    ("T3","CL2"):201.04,
    ("T3","IRP0"):201.06,
    ("T3","IRP90"):201.07
}

class CassiniIssIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, RadialDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns an instrument id for unquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire instrument kernel (IK) NAIF IDs.
        Therefore they use the same NAIF ID asin bods2c calls. Expects instrument_id to be
        defined from a mixin class. This should return a string containing either 'ISSNA' or
        'ISSWA'

        Returns
        -------
        : str
          instrument id
        """
        return iss_id_lookup[super().instrument_id]

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.

        Returns
        -------
        : str
          Name of the spacecraft
        """
        return 'CASSINI'

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return iss_name_lookup[super().instrument_id]

    @property
    def ephemeris_start_time(self):
        """
        Returns the start and stop ephemeris times for the image.

        Returns
        -------
        : float
          start time
        """
        return spice.str2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))[0]

    @property
    def center_ephemeris_time(self):
        """
        Returns the starting ephemeris time as the ssi framers center is the
        start.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        center_time = self.ephemeris_start_time + (self.exposure_duration / 2.0)
        return center_time

    @property
    def odtk(self):
        """
        The radial distortion coeffs are not defined in the ik kernels, instead
        they are defined in the ISS Data User Guide (Knowles). Therefore, we
        manually specify the codes here.
        Expects instrument_id to be defined. This should be a string containing either
        CASSINI_ISS_WAC or CASSINI_ISIS_NAC

        Returns
        -------
        : list<float>
          radial distortion coefficients
        """
        if self.instrument_id == 'CASSINI_ISS_WAC':
            # WAC
            return [0, float('-6.2e-5'), 0]
        elif self.instrument_id == 'CASSINI_ISS_NAC':
            # NAC
            return [0, float('-8e-6'), 0]

    @property
    def focal_length(self):
        """
        NAC uses multiple filter pairs, each filter combination has a different focal length.
        NAIF's Cassini kernels do not contain focal lengths for NAC filters and
        so we acquired updated NAC filter data from ISIS's IAK kernel.

        """
        # default focal defined by IAK kernel
        if not hasattr(self, "_focal_length"):
            try:
                default_focal_len = super(CassiniIssPds3LabelNaifSpiceDriver, self).focal_length
            except:
                default_focal_len = float(spice.gdpool('INS{}_DEFAULT_FOCAL_LENGTH'.format(self.ikid), 0, 2)[0])

            filters = tuple(self.label["IsisCube"]["BandBin"]['FilterName'].split("/"))

            if self.instrument_id == "CASSINI_ISS_NAC":
                self._focal_length = nac_filter_to_focal_length.get(filters, default_focal_len)

            elif self.instrument_id == "CASSINI_ISS_WAC":
                self._focal_length = wac_filter_to_focal_length.get(filters, default_focal_len)
        return self._focal_length

    @property
    def _original_naif_sensor_frame_id(self):
        """
        Original sensor frame ID as defined in Cassini's IK kernel. This
        is the frame ID you want to default to for WAC. For NAC, this Frame ID
        sits between J2000 and an extra 180 rotation since NAC was mounted
        upside down.

        Returns
        -------
        : int
          sensor frame code from NAIF's IK kernel
        """
        return self.ikid

    @property
    def sensor_frame_id(self):
        """
        Overwrite sensor frame id to return fake frame ID for NAC representing a
        mounting point with a 180 degree rotation. ID was taken from ISIS's IAK
        kernel for Cassini. This is because NAC requires an extra rotation not
        in NAIF's Cassini kernels. Wac does not require an extra rotation so
        we simply return original sensor frame id for Wac.

        Returns
        -------
        : int
          NAIF's Wac sensor frame ID, or ALE's Nac sensor frame ID
        """
        if self.instrument_id == "CASSINI_ISS_NAC":
            return 14082360
        elif self.instrument_id == "CASSINI_ISS_WAC":
            return 14082361

    @property
    def frame_chain(self):
        """
        Construct the initial frame chain using the original sensor_frame_id
        obtained from the ikid. Then tack on the ISIS iak rotation.

        Returns
        -------
        : Object
          Custom Cassini ALE Frame Chain object for rotation computation and application
        """
        if not hasattr(self, '_frame_chain'):

            try:
                # Call frinfo to check if the ISIS iak has been loaded with the
                # additional reference frame. Otherwise, Fail and add it manually
                _ = spice.frinfo(self.sensor_frame_id)
                self._frame_chain = super().frame_chain
            except spice.utils.exceptions.NotFoundError as e:
                self._frame_chain = FrameChain.from_spice(sensor_frame=self._original_naif_sensor_frame_id,
                                                          target_frame=self.target_frame_id,
                                                          center_ephemeris_time=self.center_ephemeris_time,
                                                          ephemeris_times=self.ephemeris_time,
                                                          exact_ck_times=True)

                rotation = ConstantRotation([[0, 0, 1, 0]], self.sensor_frame_id, self._original_naif_sensor_frame_id)

                self._frame_chain.add_edge(rotation=rotation)

        return self._frame_chain

    @property
    def sensor_model_version(self):
        return 1

class CassiniVimsIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):

    @property
    def vims_channel(self):
        if not hasattr(self, '_vims_channel'):
            self._vims_channel = self.label['IsisCube']["Instrument"]["Channel"]
        return self._vims_channel

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'CTX'

        Returns
        -------
        : str
          instrument id
        """
        return vims_id_lookup[super().instrument_id + "_" + self.vims_channel]

    @property
    def sensor_name(self):
        """
        ISIS doesn't propagate this to the ingested cube label, so hard-code it.
        """
        return vims_name_lookup[super().instrument_id]

    @property
    def spacecraft_name(self):
        """
        Returns the spacecraft name used in various Spice calls to acquire
        ephemeris data.
        Expects the platform_name to be defined. This should be a string of
        the form 'Mars_Reconnaissance_Orbiter'

        Returns
        -------
        : str
          spacecraft name
        """
        return spacecraft_name_lookup[super().platform_name]

    @property
    def exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        if 'ExposureDuration' in self.label['IsisCube']['Instrument']:
            exposure_duration = self.label['IsisCube']['Instrument']['ExposureDuration']

            for i in exposure_duration:
                if i.units == "VIS":
                    exposure_duration = i

            exposure_duration = exposure_duration.value * 0.001
            return exposure_duration
        else:
            return self.line_exposure_duration

    @property
    def focal_length(self):
        """
        Hardcoded value taken from ISIS
        """
        if not hasattr(self, '_focal_length'):
            if self.vims_channel == "VIS":
                self._focal_length = 143.0
            else:
                self._focal_length = 426.0
        return self._focal_length

    @property
    def detector_center_line(self):
        return 0

    @property
    def detector_center_sample(self):
        return 0

    def compute_vims_time(self, line, sample, number_of_samples, mode="VIS"):
        instrument_group = self.label["IsisCube"]["Instrument"]
        time = str(instrument_group["NativeStartTime"])
        int_time, decimal_time = str(time).split(".")

        ephemeris_time = spice.scs2e(self.spacecraft_id, int_time)
        ephemeris_time += float(decimal_time) / 15959.0

        ir_exp = float(instrument_group["ExposureDuration"][0]) * 1.01725 / 1000.0;
        vis_exp = float(instrument_group["ExposureDuration"][1]) / 1000.0

        interline_delay = (float(instrument_group["InterlineDelayDuration"]) * 1.01725) / 1000.0

        swath_width = instrument_group["SwathWidth"];

        if mode == "VIS":
            ephemeris_time = (float(ephemeris_time) + (((ir_exp * swath_width) - vis_exp) / 2.0)) + ((line + 0.5) * vis_exp)
        elif mode == "IR":
            ephemeris_time = float(ephemeris_time) + (line * number_of_samples * ir_exp) + (line * interline_delay) + ((sample + 0.5) * ir_exp)

        return ephemeris_time

    @property
    def ephemeris_start_time(self):
        return self.compute_vims_time(0 - 0.5, 0 - 0.5, self.image_samples, mode=self.vims_channel)

    @property
    def ephemeris_stop_time(self):
        return self.compute_vims_time((self.image_lines - 1) + 0.5, (self.image_samples - 1) + 0.5, self.image_samples, mode=self.vims_channel)

    @property
    def sensor_model_version(self):
        """
        Returns instrument model version
        Returns
        -------
        : int
          ISIS sensor model version
        """
        try:
            return super().sensor_model_version
        except:
            return 1


class CassiniVimsIsisLabelIsisSpiceDriver(LineScanner, IsisLabel, IsisSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'CTX'

        Returns
        -------
        : str
          instrument id
        """

        image_type = self.label['IsisCube']["Instrument"]["Channel"]
        return vims_id_lookup[super().instrument_id + "_" + image_type]

    @property
    def sensor_name(self):
        """
        ISIS doesn't propagate this to the ingested cube label, so hard-code it.
        """
        return "Visible and Infrared Mapping Spectrometer"

    @property
    def spacecraft_name(self):
        """
        Returns the spacecraft name used in various Spice calls to acquire
        ephemeris data.
        Expects the platform_name to be defined. This should be a string of
        the form 'Mars_Reconnaissance_Orbiter'

        Returns
        -------
        : str
          spacecraft name
        """
        return spacecraft_name_lookup[super().platform_name]

    @property
    def exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        if 'ExposureDuration' in self.label['IsisCube']['Instrument']:
            exposure_duration = self.label['IsisCube']['Instrument']['ExposureDuration']

            for i in exposure_duration:
                if i.units == "VIS":
                    exposure_duration = i

            exposure_duration = exposure_duration.value * 0.001
            return exposure_duration
        else:
            return self.line_exposure_duration


class CassiniIssPds3LabelNaifSpiceDriver(Framer, Pds3Label, NaifSpice, RadialDistortion, Driver):
    """
    Cassini mixin class for defining Spice calls.
    """


    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire instrument kernel (IK) NAIF IDs.
        Therefore they use the same NAIF ID asin bods2c calls. Expects instrument_id to be
        defined from a mixin class. This should return a string containing either 'ISSNA' or
        'ISSWA'

        Returns
        -------
        : str
          instrument id
        """
        return iss_id_lookup[super().instrument_id]

    @property
    def focal_epsilon(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : float
          focal epsilon
        """
        return float(spice.gdpool('INS{}_FL_UNCERTAINTY'.format(self.ikid), 0, 1)[0])

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.

        Returns
        -------
        : str
          Name of the spacecraft
        """
        return 'CASSINI'

    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        # Microns to mm
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 1/pixel_size, 0.0]

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 0.0, 1/pixel_size]

    @property
    def odtk(self):
        """
        The radial distortion coeffs are not defined in the ik kernels, instead
        they are defined in the ISS Data User Guide (Knowles). Therefore, we
        manually specify the codes here.
        Expects instrument_id to be defined. This should be a string containing either
        CASSINI_ISS_WAC or CASSINI_ISIS_NAC

        Returns
        -------
        : list<float>
          radial distortion coefficients
        """
        if self.instrument_id == 'CASSINI_ISS_WAC':
            # WAC
            return [0, float('-6.2e-5'), 0]
        elif self.instrument_id == 'CASSINI_ISS_NAC':
            # NAC
            return [0, float('-8e-6'), 0]

    @property
    # FOV_CENTER_PIXEL doesn't specify which coordinate is sample or line, but they are the same
    # number, so the order doesn't matter
    def detector_center_line(self):
        """
        Detector center based on ISIS's corrected values.

        Returns
        -------
        : int
          The detector line of the principle point
        """
        return 512

    @property
    # FOV_CENTER_PIXEL doesn't specify which coordinate is sample or line, but they are the same
    # number, so the order doesn't matter
    def detector_center_sample(self):
        """
        Detector center based on ISIS's corrected values.

        Returns
        -------
        : int
          The detector sample of the principle point
        """
        return 512

    @property
    def sensor_model_version(self):
        """
        Returns instrument model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def focal_length(self):
        """
        NAC uses multiple filter pairs, each filter combination has a different focal length.
        NAIF's Cassini kernels do not contain focal lengths for NAC filters and
        so we acquired updated NAC filter data from ISIS's IAK kernel.

        """
        # default focal defined by IK kernel
        try:
            default_focal_len = super(CassiniIssPds3LabelNaifSpiceDriver, self).focal_length
        except:
            default_focal_len = float(spice.gdpool('INS{}_FOV_CENTER_PIXEL'.format(self.ikid), 0, 2)[0])

        filters = tuple(self.label['FILTER_NAME'])

        if self.instrument_id == "CASSINI_ISS_NAC":
          return nac_filter_to_focal_length.get(filters, default_focal_len)

        elif self.instrument_id == "CASSINI_ISS_WAC":
          return wac_filter_to_focal_length.get(filters, default_focal_len)

    @property
    def _original_naif_sensor_frame_id(self):
        """
        Original sensor frame ID as defined in Cassini's IK kernel. This
        is the frame ID you want to default to for WAC. For NAC, this Frame ID
        sits between J2000 and an extra 180 rotation since NAC was mounted
        upside down.

        Returns
        -------
        : int
          sensor frame code from NAIF's IK kernel
        """
        return self.ikid

    @property
    def sensor_frame_id(self):
        """
        Overwrite sensor frame id to return fake frame ID for NAC representing a
        mounting point with a 180 degree rotation. ID was taken from ISIS's IAK
        kernel for Cassini. This is because NAC requires an extra rotation not
        in NAIF's Cassini kernels. Wac does not require an extra rotation so
        we simply return original sensor frame id for Wac.

        Returns
        -------
        : int
          NAIF's Wac sensor frame ID, or ALE's Nac sensor frame ID
        """
        if self.instrument_id == "CASSINI_ISS_NAC":
          return 14082360
        elif self.instrument_id == "CASSINI_ISS_WAC":
          return 14082361

    @property
    def frame_chain(self):
        """
        Construct the initial frame chain using the original sensor_frame_id
        obtained from the ikid. Then tack on the ISIS iak rotation.

        Returns
        -------
        : Object
          Custom Cassini ALE Frame Chain object for rotation computation and application
        """
        if not hasattr(self, '_frame_chain'):

            try:
                # Call frinfo to check if the ISIS iak has been loaded with the
                # additional reference frame. Otherwise, Fail and add it manually
                _ = spice.frinfo(self.sensor_frame_id)
                self._frame_chain = super().frame_chain
            except spice.utils.exceptions.NotFoundError as e:
                self._frame_chain = FrameChain.from_spice(sensor_frame=self._original_naif_sensor_frame_id,
                                                          target_frame=self.target_frame_id,
                                                          center_ephemeris_time=self.center_ephemeris_time,
                                                          ephemeris_times=self.ephemeris_time,
                                                          exact_ck_times=True)

                rotation = ConstantRotation([[0, 0, 1, 0]], self.sensor_frame_id, self._original_naif_sensor_frame_id)

                self._frame_chain.add_edge(rotation=rotation)

        return self._frame_chain

class CassiniIssIsisLabelIsisSpiceDriver(Framer, IsisLabel, IsisSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns the ID of the instrument

        Returns
        -------
        : str
          ID of the sensor
        """
        return iss_id_lookup[super().instrument_id]

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return iss_name_lookup[super().instrument_id]

    @property
    def center_ephemeris_time(self):
        """
        Returns the middle exposure time for the image in ephemeris seconds.

        This is overridden because the ISIS ISSNAC and ISSWAC sensor models use the
        label utc times so the converted times are not available in the
        NaifKeywords. Instead we get it from the tables.

        Returns
        -------
        : float
        """
        return self.inst_position_table['SpkTableStartTime']

    @property
    def focal_length(self):
        """
        The focal length of the instrument
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keyworkds from the label.
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument.

        Returns
        -------
        float :
            The focal length in millimeters
        """
        filters = self.label["IsisCube"]["BandBin"]['FilterName'].split("/")
        return self.naif_keywords.get('INS{}_{}_{}_FOCAL_LENGTH'.format(self.ikid, filters[0], filters[1]), None)
