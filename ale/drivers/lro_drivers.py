import os
import re
import numpy as np
import pvl
import spiceypy as spice
from glob import glob

from ale.util import get_metakernels, query_kernel_pool
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import LineScanner, Radar, PushFrame
from ale.base.type_distortion import RadialDistortion

class LroLrocNacPds3LabelNaifSpiceDriver(LineScanner, NaifSpice, Pds3Label, Driver):
    """
    Driver for reading LROC NACL, NACR (not WAC, it is a push frame) labels. Requires a Spice mixin to
    acquire additional ephemeris and instrument data located exclusively in SPICE kernels, A PDS3 label,
    and the LineScanner and Driver bases.
    """

    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """

        instrument = super().instrument_id

        frame_id = self.label.get("FRAME_ID")

        if instrument == "LROC" and frame_id == "LEFT":
            return "LRO_LROCNACL"
        elif instrument == "LROC" and frame_id == "RIGHT":
            return "LRO_LROCNACR"

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. LROC NAC img PDS3 labels do not the have SPACECRAFT_NAME keyword, so we
        override it here to use the label_pds3 property for instrument_host_id

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_id

    @property
    def sensor_model_version(self):
        """
        Returns ISIS instrument sensor model version number

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def usgscsm_distortion_model(self):
        """
        The distortion model name with its coefficients

        LRO LROC NAC does not use the default distortion model so we need to overwrite the
        method packing the distortion model into the ISD.

        Returns
        -------
        : dict
          Returns a dict with the model name : dict of the coefficients
        """

        return {"lrolrocnac":
                {"coefficients": self.odtk}}

    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients. There is only one coefficient for LROC NAC l/r
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid), 0, 1).tolist()

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correction and aberration correction to
        use in NAIF calls.

        LROC is specifically set to not use light time correction because it is
        so close to the surface of the moon that light time correction to the
        center of the body is incorrect.

        Returns
        -------
        : str
          The light time and aberration correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'

    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels
        ISIS uses 0.5 based CCD samples, so we need to convert to 0 based.

        Returns
        -------
        float :
            The center sample of the CCD
        """
        return super().detector_center_sample - 0.5

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of
        the instrument. For LROC NAC this is flipped depending on the spacecraft
        direction.

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        focal2pixel_lines = np.array(list(spice.gdpool('INS{}_ITRANSL'.format(self.ikid), 0, 3))) / self.sampling_factor
        if self.spacecraft_direction < 0:
            return -focal2pixel_lines
        else:
            return focal2pixel_lines

    @property
    def ephemeris_start_time(self):
        """
        The starting ephemeris time for LRO is computed by taking the
        LRO:SPACECRAFT_CLOCK_PREROLL_COUNT, as defined in the label, and
        adding offsets that were taken from an IAK.

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        start_time = spice.scs2e(self.spacecraft_id, self.label['LRO:SPACECRAFT_CLOCK_PREROLL_COUNT'])
        return start_time + self.constant_time_offset + self.additional_preroll * self.exposure_duration

    @property
    def exposure_duration(self):
        """
        Takes the exposure_duration defined in a parent class and adds
        offsets taken from an IAK.

        Returns
        -------
        : float
          Returns the exposure duration in seconds.
        """
        return super().exposure_duration * (1 + self.multiplicative_line_error) + self.additive_line_error

    @property
    def multiplicative_line_error(self):
        """
        Returns the multiplicative line error defined in an IAK.

        Returns
        -------
        : float
          Returns the multiplicative line error.
        """
        return 0.0045

    @property
    def additive_line_error(self):
        """
        Returns the additive line error defined in an IAK.

        Returns
        -------
        : float
          Returns the additive line error.
        """
        return 0.0

    @property
    def constant_time_offset(self):
        """
        Returns the constant time offset defined in an IAK.

        Returns
        -------
        : float
          Returns the constant time offset.
        """
        return 0.0

    @property
    def additional_preroll(self):
        """
        Returns the addition preroll defined in an IAK.

        Returns
        -------
        : float
          Returns the additional preroll.
        """
        return 1024.0

    @property
    def mission_name(self):
        return self.label['MISSION_NAME']


    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label that is defined by the CROSSTRACK_SUMMING.
        For example a return value of 2 indicates that 2 lines and 2 samples (4 pixels)
        were summed and divided by 4 to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        return self.crosstrack_summing

    @property
    def spacecraft_direction(self):
        """
        Returns the x axis of the first velocity vector relative to the
        spacecraft. This indicates of the craft is moving forwards or backwards.

        From LROC Frame Kernel: lro_frames_2014049_v01.tf
        "+X axis is in the direction of the velocity vector half the year. The
        other half of the year, the +X axis is opposite the velocity vector"

        Hence we rotate the first velocity vector into the sensor reference
        frame, but the X component of that vector is inverted compared to the
        spacecraft so a +X indicates backwards and -X indicates forwards

        The returned velocity is also slightly off from the spacecraft velocity
        due to the sensor being attached to the craft with wax.

        Returns
        -------
        direction : double
                    X value of the first velocity relative to the sensor
        """
        frame_chain = self.frame_chain
        lro_bus_id = spice.bods2c('LRO_SC_BUS')
        time = self.ephemeris_start_time
        state, _ = spice.spkezr(self.spacecraft_name, time, 'J2000', 'None', self.target_name)
        position = state[:3]
        velocity = state[3:]
        rotation = frame_chain.compute_rotation(1, lro_bus_id)
        rotated_velocity = spice.mxv(rotation._rots.as_matrix()[0], velocity)
        return rotated_velocity[0]



class LroLrocNacIsisLabelNaifSpiceDriver(LineScanner, NaifSpice, IsisLabel, Driver):
    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """
        id_lookup = {
            "NACL": "LRO_LROCNACL",
            "NACR": "LRO_LROCNACR"
        }

        return id_lookup[super().instrument_id]

    @property
    def sensor_model_version(self):
        """
        Returns ISIS instrument sensor model version number

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def usgscsm_distortion_model(self):
        """
        The distortion model name with its coefficients

        LRO LROC NAC does not use the default distortion model so we need to overwrite the
        method packing the distortion model into the ISD.

        Returns
        -------
        : dict
          Returns a dict with the model name : dict of the coefficients
        """

        return {"lrolrocnac":
                {"coefficients": self.odtk}}

    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients. There is only one coefficient for LROC NAC l/r
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid), 0, 1).tolist()

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correction and abberation correction to
        use in NAIF calls.

        LROC is specifically set to not use light time correction because it is
        so close to the surface of the moon that light time correction to the
        center of the body is incorrect.

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'

    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels
        ISIS uses 0.5 based CCD samples, so we need to convert to 0 based.

        Returns
        -------
        float :
            The center sample of the CCD
        """
        return super().detector_center_sample - 0.5

    @property
    def ephemeris_start_time(self):
        """
        The starting ephemeris time for LRO is computed by taking the
        LRO:SPACECRAFT_CLOCK_PREROLL_COUNT, as defined in the label, and
        adding offsets that were taken from an IAK.

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        start_time = spice.scs2e(self.spacecraft_id, self.label['IsisCube']['Instrument']['SpacecraftClockPrerollCount'])
        return start_time + self.constant_time_offset + self.additional_preroll * self.exposure_duration

    @property
    def exposure_duration(self):
        """
        Takes the exposure_duration defined in a parent class and adds
        offsets taken from an IAK.

         Returns
         -------
         : float
           Returns the exposure duration in seconds.
         """
        return super().exposure_duration * (1 + self.multiplicative_line_error) + self.additive_line_error

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of
        the instrument. For LROC NAC this is flipped depending on the spacecraft
        direction.

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        focal2pixel_lines = np.array(list(spice.gdpool('INS{}_ITRANSL'.format(self.ikid), 0, 3))) / self.sampling_factor
        if self.spacecraft_direction < 0:
            return -focal2pixel_lines
        else:
            return focal2pixel_lines

    @property
    def multiplicative_line_error(self):
        """
        Returns the multiplicative line error defined in an IAK.

        Returns
        -------
        : float
          Returns the multiplicative line error.
        """
        return 0.0045

    @property
    def additive_line_error(self):
        """
        Returns the additive line error defined in an IAK.

        Returns
        -------
        : float
          Returns the additive line error.
        """
        return 0.0

    @property
    def constant_time_offset(self):
        """
        Returns the constant time offset defined in an IAK.

        Returns
        -------
        : float
          Returns the constant time offset.
        """
        return 0.0

    @property
    def additional_preroll(self):
        """
        Returns the addition preroll defined in an IAK.

        Returns
        -------
        : float
          Returns the additional preroll.
        """
        return 1024.0

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label that is defined by the CROSSTRACK_SUMMING.
        For example a return value of 2 indicates that 2 lines and 2 samples (4 pixels)
        were summed and divided by 4 to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        return self.label['IsisCube']['Instrument']['SpatialSumming']

    @property
    def spacecraft_direction(self):
        """
        Returns the x axis of the first velocity vector relative to the
        spacecraft. This indicates of the craft is moving forwards or backwards.

        From LROC Frame Kernel: lro_frames_2014049_v01.tf
        "+X axis is in the direction of the velocity vector half the year. The
        other half of the year, the +X axis is opposite the velocity vector"

        Hence we rotate the first velocity vector into the sensor reference
        frame, but the X component of that vector is inverted compared to the
        spacecraft so a +X indicates backwards and -X indicates forwards

        The returned velocity is also slightly off from the spacecraft velocity
        due to the sensor being attached to the craft with wax.

        Returns
        -------
        direction : double
                    X value of the first velocity relative to the sensor
        """
        frame_chain = self.frame_chain
        lro_bus_id = spice.bods2c('LRO_SC_BUS')
        time = self.ephemeris_start_time
        state, _ = spice.spkezr(self.spacecraft_name, time, 'J2000', 'None', self.target_name)
        position = state[:3]
        velocity = state[3:]
        rotation = frame_chain.compute_rotation(1, lro_bus_id)
        rotated_velocity = spice.mxv(rotation._rots.as_matrix()[0], velocity)
        return rotated_velocity[0]


class LroLrocNacIsisLabelIsisSpiceDriver(LineScanner, IsisSpice, IsisLabel, Driver):
    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """
        id_lookup = {
            "NACL": "LRO_LROCNACL",
            "NACR": "LRO_LROCNACR"
        }

        return id_lookup[super().instrument_id]

    @property
    def sensor_model_version(self):
        """
        Returns ISIS instrument sensor model version number

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def usgscsm_distortion_model(self):
        """
        The distortion model name with its coefficients

        LRO LROC NAC does not use the default distortion model so we need to overwrite the
        method packing the distortion model into the ISD.

        Returns
        -------
        : dict
          Returns a dict with the model name : dict of the coefficients
        """

        return {"lrolrocnac":
                {"coefficients": self.odtk}}

    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients. There is only one coefficient for LROC NAC l/r
        """
        key = 'INS{}_OD_K'.format(self.ikid)
        ans = self.naif_keywords.get(key, None)
        if ans is None:
            raise Exception('Could not parse the distortion model coefficients using key: ' + key)
        return [ans]

    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels
        ISIS uses 0.5 based CCD samples, so we need to convert to 0 based.

        Returns
        -------
        float :
            The center sample of the CCD
        """
        return super().detector_center_sample - 0.5

    @property
    def ephemeris_start_time(self):
        """
        The starting ephemeris time for LRO is computed by taking the
        LRO:SPACECRAFT_CLOCK_PREROLL_COUNT, as defined in the label, and
        adding offsets that were taken from an IAK.

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        return super().ephemeris_start_time + self.constant_time_offset + self.additional_preroll * self.exposure_duration

    @property
    def exposure_duration(self):
        """
        Takes the exposure_duration defined in a parent class and adds
        offsets taken from an IAK.

         Returns
         -------
         : float
           Returns the exposure duration in seconds.
         """
        return super().exposure_duration * (1 + self.multiplicative_line_error) + self.additive_line_error

    @property
    def multiplicative_line_error(self):
        """
        Returns the multiplicative line error defined in an IAK.

        Returns
        -------
        : float
          Returns the multiplicative line error.
        """
        return 0.0045

    @property
    def additive_line_error(self):
        """
        Returns the additive line error defined in an IAK.

        Returns
        -------
        : float
          Returns the additive line error.
        """
        return 0.0

    @property
    def constant_time_offset(self):
        """
        Returns the constant time offset defined in an IAK.

        Returns
        -------
        : float
          Returns the constant time offset.
        """
        return 0.0

    @property
    def additional_preroll(self):
        """
        Returns the addition preroll defined in an IAK.

        Returns
        -------
        : float
          Returns the additional preroll.
        """
        return 1024.0

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label that is defined by the CROSSTRACK_SUMMING.
        For example a return value of 2 indicates that 2 lines and 2 samples (4 pixels)
        were summed and divided by 4 to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        return self.label['IsisCube']['Instrument']['SpatialSumming']

    @property
    def spacecraft_direction(self):
        """
        Returns the x axis of the first velocity vector relative to the
        spacecraft. This indicates if the craft is moving forwards or backwards.

        From LROC Frame Kernel: lro_frames_2014049_v01.tf
        "+X axis is in the direction of the velocity vector half the year. The
        other half of the year, the +X axis is opposite the velocity vector"

        The returned velocity is also slightly off from the spacecraft velocity
        due to the sensor being attached to the craft with wax.

        Returns
        -------
        direction : double
                    X value of the first velocity relative to the spacecraft bus
        """
        _, velocities, _ = self.sensor_position
        rotation = self.frame_chain.compute_rotation(1, self.sensor_frame_id)
        rotated_velocity = rotation.apply_at(velocities[0], self.ephemeris_start_time)
        # We need the spacecraft bus X velocity which is parallel to the left
        # NAC X velocity and opposite the right NAC velocity.
        if (self.instrument_id == 'LRO_LROCNACR'):
          return -rotated_velocity[0]
        return rotated_velocity[0]


class LroMiniRfIsisLabelNaifSpiceDriver(Radar, NaifSpice, IsisLabel, Driver):
    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """
        id_lookup = {
            "MRFLRO": "LRO_MINIRF"
        }

        return id_lookup[super().instrument_id]

    @property
    def wavelength(self):
        """
        Returns the wavelength in meters used for image acquisition.

        Returns
        -------
        : double
          Wavelength in meters used to create an image
        """

        # Get float value of frequency in GHz
        frequency = self.label['IsisCube']['Instrument']['Frequency'].value
        wavelength = spice.clight() / frequency / 1000.0
        return wavelength

    @property
    def scaled_pixel_width(self):
        """
        Returns the scaled pixel width

        Returns
        -------
        : double
          scaled pixel width
        """
        return self.label['IsisCube']['Instrument']['ScaledPixelHeight'];


    # Default line_exposure_duration assumes that time is given in milliseconds and coverts
    # in this case, the time is already given in seconds.
    @property
    def line_exposure_duration(self):
        """
        Line exposure duration in seconds. The sum of the burst and the delay for the return.

        Returns
        -------
        : double
          scaled pixel width
        """
        return self.label['IsisCube']['Instrument']['LineExposureDuration']

    @property
    def range_conversion_coefficients(self):
        """
        Range conversion coefficients

        Returns
        -------
        : List
          range conversion coefficients
        """

        range_coefficients_orig = self.label['IsisCube']['Instrument']['RangeCoefficientSet']

        # The first elt of each list is time, which we handle separately in range_conversion_time
        range_coefficients = [elt[1:] for elt in range_coefficients_orig]
        return range_coefficients

    @property
    def range_conversion_times(self):
        """
        Times, in et, associated with range conversion coefficients

        Returns
        -------
        : List
          times for range conversion coefficients
        """
        range_coefficients_utc = self.label['IsisCube']['Instrument']['RangeCoefficientSet']
        range_coefficients_et = [spice.str2et(elt[0]) for elt in range_coefficients_utc]
        return range_coefficients_et


    @property
    def ephemeris_start_time(self):
        """
        Returns the start ephemeris time for the image.

        Returns
        -------
        : float
          start time
        """
        return spice.str2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")) - self.line_exposure_duration

    @property
    def ephemeris_stop_time(self):
        """
        Returns the stop ephemeris time for the image. This is computed from 
        the start time plus the line exposure per line, plus the line exposure
        removed from the start time, plus the line exposure for the final line.

        Returns
        -------
        : float
          stop time
        """
        return self.ephemeris_start_time + (self.image_lines * self.line_exposure_duration) + (self.line_exposure_duration * 2)

    @property
    def look_direction(self):
        """
        Direction of the look (left or right)

        Returns
        -------
        : string
          left or right
        """
        return self.label['IsisCube']['Instrument']['LookDirection'].lower()

    @property
    def sensor_frame_id(self):
        """
        Returns the Naif ID code for the sensor reference frame
        We replace this with the target frame ID because the sensor operates
        entirely in the target reference frame
        Returns
        -------
        : int
          Naif ID code for the sensor frame
        """
        return self.target_frame_id

    @property
    def naif_keywords(self):
        """
        Adds the correct TRANSX/Y and ITRANS/L values for use in ISIS. By default
        these values are placeholders in the ISIS iaks and need to be computed manully
        from the ground range resolution. See RadarGroundRangeMap.cpp in ISIS for
        the calculations.

        Returns
        -------
          : dict
            An updated dictionary of NAIF keywords with the correct TRANSX/Y and ITRANSS/L
            values computed
        """
        naif_keywords = super().naif_keywords
        ground_range_resolution = self.label['IsisCube']['Instrument']["ScaledPixelHeight"]
        icode = "INS" + str(self.ikid)
        naif_keywords[icode + "_TRANSX"] = [-1.0 * ground_range_resolution, ground_range_resolution, 0.0]
        naif_keywords[icode + "_TRANSY"] = [0.0, 0.0, 0.0]
        naif_keywords[icode + "_ITRANSS"] = [1.0, 1.0 / ground_range_resolution, 0.0]
        naif_keywords[icode + "_ITRANSL"] = [0.0, 0.0, 0.0]
        return naif_keywords

class LroLrocWacIsisLabelIsisSpiceDriver(PushFrame, IsisLabel, IsisSpice, RadialDistortion, Driver):
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        expect the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'WAC-UV' or 'WAC-VIS'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
        "WAC-UV" : "LRO_LROCWAC_UV",
        "WAC-VIS" : "LRO_LROCWAC_VIS"
        }
        return id_lookup[super().instrument_id]

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']


    @property
    def sensor_model_version(self):
        """
        Returns ISIS instrument sensor model version number

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2


    @property
    def filter_number(self):
        """
        Return the filter number from the cub label

        Returns
        -------
        : int
          The filter number
        """
        try:
            return self.label['IsisCube']['BandBin']['FilterNumber'][0]
        except:
            return self.label['IsisCube']['BandBin']['FilterNumber']


    @property
    def fikid(self):
        """
        Naif ID code of the filter dependent instrument codes.

        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        if self.instrument_id == "LRO_LROCWAC_UV":
            base = -85640
        elif self.instrument_id == "LRO_LROCWAC_VIS":
            base = -85630

        fikid = base - self.filter_number
        return fikid


    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients.
        """
        key = 'INS{}_OD_K'.format(self.fikid)
        ans = self.naif_keywords.get(key, None)
        if ans is None:
            raise Exception('Could not parse the distortion model coefficients using key: ' + key)
        
        ans = [x * -1 for x in ans]
        return ans

    @property
    def framelet_height(self):
        if self.instrument_id == "LRO_LROCWAC_UV":
            return 16
        elif self.instrument_id == "LRO_LROCWAC_VIS":
            return 14


    @property
    def pixel2focal_x(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of the filter

        Returns
        -------
        : list<double>
        detector to focal plane x
        """
        key = 'INS{}_TRANSX'.format(self.fikid)
        ans = self.naif_keywords.get(key, None)
        if ans is None:
            raise Exception('Could not parse detector to focal plane x using key: ' + key)
        return ans

    @property
    def pixel2focal_y(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of the filter

        Returns
        -------
        : list<double>
        detector to focal plane y
        """
        key = 'INS{}_TRANSY'.format(self.fikid)
        ans = self.naif_keywords.get(key, None)
        if ans is None:
            raise Exception('Could not parse detector to focal plane y using key: ' + key)
        return ans

    @property
    def focal_length(self):
        """
        The focal length of the instrument
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keywords from the label.
        Expects fikid to be defined. This should be the integer Naif ID code
        for the filter.

        Returns
        -------
        float :
            The focal length in millimeters
        """
        key = 'INS{}_FOCAL_LENGTH'.format(self.fikid)
        ans = self.naif_keywords.get(key, None)
        if ans is None:
            raise Exception('Could not parse the focal length using key: ' + key)
        return ans
    
    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels
        Expects fikid to be defined. This should be the integer Naif ID code
        for the filter.

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        key = 'INS{}_BORESIGHT_SAMPLE'.format(self.fikid)
        ans = self.naif_keywords.get(key, None)
        if ans is None:
            raise Exception('Could not parse the detector center sample using key: ' + key)
        return ans

    @property
    def detector_center_line(self):
        """
        The center of the CCD in detector pixels
        Expects fikid to be defined. This should be the integer Naif ID code
        for the filter.

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        key = 'INS{}_BORESIGHT_LINE'.format(self.fikid)
        ans = self.naif_keywords.get(key, None)
        if ans is None:
            raise Exception('Could not parse the detector center line using key: ' + key)
        return ans

class LroLrocWacIsisLabelNaifSpiceDriver(PushFrame, IsisLabel, NaifSpice, RadialDistortion, Driver):
    """
    Driver for Lunar Reconnaissance Orbiter WAC ISIS cube
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        expect the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'WAC-UV' or 'WAC-VIS'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            "WAC-UV" : "LRO_LROCWAC_UV",
            "WAC-VIS" : "LRO_LROCWAC_VIS"
        }
        return id_lookup[super().instrument_id]

    @property
    def sensor_model_version(self):
        return 2


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
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.label['IsisCube']['Instrument']['SpacecraftClockStartCount']
            self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, sclock)
        return self._ephemeris_start_time


    @property
    def sensor_name(self):
        """
        Returns
        -------
        : String
          The name of the spacecraft
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']


    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients.
        """
        coeffs = spice.gdpool('INS{}_OD_K'.format(self.fikid), 0, 3).tolist()
        coeffs = [x * -1 for x in coeffs]
        return coeffs


    @property
    def naif_keywords(self):
        """
        Updated set of naif keywords containing the NaifIkCode for the specific
        WAC filter used when taking the image.

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        _naifKeywords = {**super().naif_keywords,
                         **query_kernel_pool("*_FOCAL_LENGTH"),
                         **query_kernel_pool("*_BORESIGHT_SAMPLE"),
                         **query_kernel_pool("*_BORESIGHT_LINE"),
                         **query_kernel_pool("*_TRANS*"),
                         **query_kernel_pool("*_ITRANS*"),
                         **query_kernel_pool("*_OD_K")}
        return _naifKeywords


    @property
    def framelets_flipped(self):
        """
        Returns
        -------
        : boolean
          True if framelets are flipped, else false
        """
        return self.label['IsisCube']['Instrument']['DataFlipped'] == "Yes"


    @property
    def sampling_factor(self):
        if self.instrument_id == "LRO_LROCWAC_UV":
            return 4
        elif self.instrument_id == "LRO_LROCWAC_VIS":
            return 1


    @property
    def num_frames(self):
        """
        Number of frames in the image

        Returns
        -------
        : int
          Number of frames in the image
        """
        return self.image_lines // (self.framelet_height // self.sampling_factor)


    @property
    def framelet_height(self):
        """
        Return the number of lines in a framelet.

        Returns
        -------
        : int
          The number of lines in a framelet
        """
        if self.instrument_id == "LRO_LROCWAC_UV":
            return 16
        elif self.instrument_id == "LRO_LROCWAC_VIS":
            return 14

    @property
    def num_lines_overlap(self):
        """
        Returns
        -------
        : int
          How many many lines of a framelet overlap with neighboring framelets.
        """
        try:
            return self.label['IsisCube']['Instrument']['NumLinesOverlap']
        except:
            # May be missing, and then the default is 0
            return 0

    @property
    def filter_number(self):
        """
        Return the filter number from the cub label

        Returns
        -------
        : int
          The filter number
        """
        try:
            return self.label['IsisCube']['BandBin']['FilterNumber'][0]
        except:
            return self.label['IsisCube']['BandBin']['FilterNumber']


    @property
    def fikid(self):
        """
        Naif ID code of the filter dependent instrument codes.

        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        if self.instrument_id == "LRO_LROCWAC_UV":
            base = -85640
        elif self.instrument_id == "LRO_LROCWAC_VIS":
            # Offset by 2 because the first 2 filters are UV
            base = -85628

        fikid = base - self.filter_number
        return fikid


    @property
    def pixel2focal_x(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of the filter

        Returns
        -------
        : list<double>
        detector to focal plane x
        """
        return list(spice.gdpool('INS{}_TRANSX'.format(self.fikid), 0, 3))


    @property
    def pixel2focal_y(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of the filter

        Returns
        -------
        : list<double>
        detector to focal plane y
        """
        return list(spice.gdpool('INS{}_TRANSY'.format(self.fikid), 0, 3))

    @property
    def focal2pixel_lines(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of the filter

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        return list(spice.gdpool('INS{}_ITRANSL'.format(self.fikid), 0, 3))

    @property
    def focal2pixel_samples(self):
        """
        Expects fikid to be defined. This must be the integer Naif id code of the filter

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        return list(spice.gdpool('INS{}_ITRANSS'.format(self.fikid), 0, 3))


    @property
    def detector_start_line(self):
        """
        Filter-specific starting line

        Returns
        -------
        : int
          Zero based Detector line corresponding to the first image line
        """
        offset = list(spice.gdpool('INS{}_FILTER_OFFSET'.format(self.fikid), 0, 3))
        try:
            # If multiple items are present, use the first one
            offset = offset[0]
        except (IndexError, TypeError):
            pass
        return super().detector_start_line + offset

    @property
    def focal_length(self):
        """
        Returns the focal length of the sensor
        Expects fikid to be defined. This must be the integer Naif id code of
        the filter.

        Returns
        -------
        : float
          focal length
        """
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.fikid), 0, 1)[0])

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return float(spice.gdpool('INS{}_BORESIGHT_SAMPLE'.format(self.fikid), 0, 1)[0]) - 0.5

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return float(spice.gdpool('INS{}_BORESIGHT_LINE'.format(self.fikid), 0, 1)[0]) - 0.5
