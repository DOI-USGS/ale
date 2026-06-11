import numpy as np
import scipy.constants
import spiceypy as spice
from pyspiceql import pyspiceql

from ale.base import Driver, WrongInstrumentException
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import CassisDistortion

class TGOCassisIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, CassisDistortion, Driver):
    """
    Driver for reading TGO Cassis ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for unquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form CaSSIS

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            'CaSSIS': 'TGO_CASSIS',
        }
        key = super().instrument_id
        if key not in id_lookup:
            raise WrongInstrumentException(f"Unknown instrument id: {key}.")
        return id_lookup[key]

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris_start_time of the image.
        Expects spacecraft_clock_start_count to be defined. This should be a float
        containing the start clock count of the spacecraft.
        Expects spacecraft_id to be defined. This should be the integer Naif ID code
        for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image.
        """
        if not hasattr(self, "_ephemeris_start_time"):
            self._ephemeris_start_time = pyspiceql.utcToEt(utc=self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"), searchKernels=self.search_kernels, useWeb=self.use_web)[0]
        return self._ephemeris_start_time

    @property
    def sensor_frame_id(self):
        return -143420

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def sample_summing(self):
        """
        CaSSIS stores SummingMode as an enum (0 = 1x1, 1 = 2x2, 2 = 4x4), not as
        the summing factor itself. ISIS converts it as summing = sumMode * 2, then
        falls back to 1 when that is 0 (see TgoCassisCamera). Replicate that here,
        otherwise the CSM detector summing becomes 0 and groundToImage diverges.
        """
        sum_mode = self.label['IsisCube']['Instrument']['SummingMode']
        summing = sum_mode * 2
        if summing <= 0:
            summing = 1
        return summing

    @property
    def line_summing(self):
        return self.sample_summing

    @property
    def detector_center_sample(self):
        """
        ISIS uses 0.5-based CCD coordinates (pixel centers at half integers),
        so convert the IK boresight sample to the CSM 0-based convention by
        subtracting 0.5, as the LRO, MRO, Dawn, MESSENGER, MEX, Kaguya and KPLO
        drivers do. Without this the CSM look is offset from ISIS by half a pixel
        in sample (and half in line), i.e. sqrt(0.5^2+0.5^2) ~ 0.707 px.
        """
        return super().detector_center_sample - 0.5

    @property
    def detector_center_line(self):
        """
        ISIS uses 0.5-based CCD coordinates; convert to the CSM 0-based
        convention by subtracting 0.5 (see detector_center_sample).
        """
        return super().detector_center_line - 0.5

    @property
    def sensor_position(self):
        """
        CaSSIS sets LT_SURFACE_CORRECT with LIGHTTIME_CORRECTION=LT+S, so ISIS
        applies the surface light-time correction. The shared
        NaifSpice.sensor_position samples the target body at the raw ephemeris
        time in that branch, but ISIS samples it at the surface-light-time
        adjusted time (ephem - obs_tar_lt + radius_lt); the body moves along its
        orbit during that interval, which otherwise leaves a constant
        tens-of-meters camera-center bias versus ISIS. This override applies that
        for CaSSIS only, so the shared path is unchanged for every other sensor.
        It is the shared surface-light-time branch with the single change that the
        body is sampled at adjusted_time. CaSSIS is a single-record framer, so
        sampling the body at adjusted_time[0]..[-1] is exact.
        """
        if not (self.correct_lt_to_surface
                and self.light_time_correction.upper() == 'LT+S'):
            return super().sensor_position

        if not hasattr(self, '_position'):
            ephem = self.ephemeris_time
            pos = []
            vel = []

            target = self.spacecraft_name
            observer = self.target_name
            if self.swap_observer_target:
                target = self.target_name
                observer = self.spacecraft_name

            ephem_kwargs = {"startEt": ephem[0],
                            "stopEt": ephem[-1],
                            "numRecords": len(ephem),
                            "ckQualities": ["reconstructed"],
                            "spkQualities": ["reconstructed"],
                            "searchKernels": self.search_kernels,
                            "useWeb": self.use_web}

            obs_tars_kwargs = {**ephem_kwargs,
                               "target": target,
                               "observer": observer,
                               "frame": "J2000",
                               "abcorr": self.light_time_correction,
                               "mission": self.spiceql_mission}
            ssb_obs_kwargs = {**ephem_kwargs,
                              "target": observer,
                              "observer": "SSB",
                              "frame": "J2000",
                              "abcorr": "NONE",
                              "mission": self.spiceql_mission}

            obs_tars = pyspiceql.getTargetStatesRanged(**obs_tars_kwargs)[0]
            ssb_obs = pyspiceql.getTargetStatesRanged(**ssb_obs_kwargs)[0]

            obs_tar_lts = np.array(obs_tars)[:, -1]
            ssb_obs_states = np.array(ssb_obs)[:, 0:6]

            radius_lt = (self.target_body_radii[2] + self.target_body_radii[0]) / 2 \
                / (scipy.constants.c / 1000.0)
            adjusted_time = ephem - obs_tar_lts + radius_lt

            # The only change from the shared method: sample the target body at
            # the surface-light-time adjusted time rather than the raw ephem time.
            ssb_tars_kwargs = {**ephem_kwargs,
                               "target": target,
                               "observer": "SSB",
                               "frame": "J2000",
                               "abcorr": "NONE",
                               "mission": self.spiceql_mission}
            ssb_tars_kwargs["startEt"] = adjusted_time[0]
            ssb_tars_kwargs["stopEt"] = adjusted_time[-1]
            ssb_tars = pyspiceql.getTargetStatesRanged(**ssb_tars_kwargs)[0]
            ssb_tar_states = np.array(ssb_tars)[:, 0:6]

            _states = ssb_tar_states - ssb_obs_states

            reference_frame_id = pyspiceql.translateNameToCode(frame=self.reference_frame,
                                                               mission=self.spiceql_mission,
                                                               searchKernels=self.search_kernels,
                                                               useWeb=self.use_web)[0]

            function_args = {**ephem_kwargs,
                             "toFrame": reference_frame_id,
                             "refFrame": 1,
                             "mission": self.spiceql_mission}
            function_args.pop("spkQualities")
            rotations = pyspiceql.getTargetOrientationsRanged(**function_args)[0]

            states = []
            for i, rotation in enumerate(rotations):
                quaternion = rotation[:4]
                av = [0, 0, 0]
                if len(rotation) > 4:
                    av = rotation[4:]
                rotation_matrix = spice.q2m(quaternion)
                matrix = spice.rav2xf(rotation_matrix, av)
                rotated_state = spice.mxvg(matrix, _states[i])
                states.append(rotated_state)

            for state in states:
                if self.swap_observer_target:
                    pos.append(-state[:3])
                    vel.append(-state[3:])
                else:
                    pos.append(state[:3])
                    vel.append(state[3:])

            # SPICE works in km, so convert to m.
            self._position = 1000 * np.asarray(pos)
            self._velocity = 1000 * np.asarray(vel)
            self._ephem = ephem
        return self._position, self._velocity, self._ephem
