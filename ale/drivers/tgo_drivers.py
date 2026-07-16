import os
import glob
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
        Returns an instrument id for uniquely identifying the instrument, but often
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
    def kernels(self):
        """
        Furnish the ISIS CaSSIS addendum (tgoCassisAddendum) alongside the mission
        metakernel. Several CaSSIS constants (focal length, ITRANSS/ITRANSL,
        boresight, the optical distortion, and the light-time flags) live only in
        the ISIS addendum, not in any NAIF instrument kernel.
        """
        kernels = super().kernels
        if not getattr(self, "_cassis_addendum_furnished", False):
            self._cassis_addendum_furnished = True
            iak = self._isis_cassis_addendum()
            if iak is not None:
                if isinstance(kernels, dict):
                    existing = kernels.get("iak", [])
                    if not any("Addendum" in str(k) for k in existing):
                        kernels["iak"] = existing + [iak]
                elif isinstance(kernels, list):
                    if not any("Addendum" in str(k) for k in kernels):
                        kernels = kernels + [iak]
                        self._kernels = kernels
        return kernels

    def _isis_cassis_addendum(self):
        """
        Locate the latest ISIS CaSSIS addendum in the ISIS data area, mirroring the
        ISIS iak kernel database (match on the CaSSIS instrument, tgoCassisAddendum
        with the highest version). Returns the path, or None when the data area has
        no addendum (so the caller falls back to hardcoded constants).
        """
        root = os.environ.get("ALESPICEROOT") or os.environ.get("ISISDATA")
        if not root:
            return None
        matches = sorted(glob.glob(os.path.join(root, "tgo", "kernels", "iak",
                                                 "tgoCassisAddendum*.ti")))
        return matches[-1] if matches else None

    @property
    def _addendum_furnished(self):
        """
        True when the ISIS addendum is furnished. FOCAL_LENGTH is addendum-only (no
        NAIF kernel has it), so its presence is a reliable flag. Needed for the
        distortion, whose OD_A keys exist in both the NAIF kernel and the addendum
        with different values, so their presence alone cannot tell them apart.
        """
        return f"INS{self.ikid}_FOCAL_LENGTH" in self.naif_keywords

    def _addendum_keyword(self, suffix, default):
        """
        Read INS<ikid>_<suffix> from the furnished pool, or return default (the
        latest known addendum constant) when the addendum is not furnished.
        """
        value = self.naif_keywords.get(f"INS{self.ikid}_{suffix}")
        return default if value is None else value

    @property
    def focal_length(self):
        """
        CaSSIS focal length in mm, read from the furnished addendum
        (INS-143400_FOCAL_LENGTH); no NAIF kernel carries it. Fallback is the latest
        known value (007 = 874.9), used only when the addendum is not furnished.
        """
        return self._addendum_keyword("FOCAL_LENGTH", 874.9)

    @property
    def light_time_correction(self):
        """
        INS-143400_LIGHTTIME_CORRECTION (addendum), read from the furnished pool.
        Fallback LT+S so the sensor_position override still fires.
        """
        return self._addendum_keyword("LIGHTTIME_CORRECTION", 'LT+S')

    @property
    def correct_lt_to_surface(self):
        """
        INS-143400_LT_SURFACE_CORRECT (addendum). Without it the sensor_position
        override does not fire and the camera center is off about 96 m from ISIS,
        so the fallback is True.
        """
        return bool(self._addendum_keyword("LT_SURFACE_CORRECT", True))

    @property
    def swap_observer_target(self):
        """
        INS-143400_SWAP_OBSERVER_TARGET (addendum), read from the furnished pool,
        fallback True to match ISIS.
        """
        return bool(self._addendum_keyword("SWAP_OBSERVER_TARGET", True))

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
        Boresight sample INS-143400_BORESIGHT_SAMPLE (addendum), converted from the
        ISIS 0.5-based CCD convention to CSM 0-based by subtracting 0.5 (as LRO, MRO,
        etc. do); without it the look is off about 0.707 px. ISIS keys the camera on
        the base -143400 frame for every filter, so this holds for PAN/RED/NIR/BLU.
        """
        return self._addendum_keyword("BORESIGHT_SAMPLE", 1024.5) - 0.5

    @property
    def detector_center_line(self):
        """
        Boresight line INS-143400_BORESIGHT_LINE (addendum), converted to CSM
        0-based by subtracting 0.5 (see detector_center_sample).
        """
        return self._addendum_keyword("BORESIGHT_LINE", 1024.5) - 0.5

    @property
    def focal2pixel_lines(self):
        """
        Focal-plane mm to detector lines, INS-143400_ITRANSL (addendum). The
        fallback [0, 0, 100] encodes 1 / pixel pitch (0.01 mm).
        """
        return list(self._addendum_keyword("ITRANSL", [0.0, 0.0, 100.0]))

    @property
    def focal2pixel_samples(self):
        """
        Focal-plane mm to detector samples, INS-143400_ITRANSS (addendum).
        """
        return list(self._addendum_keyword("ITRANSS", [0.0, 100.0, 0.0]))

    @property
    def usgscsm_distortion_model(self):
        """
        CaSSIS rational (ratio-of-quadratics) distortion, INS-143400_OD_A{1,2,3}_
        {CORR,DIST}, read from the furnished addendum. These keys exist in both the
        NAIF kernel and the addendum with different values (ISIS uses the addendum),
        so gate on FOCAL_LENGTH (addendum-only), not on OD_A presence. Fall back to
        the latest known coefficients when the addendum is not furnished. Packed
        A1_corr, A2_corr, A3_corr, A1_dist, A2_dist, A3_dist.
        """
        if self._addendum_furnished:
            nk = self.naif_keywords
            i = self.ikid
            coefficients = (list(nk[f"INS{i}_OD_A1_CORR"]) + list(nk[f"INS{i}_OD_A2_CORR"]) +
                            list(nk[f"INS{i}_OD_A3_CORR"]) + list(nk[f"INS{i}_OD_A1_DIST"]) +
                            list(nk[f"INS{i}_OD_A2_DIST"]) + list(nk[f"INS{i}_OD_A3_DIST"]))
            return {"cassis": {"coefficients": coefficients}}

        A1_CORR = [0.0037613053094826604, -0.0134154156065812, -1.8674952100723702e-05,
                   1.00021352681836, -0.00043236237170395304, -0.000948065735350123]
        A2_CORR = [9.9842559363676e-05, 0.00373543707958162, -0.0133299918873929,
                   -0.000215311328389359, 0.995296015537294, -0.0183542717710778]
        A3_CORR = [-3.13320167004204e-05, -7.3565512574980695e-06, -1.57664245066771e-05,
                   0.0037354946543915104, -0.014167194693093499, 1.0]
        A1_DIST = [0.0021365879556062197, -0.00711785765064197, 1.10355974742147e-05,
                   0.573607182625377, 0.000250884350194894, 0.000550623913037132]
        A2_DIST = [-5.6972574101540595e-05, 0.00215155905679149, -0.00716392991767185,
                   0.000124152787728634, 0.5764595443924261, 0.010576940564854]
        A3_DIST = [1.7825077148350597e-05, 4.24592743471094e-06, 9.51220699036653e-06,
                   0.0021515842542073798, -0.0066835595774833, 0.573741540971609]
        return {"cassis": {"coefficients":
                A1_CORR + A2_CORR + A3_CORR + A1_DIST + A2_DIST + A3_DIST}}
