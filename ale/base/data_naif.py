import spiceypy as spice
import numpy as np
import scipy.constants

import ale
from ale.base.type_sensor import Framer
from ale.transformation import FrameChain
from ale.rotation import TimeDependentRotation
from ale import util

class NaifSpice():
    """
    Mix-in for reading data from NAIF SPICE Kernels.
    """

    def __enter__(self):
        """
        Called when the context is created. This is used
        to get the kernels furnished.
        """
        if self.kernels:
            [spice.furnsh(k) for k in self.kernels]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context goes out of scope. Once
        this is done, the object is out of scope and the
        kernels can be unloaded.
        """
        if self.kernels:
            [spice.unload(k) for k in self.kernels]

    @property
    def kernels(self):
        """
        Get the NAIF SPICE Kernels to furnish

        There are two ways to specify which kernels a driver will use:

        1. Passing the 'kernels' property into load(s) or at instantiation.
           This can be either a straight iterable or a dictionary that specifies
           the kernels in ISIS style ('TargetPosition', 'InstrumentPosition', etc).
        2. Set the ALESPICEROOT environment variable. This variable should be
           the path to a directory that contains directories whose naming
           convention matches the PDS Kernel Archives format,
           `shortMissionName-versionInfo`. The directory corresponding to the
           driver's mission will be searched for the appropriate meta kernel to
           load.

        See Also
        --------
        ale.util.get_kernels_from_isis_pvl : Function used to parse ISIS style dict
        ale.util.get_metakernels : Function that searches ALESPICEROOT for meta kernels
        ale.util.generate_kernels_from_cube : Helper function to get an ISIS style dict
                                              from an ISIS cube that has been through
                                              spiceinit

        """
        if not hasattr(self, '_kernels'):
            if 'kernels' in self._props.keys():
                try:
                    self._kernels = util.get_kernels_from_isis_pvl(self._props['kernels'])
                except Exception as e:
                    self._kernels =  self._props['kernels']
            else:
                if not ale.spice_root:
                    raise EnvironmentError(f'ale.spice_root is not set, cannot search for metakernels. ale.spice_root = "{ale.spice_root}"')

                search_results = util.get_metakernels(ale.spice_root, missions=self.short_mission_name, years=self.utc_start_time.year, versions='latest')

                if search_results['count'] == 0:
                    raise ValueError(f'Failed to find metakernels. mission: {self.short_mission_name}, year:{self.utc_start_time.year}, versions="latest" spice root = "{ale.spice_root}"')
                self._kernels = [search_results['data'][0]['path']]

        return self._kernels

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correction and abberation correction to
        use in NAIF calls. Expects ikid to be defined. This must be the integer
        Naif id code of the instrument.

        This searches for the value of the NAIF keyword INS<ikid>_LIGHTTIME_CORRECTION.
        If the keyword is not defined, then this defaults to light time
        correction and abberation correction (LT+S).

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        try:
            return spice.gcpool('INS{}_LIGHTTIME_CORRECTION'.format(self.ikid), 0, 1)[0]
        except:
            return 'LT+S'

    @property
    def odtx(self):
        """
        Returns the x coefficient for the optical distortion model
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_OD_T_X'.format(self.ikid),0, 10).tolist()

    @property
    def odty(self):
        """
        Returns the y coefficient for the optical distortion model.
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_OD_T_Y'.format(self.ikid), 0, 10).tolist()

    @property
    def odtk(self):
        """
        The coefficients for the radial distortion model
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list
          Radial distortion coefficients
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid),0, 3).tolist()

    @property
    def ikid(self):
        """
        Returns the Naif ID code for the instrument
        Expects the instrument_id to be defined. This must be a string containing
        the short name of the instrument.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return spice.bods2c(self.instrument_id)

    @property
    def spacecraft_id(self):
        """
        Returns the Naif ID code for the spacecraft
        Expects the spacecraft_name to be defined. This must be a string containing
        the name of the spacecraft.

        Returns
        -------
        : int
          Naif ID code for the spacecraft
        """
        return spice.bods2c(self.spacecraft_name)

    @property
    def target_id(self):
        """
        Returns the Naif ID code for the target body
        Expects target_name to be defined. This must be a string containing the name
        of the target body.

        Returns
        -------
        : int
          Naif ID code for the target body
        """
        return spice.bods2c(self.target_name)

    @property
    def target_frame_id(self):
        """
        Returns the Naif ID code for the target reference frame
        Expects the target_id to be defined. This must be the integer Naif ID code
        for the target body.

        Returns
        -------
        : int
          Naif ID code for the target frame
        """
        frame_info = spice.cidfrm(self.target_id)
        return frame_info[0]

    @property
    def sensor_frame_id(self):
        """
        Returns the Naif ID code for the sensor reference frame
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : int
          Naif ID code for the sensor frame
        """
        return self.ikid

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        return list(spice.gdpool('INS{}_ITRANSL'.format(self.ikid), 0, 3))

    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        return list(spice.gdpool('INS{}_ITRANSS'.format(self.ikid), 0, 3))

    @property
    def pixel2focal_x(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane x
        """
        return list(spice.gdpool('INS{}_TRANSX'.format(self.ikid), 0, 3))

    @property
    def pixel2focal_y(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane y
        """
        return list(spice.gdpool('INS{}_TRANSY'.format(self.ikid), 0, 3))

    @property
    def focal_length(self):
        """
        Returns the focal length of the sensor
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : float
          focal length
        """
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

    @property
    def pixel_size(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : float pixel size
        """
        return spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001

    @property
    def target_body_radii(self):
        """
        Returns a list containing the radii of the target body
        Expects target_name to be defined. This must be a string containing the name
        of the target body

        Returns
        -------
        : list<double>
          Radius of all three axis of the target body
        """
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1]

    @property
    def reference_frame(self):
        """
        Returns a string containing the name of the target reference frame
        Expects target_name to be defined. This must be a string containing the name
        of the target body

        Returns
        -------
        : str
        String name of the target reference frame
        """
        try:
            return spice.cidfrm(spice.bodn2c(self.target_name))[1]
        except:
            return 'IAU_{}'.format(self.target_name)

    @property
    def sun_position(self):
        """
        Returns a tuple with information detailing the sun position at the time
        of the image. Expects center_ephemeris_time to be defined. This must be
        a floating point number containing the average of the start and end ephemeris time.
        Expects reference frame to be defined. This must be a string containing the name of
        the target reference frame. Expects target_name to be defined. This must be
        a string containing the name of the target body.

        Returns
        -------
        : (sun_positions, sun_velocities)
          a tuple containing a list of sun positions, a list of sun velocities
        """
        times = [self.center_ephemeris_time]
        positions = []
        velocities = []

        for time in times:
            sun_state, _ = spice.spkezr("SUN",
                                         time,
                                         self.reference_frame,
                                         'LT+S',
                                         self.target_name)
            positions.append(sun_state[:3])
            velocities.append(sun_state[3:6])
        positions = 1000 * np.asarray(positions)
        velocities = 1000 * np.asarray(velocities)

        return positions, velocities, times

    @property
    def sensor_position(self):
        """
        Returns a tuple with information detailing the position of the sensor at the time
        of the image. Expects ephemeris_time to be defined. This must be a floating point number
        containing the ephemeris time. Expects spacecraft_name to be defined. This must be a
        string containing the name of the spacecraft containing the sensor. Expects
        reference_frame to be defined. This must be a string containing the name of
        the target reference frame. Expects target_name to be defined. This must be
        a string containing the name of the target body.

        Returns
        -------
        : (positions, velocities, times)
          a tuple containing a list of positions, a list of velocities, and a list of times
        """
        if not hasattr(self, '_position'):
            ephem = self.ephemeris_time
            pos = []
            vel = []

            target = self.spacecraft_name
            observer = self.target_name
            # Check for ISIS flag to fix target and observer swapping
            if self.swap_observer_target:
                target = self.target_name
                observer = self.spacecraft_name

            for time in ephem:
                # spkezr returns a vector from the observer's location to the aberration-corrected
                # location of the target. For more information, see:
                # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/spkezr.html
                if self.correct_lt_to_surface and self.light_time_correction.upper() == 'LT+S':
                    obs_tar_state, obs_tar_lt = spice.spkezr(target,
                                                             time,
                                                             'J2000',
                                                             self.light_time_correction,
                                                             observer)
                    # ssb to spacecraft
                    ssb_obs_state, ssb_obs_lt = spice.spkezr(observer,
                                                             time,
                                                             'J2000',
                                                             'NONE',
                                                             'SSB')

                    radius_lt = (self.target_body_radii[2] + self.target_body_radii[0]) / 2 / (scipy.constants.c/1000.0)
                    adjusted_time = time - obs_tar_lt + radius_lt
                    ssb_tar_state, ssb_tar_lt = spice.spkezr(target,
                                                             adjusted_time,
                                                             'J2000',
                                                             'NONE',
                                                             'SSB')
                    state = ssb_tar_state - ssb_obs_state
                    matrix = spice.sxform("J2000", self.reference_frame, time)
                    state = spice.mxvg(matrix, state)
                else:
                    state, _ = spice.spkezr(target,
                                            time,
                                            self.reference_frame,
                                            self.light_time_correction,
                                            observer)

                if self.swap_observer_target:
                    pos.append(-state[:3])
                    vel.append(-state[3:])
                else:
                    pos.append(state[:3])
                    vel.append(state[3:])

            # By default, SPICE works in km, so convert to m
            self._position = [p * 1000 for p in pos]
            self._velocity = [v * 1000 for v in vel]
        return self._position, self._velocity, self.ephemeris_time

    @property
    def frame_chain(self):
        if not hasattr(self, '_frame_chain'):
            nadir = self._props.get('nadir', False)
            self._frame_chain = FrameChain.from_spice(sensor_frame=self.sensor_frame_id,
                                                      target_frame=self.target_frame_id,
                                                      center_ephemeris_time=self.center_ephemeris_time,
                                                      ephemeris_times=self.ephemeris_time,
                                                      nadir=nadir)

            if nadir:
                # Logic for nadir calculation was taken from ISIS3
                #  SpiceRotation::setEphemerisTimeNadir
                rotation = self._frame_chain.compute_rotation(self.target_frame_id, 1)
                p_vec, v_vec, times = self.sensor_position
                rotated_positions = rotation.apply_at(p_vec, times)
                rotated_velocities = rotation.rotate_velocity_at(p_vec, v_vec, times)

                p_vec = rotated_positions
                v_vec = rotated_velocities

                velocity_axis = 2
                # Get the default line translation with no potential flipping
                # from the driver
                trans_x = np.array(list(spice.gdpool('INS{}_ITRANSL'.format(self.ikid), 0, 3)))

                if (trans_x[0] < trans_x[1]):
                    velocity_axis = 1

                quats = [spice.m2q(spice.twovec(-p_vec[i], 3, v_vec[i], velocity_axis)) for i, time in enumerate(times)]
                quats = np.array(quats)[:,[1,2,3,0]]

                rotation = TimeDependentRotation(quats, times, 1, self.sensor_frame_id)
                self._frame_chain.add_edge(rotation)

        return self._frame_chain


    @property
    def sensor_orientation(self):
        """
        Returns quaternions describing the sensor orientation. Expects ephemeris_time
        to be defined. This must be a floating point number containing the
        ephemeris time. Expects instrument_id to be defined. This must be a string
        containing the short name of the instrument. Expects reference frame to be defined.
        This must be a string containing the name of the target reference frame.

        Returns
        -------
        : list
          Quaternions describing the orientation of the sensor
        """
        if not hasattr(self, '_orientation'):
            self._orientation = self.frame_chain.compute_rotation(self.sensor_frame_id, self.target_frame_id).quats
        return self._orientation.tolist()

    @property
    def ephemeris_start_time(self):
        """
        Returns the starting ephemeris time of the image. Expects spacecraft_id to
        be defined. This must be the integer Naif Id code for the spacecraft. Expects
        spacecraft_clock_start_count to be defined. This must be a string
        containing the start clock count of the spacecraft

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        return spice.scs2e(self.spacecraft_id, self.spacecraft_clock_start_count)

    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris stop time of the image. Expects spacecraft_id to
        be defined. This must be the integer Naif Id code for the spacecraft.
        Expects spacecraft_clock_stop_count to be defined. This must be a string
        containing the stop clock count of the spacecraft

        Returns
        -------
        : double
          Ephemeris stop time of the image
        """
        return spice.scs2e(self.spacecraft_id, self.spacecraft_clock_stop_count)

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
        return float(spice.gdpool('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), 0, 1)[0])

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
        return float(spice.gdpool('INS{}_BORESIGHT_LINE'.format(self.ikid), 0, 1)[0])


    @property
    def swap_observer_target(self):
        """
        Returns if the observer and target should be swapped when determining the
        sensor state relative to the target. This is defined by a keyword in
        ISIS IAKs. If the keyword is not defined in any loaded kernels then False
        is returned.

        Expects ikid to be defined. This should be an integer containing the
        Naif Id code of the instrument.
        """
        try:
            swap = spice.gcpool('INS{}_SWAP_OBSERVER_TARGET'.format(self.ikid), 0, 1)[0]
            return swap.upper() == "TRUE"
        except:
            return False

    @property
    def correct_lt_to_surface(self):
        """
        Returns if light time correction should be made to the surface instead of
        to the center of the body. This is defined by a keyword in ISIS IAKs.
        If the keyword is not defined in any loaded kernels then False is returned.

        Expects ikid to be defined. This should be an integer containing the
        Naif Id code of the instrument.
        """
        try:
            surface_correct = spice.gcpool('INS{}_LT_SURFACE_CORRECT'.format(self.ikid), 0, 1)[0]
            return surface_correct.upper() == "TRUE"
        except:
            return False

    @property
    def naif_keywords(self):
        """
        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        if not hasattr(self, "_naif_keywords"):
            self._naif_keywords = dict()

            self._naif_keywords['BODY{}_RADII'.format(self.target_id)] = self.target_body_radii
            self._naif_keywords['BODY_FRAME_CODE'] = self.target_frame_id
            self._naif_keywords['BODY_CODE'] = self.target_id

            self._naif_keywords = {**self._naif_keywords, **util.query_kernel_pool(f"*{self.ikid}*"),  **util.query_kernel_pool(f"*{self.target_id}*")}

            try:
                self._naif_keywords = {**self._naif_keywords, **util.query_kernel_pool(f"*{self.fikid}*")}
            except AttributeError as error:
                pass

        return self._naif_keywords
