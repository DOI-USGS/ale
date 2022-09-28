import spiceypy as spice
from pyspiceql import pyspiceql
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
            [pyspiceql.KernelPool.getInstance().load(k) for k in self.kernels]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context goes out of scope. Once
        this is done, the object is out of scope and the
        kernels can be unloaded.
        """
        if self.kernels:
            [pyspiceql.KernelPool.getInstance().unload(k) for k in self.kernels]

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
        if not hasattr(self, "_light_time_correction"):
            try:
                self._light_time_correction = pyspiceql.getKernelStringValue('INS{}_LIGHTTIME_CORRECTION'.format(self.ikid))[0]
            except:
                self._light_time_correction = 'LT+S'
        return self._light_time_correction

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
        if not hasattr(self, "_odtx"):
            self._odtx = pyspiceql.getKernelVectorValue('INS{}_OD_T_X'.format(self.ikid)).toList()
        return self._odtx

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
        if not hasattr(self, "_odty"):
            self._odty = pyspiceql.getKernelVectorValue('INS{}_OD_T_Y'.format(self.ikid)).toList()
        return self._odty

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
        if not hasattr(self, "_odtk"):
            self._odtk = list(pyspiceql.getKernelVectorValue('INS{}_OD_K'.format(self.ikid)))
        return self._odtk

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
        if not hasattr(self, "_ikid"):
            self._ikid = pyspiceql.Kernel_translateFrame(self.instrument_id)
        return self._ikid

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
        if not hasattr(self, "_spacecraft_id"):
            self._spacecraft_id = pyspiceql.Kernel_translateFrame(self.spacecraft_name)
        return self._spacecraft_id

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
        if not hasattr(self, "_target_id"):
            self._target_id = pyspiceql.Kernel_translateFrame(self.target_name)
        return self._target_id

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
        if not hasattr(self, "_target_frame_id"):
            frame_info = spice.cidfrm(self.target_id)
            self._target_frame_id = frame_info[0]
        return self._target_frame_id

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
        if not hasattr(self, "_sensor_frame_id"):
            self._sensor_frame_id = self.ikid
        return self._sensor_frame_id

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        if not hasattr(self, "_focal2pixel_lines"):
            self._focal2pixel_lines = list(pyspiceql.getKernelVectorValue('INS{}_ITRANSL'.format(self.ikid)))
        return self._focal2pixel_lines

    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        if not hasattr(self, "_focal2pixel_samples"):
            self._focal2pixel_samples = list(pyspiceql.getKernelVectorValue('INS{}_ITRANSS'.format(self.ikid)))
        return self._focal2pixel_samples

    @property
    def pixel2focal_x(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane x
        """
        if not hasattr(self, "_pixel2focal_x"):
            self._pixel2focal_x = list(pyspiceql.getKernelVectorValue('INS{}_ITRANSX'.format(self.ikid)))
        return self._pixel2focal_x

    @property
    def pixel2focal_y(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane y
        """
        if not hasattr(self, "_pixel2focal_y"):
            self._pixel2focal_y = list(pyspiceql.getKernelVectorValue('INS{}_ITRANSY'.format(self.ikid)))
        return self._pixel2focal_y

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
        if not hasattr(self, "_focal_length"):
            self._focal_length = float(pyspiceql.getKernelVectorValue('INS{}_FOCAL_LENGTH'.format(self.ikid))[0])
        return self._focal_length

    @property
    def pixel_size(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : float pixel size
        """
        if not hasattr(self, "_pixel_size"):
            self._pixel_size = pyspiceql.getKernelVectorValue('INS{}_PIXEL_SIZE'.format(self.ikid))[0] * 0.001
        return self._pixel_size

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
        if not hasattr(self, "_target_body_radii"):
            self._target_body_radii = spice.bodvrd(self.target_name, 'RADII', 3)[1]
        return self._target_body_radii

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
        if not hasattr(self, "_reference_frame"):
            try:
                self._reference_frame = spice.cidfrm(spice.bodn2c(self.target_name))[1]
            except:
                self._reference_frame = 'IAU_{}'.format(self.target_name)
        return self._reference_frame

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
        if not hasattr(self, "_sun_position"):
            times = [self.center_ephemeris_time]
            if len(times) > 1:
                times = [times[0], times[-1]]
            positions = []
            velocities = []

            for time in times:
                sun_lt_state = pyspiceql.getTargetState(time,
                                                        self.target_name,
                                                        self.spacecraft_name,
                                                        'J2000',
                                                        self.light_time_correction)
                sun_state = np.array(list(sun_lt_state.starg))
                positions.append(sun_state[:3])
                velocities.append(sun_state[3:6])
            positions = 1000 * np.asarray(positions)
            velocities = 1000 * np.asarray(velocities)

            self._sun_position = positions, velocities, times
        return self._sun_position

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
            ## Check for ISIS flag to fix target and observer swapping
            if self.swap_observer_target:
                target = self.target_name
                observer = self.spacecraft_name

            for time in ephem:
                # spkezr returns a vector from the observer's location to the aberration-corrected
                # location of the target. For more information, see:
                # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/spkezr.html
                if self.correct_lt_to_surface and self.light_time_correction.upper() == 'LT+S':
                    obs_tar = pyspiceql.getTargetState(time,
                                                       target,
                                                       observer,
                                                       'J2000',
                                                       self.light_time_correction)
                    obs_tar_lt = obs_tar.lt

                    # ssb to spacecraft
                    ssb_obs = pyspiceql.getTargetState(time,
                                                       target,
                                                       'SSB',
                                                       'J2000',
                                                       "NONE")
                    ssb_obs_state = np.array(list(ssb_obs.starg))

                    radius_lt = (self.target_body_radii[2] + self.target_body_radii[0]) / 2 / (scipy.constants.c/1000.0)
                    adjusted_time = time - obs_tar_lt + radius_lt

                    ssb_tar = pyspiceql.getTargetState(adjusted_time,
                                                       target,
                                                       'SSB',
                                                       'J2000',
                                                       "NONE")
                    ssb_tar_state = np.array(list(ssb_tar.starg))
                    state = ssb_tar_state - ssb_obs_state

                    matrix = spice.sxform("J2000", self.reference_frame, time)
                    state = spice.mxvg(matrix, state)
                else:
                    state = pyspiceql.getTargetState(time,
                                                     target,
                                                     observer,
                                                     self.reference_frame,
                                                     self.light_time_correction)
                    state = np.array(list(state.starg))

                if self.swap_observer_target:
                    pos.append(-state[:3])
                    vel.append(-state[3:])
                else:
                    pos.append(state[:3])
                    vel.append(state[3:])


            # By default, SPICE works in km, so convert to m
            self._position = 1000 * np.asarray(pos)
            self._velocity = 1000 * np.asarray(vel)
        return self._position, self._velocity, ephem

    @property
    def frame_chain(self):
        if not hasattr(self, '_frame_chain'):
            nadir = self._props.get('nadir', False)
            exact_ck_times = self._props.get('exact_ck_times', True)
            self._frame_chain = FrameChain.from_spice(sensor_frame=self.sensor_frame_id,
                                                      target_frame=self.target_frame_id,
                                                      center_ephemeris_time=self.center_ephemeris_time,
                                                      ephemeris_times=self.ephemeris_time,
                                                      nadir=nadir, exact_ck_times=exact_ck_times,
                                                      inst_time_bias=self.instrument_time_bias)

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
                trans_x = np.array(self.focal2pixel_lines)

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
        if not hasattr(self, "_ephemeris_start_time"):
            self._ephemeris_start_time = pyspiceql.sclkToEt(str(self.spacecraft_name), self.spacecraft_clock_start_count)
        return self._ephemeris_start_time

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
        if not hasattr(self, "_ephemeris_stop_time"):
            self._ephemeris_stop_time = pyspiceql.sclkToEt(self.spacecraft_name, self.spacecraft_clock_stop_count)
        return self._ephemeris_stop_time

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
        if not hasattr(self, "_detector_center_sample"):
            self._detector_center_sample = float(pyspiceql.getKernelStringValue('INS{}_BORESIGHT_SAMPLE'.format(self.ikid))[0])
        return self._detector_center_sample

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
        if not hasattr(self, "_detector_center_line"):
            self._detector_center_line = float(pyspiceql.getKernelStringValue('INS{}_BORESIGHT_LINE'.format(self.ikid))[0])
        return self._detector_center_line

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
        if not hasattr(self, "_swap_observer_target"):
            try:
                swap = pyspiceql.getKernelStringValue('INS{}_SWAP_OBSERVER_TARGET'.format(self.ikid))[0]
                self._swap_observer_target = swap.upper() == "TRUE"
            except:
                self._swap_observer_target = False
        return self._swap_observer_target

    @property
    def correct_lt_to_surface(self):
        """
        Returns if light time correction should be made to the surface instead of
        to the center of the body. This is defined by a keyword in ISIS IAKs.
        If the keyword is not defined in any loaded kernels then False is returned.

        Expects ikid to be defined. This should be an integer containing the
        Naif Id code of the instrument.
        """
        if not hasattr(self, "_correct_lt_to_surface"):
            try:
                surface_correct = pyspiceql.getKernelStringValue('INS{}_LT_SURFACE_CORRECT'.format(self.ikid))[0]
                self._correct_lt_to_surface = surface_correct.upper() == "TRUE"
            except:
                self._correct_lt_to_surface = False
        return self._correct_lt_to_surface

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

    @property
    def instrument_time_bias(self):
        """
        Time bias used for generating sensor orientations

        The default is 0 for not time bias

        Returns
        -------
        : int
          Time bias in ephemeris time
        """
        return 0