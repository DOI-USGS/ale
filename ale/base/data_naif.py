import spiceypy as spice 

import warnings
from multiprocessing.pool import ThreadPool


import numpy as np
import pyspiceql
import scipy.constants
from scipy.spatial.transform import Rotation as R

import ale
from ale.base import spiceql_mission_map
from ale.transformation import FrameChain
from ale.rotation import TimeDependentRotation
from ale import kernel_access
from ale import spiceql_access
from ale import spice_root
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
            [pyspiceql.load(k) for k in self.kernels]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context goes out of scope. Once
        this is done, the object is out of scope and the
        kernels can be unloaded.
        """
        if self.kernels:
            [pyspiceql.unload(k) for k in self.kernels]

    @property
    def kernels(self):
        """
        Get the NAIF SPICE Kernels to furnish

        There are three ways to specify which kernels a driver will use:

        1. Passing the 'kernels' property into load(s) or at instantiation.
           This can be either a straight iterable or a dictionary that specifies
           the kernels in ISIS style ('TargetPosition', 'InstrumentPosition', etc).
        2. Set the ALESPICEROOT environment variable. This variable should be
           the path to a directory that contains directories whose naming
           convention matches the PDS Kernel Archives format,
           `shortMissionName-versionInfo`. The directory corresponding to the
           driver's mission will be searched for the appropriate meta kernel to
           load.
        3. Default to using SpiceQL for extracting spice information which does not
           require kernels to be set. If used with `web: False` the user will have to
           specifiy either `SPICEDATA, ALESPICEROOT, or ISISDATA` ato point to there
           data area

        See Also
        --------
        ale.kernel_access.get_kernels_from_isis_pvl : Function used to parse ISIS style dict
        ale.kernel_access.get_metakernels : Function that searches ALESPICEROOT for meta kernels
        ale.kernel_access.generate_kernels_from_cube : Helper function to get an ISIS style dict
                                                       from an ISIS cube that has been through
                                                       spiceinit

        """
        if not hasattr(self, '_kernels'):
            if 'kernels' in self._props.keys():
                try:
                    self._kernels = kernel_access.get_kernels_from_isis_pvl(self._props['kernels'])
                except Exception as e:
                    self._kernels =  self._props['kernels']
            elif ale.spice_root:
                search_results = kernel_access.get_metakernels(ale.spice_root, missions=self.short_mission_name, years=self.utc_start_time.year, versions='latest')

                if search_results['count'] == 0:
                    raise ValueError(f'Failed to find metakernels. mission: {self.short_mission_name}, year:{self.utc_start_time.year}, versions="latest" spice root = "{ale.spice_root}"')
                self._kernels = [search_results['data'][0]['path']]
            else:
                self._kernels = []

        return self._kernels

    @property
    def use_web(self):
        """
        Reads the web property in the props dictionary to define the use_web value.
        This property dictates if you are running in a web enabled driver

        Returns
        -------
        : bool
          Boolean defining if you are running web enabled(True) or Disabled(False)
        """
        if not hasattr(self, '_use_web'):
            self._use_web = False

            if "web" in self._props.keys():
                web_prop = self._props["web"]
                if not isinstance(web_prop, bool):
                    warnings.warn(f"Web value {web_prop} not a boolean type, setting web to False")
                    web_prop = False
                self._use_web = web_prop

        return self._use_web

    @property
    def search_kernels(self):
        if not hasattr(self, "_search_kernels"):
            self._search_kernels = False
            if not self.kernels and self.use_web:
                self._search_kernels = True
        return self._search_kernels
                
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
                self._light_time_correction = self.naif_keywords['INS{}_LIGHTTIME_CORRECTION'.format(self.ikid)]
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
            self._odtx = self.naif_keywords['INS{}_OD_T_X'.format(self.ikid)]
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
            self._odty = self.naif_keywords['INS{}_OD_T_Y'.format(self.ikid)]
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
            self._odtk = self.naif_keywords['INS{}_OD_K'.format(self.ikid)]
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
            self._ikid = self.spiceql_call("translateNameToCode", {"frame": self.instrument_id, "mission": self.spiceql_mission})
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
            self._spacecraft_id = self.spiceql_call("translateNameToCode", {"frame": self.spacecraft_name, "mission": self.spiceql_mission})
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
            self._target_id = self.spiceql_call("translateNameToCode", {"frame": self.target_name, "mission": self.spiceql_mission})
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
            frame_info = self.spiceql_call("getTargetFrameInfo", {"targetId": self.target_id, "mission": self.spiceql_mission})
            self._target_frame_id = frame_info["frameCode"]
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
            self._focal2pixel_lines = self.naif_keywords['INS{}_ITRANSL'.format(self.ikid)]
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
            self._focal2pixel_samples = self.naif_keywords['INS{}_ITRANSS'.format(self.ikid)]
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
            self._pixel2focal_x = self.naif_keywords['INS{}_TRANSX'.format(self.ikid)]
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
            self._pixel2focal_y = self.naif_keywords['INS{}_TRANSY'.format(self.ikid)]
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
            self._focal_length = self.naif_keywords['INS{}_FOCAL_LENGTH'.format(self.ikid)]
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
            self._pixel_size = self.naif_keywords['INS{}_PIXEL_SIZE'.format(self.ikid)][0] * 0.001
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
            self._target_body_radii = self.naif_keywords[f"BODY{self.target_id}_RADII"]
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
                frame_info = self.spiceql_call("getTargetFrameInfo", {"targetId": self.target_id, "mission": self.spiceql_mission})
                self._reference_frame = frame_info["frameName"]
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
            times = self.ephemeris_time
            if len(times) > 1:
                times = np.array([times[0], times[-1]])
            positions = []
            velocities = []

            sun_lt_states = self.spiceql_call("getTargetStates",{"ets": times,
                                                                 "target": "SUN",
                                                                 "observer": self.target_name,
                                                                 "frame": self.reference_frame,
                                                                 "abcorr": "LT+S",
                                                                 "mission": self.spiceql_mission})
            for sun_state in sun_lt_states:
                sun_state = np.array(sun_state)
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
        if not hasattr(self, '_position') or \
           not hasattr(self, '_velocity') or \
           not hasattr(self, '_ephem'):
            ephem = self.ephemeris_time

            # if isinstance(ephem, np.ndarray):
            #     ephem = ephem.tolist()
            #     print("[data_naif.sensor_position] ephem is an ndarray")
            # print("[data_naif.sensor_position] ephem type: " + str(type(ephem)))

            pos = []
            vel = []

            target = self.spacecraft_name
            observer = self.target_name
            ## Check for ISIS flag to fix target and observer swapping
            print(self.swap_observer_target)
            if self.swap_observer_target:
                target = self.target_name
                observer = self.spacecraft_name

            # spkezr returns a vector from the observer's location to the aberration-corrected
            # location of the target. For more information, see:
            # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/spkezr.html
            if self.correct_lt_to_surface and self.light_time_correction.upper() == 'LT+S':
                position_tasks = []
                kwargs = {"target": target,
                          "observer": observer,
                          "frame": "J2000",
                          "abcorr": self.light_time_correction,
                          "mission": self.spiceql_mission,
                          "searchKernels": self.search_kernels}
                print(kwargs)
                position_tasks.append([ephem, "getTargetStates", 400, self.use_web, kwargs])

                # ssb to spacecraft
                kwargs = {"target": observer,
                          "observer": "SSB",
                          "frame": "J2000",
                          "abcorr": "NONE",
                          "mission": self.spiceql_mission,
                          "searchKernels": self.search_kernels}
                print(kwargs)
                position_tasks.append([ephem, "getTargetStates", 400, self.use_web, kwargs])

                # Build graph async
                with ThreadPool() as pool:
                    jobs = pool.starmap_async(spiceql_access.get_ephem_data, position_tasks)
                    results = jobs.get()

                obs_tars, ssb_obs = results

                obs_tar_lts = np.array(obs_tars)[:,-1]
                ssb_obs_states = np.array(ssb_obs)[:,0:6]

                radius_lt = (self.target_body_radii[2] + self.target_body_radii[0]) / 2 / (scipy.constants.c/1000.0)
                adjusted_time = ephem - obs_tar_lts + radius_lt

                kwargs = {"target": target,
                          "observer": "SSB",
                          "frame": "J2000",
                          "abcorr": "NONE",
                          "mission": self.spiceql_mission,
                          "searchKernels": self.search_kernels}
                print(kwargs)
                ssb_tars = spiceql_access.get_ephem_data(adjusted_time, "getTargetStates", web=self.use_web, function_args=kwargs)
                ssb_tar_states = np.array(ssb_tars)[:,0:6]

                _states = ssb_tar_states - ssb_obs_states


                states = []
                for i, state in enumerate(_states):
                    matrix = spice.sxform("J2000", self.reference_frame, ephem[i])
                    rotated_state = spice.mxvg(matrix, state)
                    states.append(rotated_state)
            else:
                kwargs = {"target": target,
                          "observer": observer,
                          "frame": self.reference_frame,
                          "abcorr": self.light_time_correction,
                          "mission": self.spiceql_mission,
                          "searchKernels": self.search_kernels}
                print(kwargs)
                states = spiceql_access.get_ephem_data(ephem, "getTargetStates", web=self.use_web, function_args=kwargs)
                states = np.array(states)[:,0:6]

            for state in states:
                if self.swap_observer_target:
                    pos.append(-state[:3])
                    vel.append(-state[3:])
                else:
                    pos.append(state[:3])
                    vel.append(state[3:])


            # By default, SPICE works in km, so convert to m
            self._position = 1000 * np.asarray(pos)
            self._velocity = 1000 * np.asarray(vel)
            self._ephem = ephem
        return self._position, self._velocity, self._ephem

    @property
    def frame_chain(self):
        """
        Return the root node of the rotation frame tree/chain.
        
        Returns
        -------
        FrameNode
            The root node of the frame tree. This will always be the J2000 reference frame.
        """

        if not hasattr(self, '_frame_chain'):
            nadir = self._props.get('nadir', False)
            exact_ck_times = self._props.get('exact_ck_times', True)
            self._frame_chain = FrameChain.from_spice(sensor_frame=self.sensor_frame_id,
                                                      target_frame=self.target_frame_id,
                                                      center_ephemeris_time=self.center_ephemeris_time,
                                                      ephemeris_times=self.ephemeris_time,
                                                      nadir=nadir, exact_ck_times=exact_ck_times,
                                                      inst_time_bias=self.instrument_time_bias,
                                                      mission=self.spiceql_mission,
                                                      use_web=self.use_web,
                                                      search_kernels=self.search_kernels)

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
                trans_x_key = f"INS{self.ikid}_ITRANSL"
                trans_x = self.spiceql_call("findMissionKeywords", {"key": trans_x_key, 
                                                                    "mission": self.spiceql_mission})[trans_x_key]

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
            self._ephemeris_start_time = self.spiceql_call("strSclkToEt", {"frameCode": self.spacecraft_id, "sclk": self.spacecraft_clock_start_count, "mission": self.spiceql_mission})
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
            self._ephemeris_stop_time = self.spiceql_call("strSclkToEt", {"frameCode": self.spacecraft_id, "sclk": self.spacecraft_clock_stop_count, "mission": self.spiceql_mission})
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
            self._detector_center_sample = self.naif_keywords['INS{}_BORESIGHT_SAMPLE'.format(self.ikid)]
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
            self._detector_center_line = self.naif_keywords['INS{}_BORESIGHT_LINE'.format(self.ikid)]
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
                swap = self.naif_keywords['INS{}_SWAP_OBSERVER_TARGET'.format(self.ikid)]
                if isinstance(swap, str):
                    self._swap_observer_target = swap.upper() == "TRUE"
                elif isinstance(swap, bool):
                    self._swap_observer_target = swap
                else:
                    raise Exception(f"Cannot decode swap observer target value {swap}")
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
                surface_correct = self.naif_keywords['INS{}_LT_SURFACE_CORRECT'.format(self.ikid)]
                if isinstance(surface_correct, str):
                    self._correct_lt_to_surface = surface_correct.upper() == "TRUE"
                elif isinstance(surface_correct, bool):
                    self._correct_lt_to_surface = surface_correct
                else:
                    raise Exception(f"Cannot decode LT surface correct value {surface_correct}")
            except Exception as e:
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

            self._naif_keywords['BODY_FRAME_CODE'] = self.target_frame_id
            self._naif_keywords['BODY_CODE'] = self.target_id
            mission_keywords = self.spiceql_call("findMissionKeywords", {"key": f"*{self.ikid}*", "mission": self.spiceql_mission})
            target_keywords = self.spiceql_call("findTargetKeywords", {"key": f"*{self.target_id}*", "mission": self.spiceql_mission})

            if mission_keywords: 
                self._naif_keywords = self.naif_keywords | mission_keywords
            if target_keywords: 
                self._naif_keywords = self._naif_keywords | target_keywords
            
            try:
                frame_keywords = self.spiceql_call("findMissionKeywords", {"key": f"*{self.fikid}*", "mission": self.spiceql_mission})
                if frame_keywords: 
                    self._naif_keywords = self._naif_keywords | frame_keywords 
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
    
    @property
    def spiceql_mission(self):
        """
        Access the mapping between a SpiceQL "mission" and the driver.
        The mapping can be found under ale.base.__init__.py

        See Also
        --------
        ale.base.__init__.py
        """
        return spiceql_mission_map[self.instrument_id]

    def spiceql_call(self, function_name = "", function_args = {}):
        """
        Driver based wrapper for accessing spice data through spiceql

        See Also
        --------
        ale.kernel_access.spiceql_call

        Returns
        -------
        : json
          Json data from the SpiceQL call
        """
        # This will work if a user passed no kernels but still set ISISDATA
        # just might take a bit
        function_args["searchKernels"] = self.search_kernels
        
        # Bodge solution for memo funcs in offline mode
        memo_funcs = ["translateNameToCode", "translateCodeToName"]
        
        try:
            data_dir = pyspiceql.getDataDirectory()
        except Exception as e:
            data_dir = ""

        if function_name in memo_funcs and data_dir == "" and self.use_web == False:
            function_name = f"{function_name}"
        return spiceql_access.spiceql_call(function_name, function_args, self.use_web)
    
