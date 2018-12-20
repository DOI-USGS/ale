from dateutil import parser
import numpy as np
import pvl
import spiceypy as spice

class Driver():
    """
    Base class for all Drivers.

    """
    def __init__(self, file):
        self._file = file

    def __str__(self):
        return str(self.to_dict())

    def is_valid(self):
        try:
            iid = self.instrument_id
            return True
        except Exception as e:
            return False

    def to_dict(self):
        keys = set(dir(self)) & self.required_keys
        return {p:getattr(self, p) for p in keys}


class LineScanner(Driver):
    @property
    def name_model(self):
        return "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL"

    @property
    def t0_ephemeris(self):
        return self.starting_ephemeris_time - self.center_ephemeris_time

    @property
    def t0_quaternion(self):
        return self.starting_ephemeris_time - self.center_ephemeris_time

    @property
    def dt_ephemeris(self):
        return (self.ending_ephemeris_time - self.starting_ephemeris_time) / self.number_of_ephemerides

    @property
    def dt_quaternion(self):
        return (self.ending_ephemeris_time - self.starting_ephemeris_time) / self.number_of_ephemerides

    @property
    def line_scan_rate(self):
        """
        In the form: [start_line, line_time, exposure_duration]
        The form below is for a fixed rate line scanner.
        """
        return [[float(self.starting_detector_line), self.t0_ephemeris, self.line_exposure_duration]]

    @property
    def number_of_ephemerides(self):
        #TODO: Not make this hardcoded
        return 909

    @property
    def number_of_quaternions(self):
        #TODO: Not make this hardcoded
        return 909


class Framer(Driver):
    @property
    def name_model(self):
        return "USGS_ASTRO_FRAME_SENSOR_MODEL"

    @property
    def filter_number(self):
        return self.label.get('FILTER_NUMBER', 0)

    @property
    def number_of_ephemerides(self):
        # always one for framers
        return 1

    @property
    def number_of_quaternions(self):
        # always one for framers
        return 1


class PDS3():
    def _compute_ephemerides(self):
        """
        Helper function to pull position and velocity in one pass
        so that the results can then be cached in the associated
        properties.
        """
        eph = np.empty((self.number_of_ephemerides, 3))
        eph_rates = np.empty(eph.shape)
        current_et = self.starting_ephemeris_time
        for i in range(self.number_of_ephemerides):
            state, _ = spice.spkezr(self.spacecraft_name,
                                    current_et,
                                    self.reference_frame,
                                    'NONE',
                                    self.target_name,) # If this is the sensor, insufficient, if this is the spacecraft, it works? Huh?
            eph[i] = state[:3]
            eph_rates[i] = state[3:]
            current_et += getattr(self, 'dt_ephemeris', 0)
        # By default, spice works in km
        eph *= 1000
        eph_rates *= 1000
        self._sensor_velocity = eph_rates
        self._sensor_position = eph

    @property
    def label(self):
        if not hasattr(self, "_label"):
            if isinstance(self._file, pvl.PVLModule):
                self._label = label
            try:
                self._label = pvl.loads(self._file, strict=False)
            except AttributeError:
                self._label = pvl.load(self._file, strict=False)
            except:
                raise Exception("{} is not a valid label".format(label))
        return self._label

    @property
    def line_exposure_duration(self):
        try:
            return self.label['LINE_EXPOSURE_DURATION'].value * 0.001  # Scale to seconds
        except:
            return np.nan

    @property
    def instrument_id(self):
        pass

    @property
    def start_time(self):
        return self.label['START_TIME']

    @property
    def image_lines(self):
        return self.label['IMAGE']['LINES']

    @property
    def image_samples(self):
        return self.label['IMAGE']['LINE_SAMPLES']

    @property
    def interpolation_method(self):
        return 'lagrange'

    @property
    def target_name(self):
        return self.label['TARGET_NAME']

    @property
    def _target_id(self):
        return spice.bodn2c(self.label['TARGET_NAME'])

    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            sclock = self.label['SPACECRAFT_CLOCK_START_COUNT']
            self._starting_ephemeris_time = spice.scs2e(self.spacecraft_id, sclock)
        return self._starting_ephemeris_time

    @property
    def exposure_duration(self):
        try:
            return self.label['EXPOSURE_DURATION'].value * 0.001  # Scale to seconds
        except:
            return np.nan

    @property
    def spacecraft_clock_stop_count(self):
        sc = self.label.get('SPACECRAFT_CLOCK_STOP_COUNT', None)
        if sc == 'N/A':
            sc = None
        return sc

    @property
    def ending_ephemeris_time(self):
        return (self.image_lines * self.line_exposure_duration) + self.starting_ephemeris_time

    @property
    def center_ephemeris_time(self):
        return (self.starting_ephemeris_time + self.ending_ephemeris_time)/2


    @property
    def detector_center(self):
        return list(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2))

    @property
    def spacecraft_name(self):
        return self.label['MISSION_NAME']

    @property
    def starting_detector_line(self):
        return 1

    @property
    def starting_detector_sample(self):
        return 1

    @property
    def detector_sample_summing(self):
        return 1

    @property
    def detector_line_summing(self):
        return self.label.get('SAMPLING_FACTOR', 1)


class Spice():
    @property
    def metakernel(self):
        pass

    def __enter__(self):
        """
        Called when the context is created. This is used
        to get the kernels furnished.
        """
        if self.metakernel:
            spice.furnsh(self.metakernel)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context goes out of scope. Once
        this is done, the object is out of scope and the
        kernels can be unloaded.
        """
        spice.unload(self.metakernel)

    @property
    def odtx(self):
        return spice.gdpool('INS{}_OD_T_X'.format(self.ikid),0, 10)

    @property
    def odty(self):
        return spice.gdpool('INS{}_OD_T_Y'.format(self.ikid), 0, 10)

    @property
    def odtk(self):
        return spice.gdpool('INS{}_OD_K'.format(self.ikid),0, 3)

    @property
    def ikid(self):
        return spice.bods2c(self.instrument_id)

    @property
    def spacecraft_id(self):
        return spice.bods2c(self.spacecraft_name)

    @property
    def focal2pixel_lines(self):
        return spice.gdpool('INS{}_ITRANSL'.format(self.fikid), 0, 3)

    @property
    def focal2pixel_samples(self):
        return spice.gdpool('INS{}_ITRANSS'.format(self.fikid), 0, 3)

    @property
    def focal_length(self):
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

    @property
    def semimajor(self):
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1][1]

    @property
    def semiminor(self):
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1][0]

    @property
    def reference_frame(self):
        return 'IAU_{}'.format(self.target_name)

    @property
    def sun_position(self):
        sun_state, _ = spice.spkezr("SUN",
                                     self.center_ephemeris_time,
                                     self.reference_frame,
                                     'NONE',
                                     self.target_name)

        return [sun_state[:4].tolist()]

    @property
    def sun_velocity(self):
        sun_state, lt = spice.spkezr("SUN",
                                     self.center_ephemeris_time,
                                     self.reference_frame,
                                     'NONE',
                                     self.target_name)

        return [sun_state[3:6].tolist()]

    @property
    def sensor_position(self):
        if not hasattr(self, '_sensor_position'):
            self._compute_ephemerides()
        return self._sensor_position.tolist()

    @property
    def sensor_velocity(self):
        if not hasattr(self, '_sensor_velocity'):
            self._compute_ephemerides()
        return self._sensor_velocity.tolist()

    @property
    def sensor_orientation(self):
        if not hasattr(self, '_sensor_orientation'):
            current_et = self.starting_ephemeris_time
            qua = np.empty((self.number_of_ephemerides, 4))
            for i in range(self.number_of_quaternions):
                # Find the rotation matrix
                camera2bodyfixed = spice.pxform(self.instrument_id,
                                                self.reference_frame,
                                                current_et)
                q = spice.m2q(camera2bodyfixed)
                qua[i,:3] = q[1:]
                qua[i,3] = q[0]
                current_et += getattr(self, 'dt_quaternion', 0)
            self._sensor_orientation = qua
        return self._sensor_orientation.tolist()

    @property
    def reference_height(self):
        # TODO: This should be a reasonable #
        return 0, 100

    @property
    def detector_center(self):
        if not hasattr(self, '_detector_center'):
            center_line = float(spice.gdpool('INS{}_BORESIGHT_LINE'.format(self.ikid), 0, 1)[0])
            center_sample = float(spice.gdpool('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), 0, 1)[0])
            self._detector_center = [center_line, center_sample]
        return self._detector_center

    @property
    def center_ephemeris_time(self):
        """
        The center ephemeris time for a fixed rate line scanner.
        """
        if not hasattr(self, '_center_ephemeris_time'):
            halflines = self.image_lines / 2
            center_sclock = self.starting_ephemeris_time + halflines * self.line_exposure_duration
            self._center_ephemeris_time = center_sclock
        return self._center_ephemeris_time

    @property
    def fikid(self):
        if isinstance(self, Framer):
            fn = self.filter_number
            if fn == 'N/A':
                fn = 0
        else:
            fn = 0

        return self.ikid - int(fn)


class Isis3():
    @property
    def label(self):
        if not hasattr(self, "_label"):
            if isinstance(self._file, pvl.PVLModule):
                self._label = label
            try:
                self._label = pvl.loads(self._file, strict=False)
            except AttributeError:
                self._label = pvl.load(self._file, strict=False)
            except:
                raise Exception("{} is not a valid label".format(label))
        return self._label

    @property
    def start_time(self):
        return self.label['IsisCube']['Instrument']['StartTime']

    @property
    def spacecraft_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def image_lines(self):
        return self.label['IsisCube']['Core']['Dimensions']['Lines']

    @property
    def image_samples(self):
        return self.label['IsisCube']['Core']['Dimensions']['Samples']

    @property
    def _exposure_duration(self):
        return self.label['IsisCube']['Instrument']['ExposureDuration'].value * 0.001 # Scale to seconds

    @property
    def target_name(self):
        return self.label['IsisCube']['Instrument']['TargetName']

    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            sclock = self.label['IsisCube']['Archive']['SpacecraftClockStartCount']
            self._starting_ephemeris_time = spice.scs2e(self.spacecraft_id, sclock)
        return self._starting_ephemeris_time
