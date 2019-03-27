import os
import re
import struct
from glob import glob

import numpy as np
from dateutil import parser

import pvl
import spiceypy as spice
from ale import config


def read_table_data(table_label, cube):
    """
    Helper function to read all of the binary table data

    Parameters
    ----------
    table_label : PVLModule
                  The ISIS table label
    cube : file
           The ISIS cube file

    Returns
    -------
    bytes :
        The binary portion of the table data
    """
    cubehandle = open(cube, "rb")
    cubehandle.seek(table_label['StartByte'])
    return cubehandle.read(table_label['Bytes'])

def field_size(field_label):
    """
    Helper function to determine the size of a binary
    table field

    Parameters
    ----------
    field_label : PVLModule
                  The field label

    Returns
    -------
    int :
        The size of the one entry in bytes
    """
    data_sizes = {
        'Integer' : 4,
        'Double'  : 8,
        'Real'    : 4,
        'Text'    : 1
    }
    return data_sizes[field_label['Type']] * field_label['Size']

def field_format(field_label):
    """
    Helper function to get the format string for a
    single entry in a table field

    Parameters
    ----------
    field_label : PVLModule
                  The field label

    Returns
    -------
    str :
        The format string for the entry binary
    """
    data_formats = {
        'Integer' : 'i',
        'Double'  : 'd',
        'Real'    : 'f'
    }
    return data_formats[field_label['Type']] * field_label['Size']

def parse_field(field_label, data, encoding='latin_1'):
    """
    Parses a binary table field entry and converts it into
    an in memory data type

    Parameters
    ----------
    field_label : PVLModule
                  The field label

    data : bytes
           The binary data for the field entry

    Returns
    -------
    Union[int, float, str, list] :
        The table field entry converted to a native python
        type
    """
    if field_label['Type'] == 'Text':
        field_data = data[:field_label['Size']].decode(encoding=encoding)
    else:
        data_format = field_format(field_label)
        field_data = struct.unpack_from(data_format, data)
        if len(field_data) == 1:
            field_data = field_data[0]
    return field_data

def parse_table_data(table_label, data):
    """
    Parses an ISIS table into a dict where the keys are the
    field names and the values are lists of entries.

    Parameters
    ----------
    table_label : PVLModule
                  The table label

    data : bytes
           The binary data for the entire table

    Returns
    -------
    dict :
        The table as a dict
    """
    fields = table_label.getlist('Field')
    results = {field['Name']:[] for field in fields}
    offset = 0
    for record in range(table_label['Records']):
        for field in fields:
            field_data = parse_field(field, data[offset:])
            results[field['Name']].append(field_data)
            offset += field_size(field)
    return results

def parse_rotation_table(label, field_data):
    """
    Parses ISIS rotation table data.

    Parameters
    ----------
    table_label : PVLModule
                  The table label

    field_data : dict
                 The table data as a dict with field names
                 as keys and lists of entries as values

    Returns
    -------
    dict :
        The rotation data
    """
    results = {}
    if all (key in field_data for key in ('J2000Q0','J2000Q1','J2000Q2','J2000Q3')):
        results['Rotations'] = [ [q0, q1, q2, q3] for q0, q1, q2, q3 in zip(field_data['J2000Q0'],field_data['J2000Q1'],field_data['J2000Q2'],field_data['J2000Q3']) ]
    if all (key in field_data for key in ('AV1','AV2','AV3')):
        results['AngularVelocities'] = np.array( [ [av1, av2, av3] for av1, av2, av3 in zip(field_data['AV1'],field_data['AV2'],field_data['AV3']) ] )
    if 'ET' in field_data:
        results['Times'] = np.array(field_data['ET'])
    if all (key in field_data for key in ('J2000Ang1','J2000Ang2','J2000Ang3')):
        results['EulerCoefficients'] = np.array([field_data['J2000Ang1'],field_data['J2000Ang2'],field_data['J2000Ang3']])
        results['BaseTime'] = field_data['J2000Ang1'][-1]
        results['TimeScale'] = field_data['J2000Ang2'][-1]

    if 'TimeDependentFrames' in label:
        results['TimeDependentFrames'] = np.array(label['TimeDependentFrames'])
    if all (key in label for key in ('ConstantRotation','ConstantFrames')):
        const_rotation_mat = np.array(label['ConstantRotation'])
        results['ConstantRotation'] = np.reshape(const_rotation_mat, (3, 3))
        results['ConstantFrames'] = np.array(label['ConstantFrames'])
    if all (key in label for key in ('PoleRa','PoleDec','PrimeMeridian')):
        results['BodyRotationCoefficients'] = np.array( [label['PoleRa'],label['PoleDec'],label['PrimeMeridian']] )
    if all (key in label for key in ('PoleRaNutPrec','PoleDecNutPrec','PmNutPrec','SysNutPrec0','SysNutPrec1')):
        results['SatelliteNutationPrecessionCoefficients'] = np.array( [label['PoleRaNutPrec'],label['PoleDecNutPrec'],label['PmNutPrec']] )
        results['PlanetNutationPrecessionAngleCoefficients'] = np.array( [label['SysNutPrec0'],label['SysNutPrec1']] )
    return results

def parse_position_table(field_data):
    """
    Parses ISIS position table data.

    Parameters
    ----------
    table_label : PVLModule
        The table label

    field_data : dict
        The table data as a dict with field names as keys
        and lists of entries as values

    Returns
    -------
    dict :
      The position data
    """
    results = {}
    if all (key in field_data for key in ('J2000X','J2000Y','J2000Z')):
        results['Positions'] = np.array( [ [x, y, z] for x, y, z in zip(field_data['J2000X'],field_data['J2000Y'],field_data['J2000Z']) ] )
    if 'ET' in field_data:
        results['Times'] = np.array(field_data['ET'])
    if all (key in field_data for key in ('J2000XV','J2000YV','J2000ZV')):
        results['Velocities'] = np.array( [ [x, y, z] for x, y, z in zip(field_data['J2000XV'],field_data['J2000YV'],field_data['J2000ZV']) ] )
    if all (key in field_data for key in ('J2000SVX','J2000SVY','J2000SVZ')):
        results['PositionCoefficients'] = np.array( [field_data['J2000SVX'][:-1],field_data['J2000SVY'][:-1],field_data['J2000SVZ'][:-1]] )
        results['BaseTime'] = field_data['J2000SVX'][-1]
        results['TimeScale'] = field_data['J2000SVY'][-1]
    return results


class Driver():
    """
    Base class for all Drivers.

    Attributes
    ----------
    _file : str
            Reference to file path to be used by mixins for opening.
    """
    def __init__(self, file, num_ephem=909, num_quats=909):
        """
        Parameters
        ----------
        file : str
               path to file to be parsed
        """
        self._num_quaternions = num_quats
        self._num_ephem = num_ephem
        self._file = file

    def __str__(self):
        """
        Returns a string representation of the class

        Returns
        -------
        str
            String representation of all attributes and methods of the class
        """
        return str(self.to_dict())

    def is_valid(self):
        """
        Checks if the driver has an intrument id associated with it

        Returns
        -------
        bool
            True if an instrument_id is defined, False otherwise
        """
        try:
            iid = self.instrument_id
            return True
        except Exception as e:
            print(e)
            return False

    def to_dict(self):
        """
        Generates a dictionary of keys based on the attributes and methods assocated with
        the driver and the required keys for the driver

        Returns
        -------
        dict
            Dictionary of key, attribute pairs
        """
        keys = set()
        return {p:getattr(self, p) for p in dir(self) if p[0] != "_" and isinstance(getattr(type(self), p), property)}


    @property
    def label(self):
        if not hasattr(self, "_label"):
            if isinstance(self._file, pvl.PVLModule):
                self._label = self._file
            try:
                self._label = pvl.loads(self._file)
            except Exception:
                self._label = pvl.load(self._file)
            except:
                raise ValueError("{} is not a valid label".format(self._file))
        return self._label

    @property
    def file(self):
        return self._file

    @property
    def interpolation_method(self):
        return "lagrange"

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
        return 1

    @property
    def name_platform(self):
        return "Generic Platform"

    @property
    def name_sensor(self):
        return "Generic Sensor"

    @property
    def radii(self):
        return {
            "semimajor" : self._semimajor,
            "semiminor" : self._semiminor,
            "unit" : "km" # default to KM
        }

    @property
    def reference_height(self):
        # TODO: This should be a reasonable #
        return {
            "minheight" : 0,
            "maxheight": 1000,
            "unit": "m"
        }

    @property
    def focal_length_model(self):
        return {
            "focal_length" : self._focal_length
        }

    @property
    def detector_center(self):
        if not hasattr(self, '_detector_center'):
            self._detector_center = {
                "line" : self._detector_center_line,
                "sample" : self._detector_center_sample
            }
        return self._detector_center

    @property
    def sensor_position(self):
        return {
            "positions" : self._sensor_position,
            "velocities" : self._sensor_velocity,
            "unit" : "m"
        }

    @property
    def sensor_orientation(self):
        return {
            "quaternions" : self._sensor_orientation
        }

    @property
    def sun_position(self):
        return {
            "positions" : self._sun_position,
            "velocities" : self._sun_velocity,
            "unit" : "m"
        }


class LineScanner():
    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
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
        return (self.ending_ephemeris_time - self.starting_ephemeris_time) / self.number_of_quaternions

    @property
    def line_scan_rate(self):
        """
        Returns
        -------
        : list
          2d list of scan rates in the form: [[start_line, line_time, exposure_duration], ...]
        """
        return [[float(self.starting_detector_line), self.t0_ephemeris, self.line_exposure_duration]]

    @property
    def number_of_ephemerides(self):
        return self._num_ephem

    @property
    def number_of_quaternions(self):
        #TODO: Not make this hardcoded
        return self._num_quaternions

    @property
    def ending_ephemeris_time(self):
        return (self.image_lines * self.line_exposure_duration) + self.starting_ephemeris_time

    @property
    def center_ephemeris_time(self):
        return (self.starting_ephemeris_time + self.ending_ephemeris_time)/2

class Framer():
    @property
    def name_sensor(self):
        return "Generic Framer"

class Framer():
    @property
    def name_sensor(self):
        return "Generic Framer"

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
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
    """
    Mixin for reading from PDS3 Labels.

    Attributes
    ----------
    _label : PVLModule
             Dict-like object with PVL keys

    """

    @property
    def _focal_plane_tempature(self):
        return self.label['FOCAL_PLANE_TEMPERATURE'].value

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
        """
        Returns an target name for unquely identifying the instrument, but often
        piped into Spice Kernels to acquire Ephermis data from Spice. Therefore they
        the same ID the Spice expects in bodvrd calls.

        Returns
        -------
        : str
          target name
        """

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
    def _detector_center_line(self):
        return spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[0]

    @property
    def _detector_center_sample(self):
        return spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[1]

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.label['MISSION_NAME']

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
    def _odtx(self):
        """
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_OD_T_X'.format(self.ikid),0, 10).tolist()

    @property
    def _odty(self):
        """
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_OD_T_Y'.format(self.ikid), 0, 10).tolist()

    @property
    def _odtk(self):
        """
        Returns
        -------
        : list
          Radial distortion coefficients
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid),0, 3).tolist()

    @property
    def ikid(self):
        """
        Returns
        -------
        : int
          Naif ID used to for indentifying the instrument in Spice kernels
        """
        return spice.bods2c(self.instrument_id)

    @property
    def spacecraft_id(self):
        return spice.bods2c(self.spacecraft_name)

    @property
    def focal2pixel_lines(self):
        return list(spice.gdpool('INS{}_ITRANSL'.format(self.fikid), 0, 3))

    @property
    def focal2pixel_samples(self):
        return list(spice.gdpool('INS{}_ITRANSS'.format(self.fikid), 0, 3))

    @property
    def _focal_length(self):
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

    @property
    def _semimajor(self):
        """
        Returns
        -------
        : double
          Semimajor axis of the target body
        """
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1][0]

    @property
    def _semiminor(self):
        """
        Returns
        -------
        : double
          Semiminor axis of the target body
        """
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1][2]

    @property
    def reference_frame(self):
        return 'IAU_{}'.format(self.target_name)

    @property
    def _sun_position(self):
        sun_state, _ = spice.spkezr("SUN",
                                     self.center_ephemeris_time,
                                     self.reference_frame,
                                     'NONE',
                                     self.target_name)

        return [sun_state[:4].tolist()]

    @property
    def _sun_velocity(self):
        sun_state, lt = spice.spkezr("SUN",
                                     self.center_ephemeris_time,
                                     self.reference_frame,
                                     'NONE',
                                     self.target_name)

        return [sun_state[3:6].tolist()]

    @property
    def _sensor_position(self):
        if not hasattr(self, '_position'):
            eph = []
            current_et = self.starting_ephemeris_time
            for i in range(self.number_of_ephemerides):
                state, _ = spice.spkezr(self.spacecraft_name,
                                        current_et,
                                        self.reference_frame,
                                        'NONE',
                                        self.target_name,)
                eph.append(state[:3])
                current_et += getattr(self, "dt_ephemeris", 0)
            # By default, spice works in km
            self._position = [e * 1000 for e in eph]
        return self._position

    @property
    def _sensor_velocity(self):
        if not hasattr(self, '_velocity'):
            eph_rates = []
            current_et = self.starting_ephemeris_time
            for i in range(self.number_of_ephemerides):
                state, _ = spice.spkezr(self.spacecraft_name,
                                        current_et,
                                        self.reference_frame,
                                        'NONE',
                                        self.target_name,)
                eph_rates.append(state[3:])
                current_et += getattr(self, "dt_ephemeris", 0)
            # By default, spice works in km
            self._velocity = [e*1000 for e  in eph_rates]
        return self._velocity

    @property
    def _sensor_orientation(self):
        if not hasattr(self, '_orientation'):
            current_et = self.starting_ephemeris_time
            qua = np.empty((self.number_of_quaternions, 4))
            for i in range(self.number_of_quaternions):
                # Find the rotation matrix
                camera2bodyfixed = spice.pxform(self.instrument_id,
                                                self.reference_frame,
                                                current_et)
                q = spice.m2q(camera2bodyfixed)
                qua[i,:3] = q[1:]
                qua[i,3] = q[0]
                current_et += getattr(self, 'dt_quaternion', 0)
            self._orientation = qua
        return self._orientation.tolist()

    @property
    def _detector_center_sample(self):
        return float(spice.gdpool('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), 0, 1)[0])


    @property
    def _detector_center_line(self):
        return float(spice.gdpool('INS{}_BORESIGHT_LINE'.format(self.ikid), 0, 1)[0])

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
    def start_time(self):
        return self.label['IsisCube']['Instrument']['StartTime']

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def image_lines(self):
        """
        Returns
        -------
        : int
          Number of lines in image
        """
        return self.label['IsisCube']['Core']['Dimensions']['Lines']

    @property
    def image_samples(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        return self.label['IsisCube']['Core']['Dimensions']['Samples']

    @property
    def _exposure_duration(self):
        return self.label['IsisCube']['Instrument']['ExposureDuration'].value * 0.001 # Scale to seconds

    @property
    def target_name(self):
        """
        Target name used in various Spice calls to acquire
        target specific ephemeris data.

        Returns
        -------
        : str
          Target name
        """
        return self.label['IsisCube']['Instrument']['TargetName']

    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            sclock = self.label['IsisCube']['Archive']['SpacecraftClockStartCount']
            self._starting_ephemeris_time = spice.scs2e(self.spacecraft_id, sclock).value
        return self._starting_ephemeris_time


class IsisSpice(Isis3):
    """Mixin class for reading from an ISIS cube that has been spiceinit'd

    Attributes
    ----------
    _label : PVLModule
             Dict-like object with PVL keys

    _inst_pointing_table : dict
                           Dictionary that contains information about the
                           rotation from J2000 to the sensor reference frame.
                           All of the values for each property, such as angular
                           velocity, are stored in a list or numpy array where
                           each entry is the property at a different time.

    _body_orientation_table : dict
                              Dictionary that contains information about the
                              rotation from J2000 to the body fixed reference
                              frame. All of the  values for each property, such
                              as angular velocity, are stored in a list or
                              numpy array where each entry is the property at a
                              different time.

    _inst_position_table : dict
                           Dictionary that contains information about the
                           location of the sensor relative to the center of the
                           target body. All of the  values for each property,
                           such as velocity, are stored in a list or numpy
                           array where each entry is the property at a
                           different time.

    _sun_position_table : dict
                          Dictionary that contains information about the
                          location of the sun relative to the center of the
                          target body. All of the  values for each property,
                          such as velocity, are stored in a list or numpy
                          array where each entry is the property at a
                          different time.

    """

    def _init_tables(self):
        # init tables
        for table in self.label.getlist('Table'):
            binary_data = read_table_data(table, self._file)
            field_data = parse_table_data(table, binary_data)
            if table['Name'] == 'InstrumentPointing':
                self._inst_pointing_table = parse_rotation_table(table, field_data)
            elif table['Name'] == 'BodyRotation':
                self._body_orientation_table = parse_rotation_table(table, field_data)
            elif table['Name'] == 'InstrumentPosition':
                self._inst_position_table = parse_position_table(field_data)
            elif table['Name'] == 'SunPosition':
                self._sun_position_table = parse_position_table(field_data)

    @property
    def label(self):
        """
        Loads a PVL from from the _file attribute and
        parses the binary table data.

        Returns
        -------
        PVLModule :
            Dict-like object with PVL keys
        """
        if not hasattr(self, "_label"):
            try:
                self._label = pvl.load(self.file)
            except:
                raise ValueError("{} is not a valid label".format(self.file))
        return self._label

    def __enter__(self):
        """
        Stub method to conform with how other driver mixins
        are used.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stub method to conform with how other driver mixins
        are used.
        """
        pass

    @property
    def number_of_quaternions(self):
        """
        The number of instrument rotation quaternions

        Returns
        -------
        int :
            The number of quaternions
        """
        return len(self._sensor_orientation)

    @property
    def number_of_ephemerides(self):
        """
        The number of instrument position states. These may
        be just positions or positions and vbelocities.

        Returns
        -------
        int :
            The number of states
        """
        return len(self._sensor_position)

    @property
    def _sclock_hex_string(self):
        """
        The hex encoded image start time computed from the
        spacecraft clock count

        Returns
        -------
        str :
            The hex string representation of the image
            start time as a double
        """
        for key in self.naif_keywords:
            if re.match('CLOCK_ET_.*_COMPUTED', key[0]):
                # If the hex string is only numbers and contains leading 0s,
                # the PVL library strips them off (ie. 0000000000002040 becomes
                # 2040). Pad to 16 in case this happens.
                return str(key[1]).zfill(16)
        raise ValueError("No computed spacecraft clock time found in NaifKeywords.")

    @property
    def starting_ephemeris_time(self):
        """
        The image start time in ephemeris time

        Returns
        -------
        float :
            The image start ephemeris time
        """
        return struct.unpack('d', bytes.fromhex(self._sclock_hex_string))[0]

    @property
    def _detector_center_sample(self):
        """
        The center of the CCD in detector pixels

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return self.naif_keywords.get('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), None)

    @property
    def _detector_center_line(self):
        """
        The center of the CCD in detector pixels

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return self.naif_keywords.get('INS{}_BORESIGHT_LINE'.format(self.ikid), None)

    @property
    def _cube_label(self):
        """
        The ISIS cube label portion of the file label

        Returns
        -------
        PVLModule :
            The ISIS cube label

        """
        if 'IsisCube' not in self.label:
            raise ValueError("Could not find ISIS cube label.")
        return self.label['IsisCube']

    @property
    def _kernels_group(self):
        """
        The Kernels group from the ISIS cube label.
        This is where the original SPICE kernels are listed.

        Returns
        -------
        PVLModule :
            The kernels group
        """
        if 'Kernels' not in self._cube_label:
            raise ValueError("Could not find Kernels group in ISIS cube label.")
        return self._cube_label['Kernels']

    @property
    def ikid(self):
        """
        The NAIF id for the instrument

        Returns
        -------
        int :
            The instrument id
        """
        if 'NaifIkCode' not in self._kernels_group:
            raise ValueError("Could not find Instrument NAIF ID in Kernels group.")
        return self._kernels_group['NaifIkCode']

    @property
    def focal2pixel_lines(self):
        """
        The line component of the affine transformation
        from focal plane coordinates to centered ccd pixels

        Returns
        -------
        list :
            The coefficients of the affine transformation
            formatted as constant, x, y
        """
        return self.naif_keywords.get('INS{}_ITRANSL'.format(self.ikid), None)

    @property
    def focal2pixel_samples(self):
        """
        The sample component of the affine transformation
        from focal plane coordinates to centered ccd pixels

        Returns
        -------
        list :
            The coefficients of the affine transformation
            formatted as constant, x, y
        """
        return self.naif_keywords.get('INS{}_ITRANSS'.format(self.ikid), None)

    @property
    def _focal_length(self):
        """
        The focal length of the instrument

        Returns
        -------
        float :
            The focal length in millimeters
        """
        return self.naif_keywords.get('INS{}_FOCAL_LENGTH'.format(self.ikid), None)

    @property
    def _body_radii(self):
        """
        The triaxial radii of the target body

        Returns
        -------
        list :
            The body radii in kilometers. For most bodies,
            this is formatted as semimajor, semimajor,
            semiminor
        """
        for key in self.naif_keywords:
            if re.match('BODY-?\d*_RADII', key[0]):
                return self.naif_keywords[key[0]]

    @property
    def _semimajor(self):
        """
        The radius of the target body at its widest
        diameter

        Returns
        -------
        float :
            The radius in kilometers
        """
        return self._body_radii[0]

    @property
    def _semiminor(self):
        """
        The radius of the target body perpendicular to its
        widest diameter

        Returns
        -------
        float :
            The radius in kilometers
        """
        return self._body_radii[2]

    @property
    def _body_time_dependent_frames(self):
        """
        List of time dependent reference frames between the
        target body reference frame and the J2000 frame.

        Returns
        -------
        list :
            The list of frames starting with the body
            reference frame and ending with the final time
            dependent frame.
        """
        if not hasattr(self, "_body_orientation_table"):
            self._init_tables()
        if 'TimeDependentFrames' not in self._body_orientation_table:
            raise ValueError("Could not find body time dependent frames.")
        return self._body_orientation_table['TimeDependentFrames']

    @property
    def reference_frame(self):
        """
        The NAIF ID for the target body reference frame

        Returns
        -------
        int :
            The frame ID
        """
        return self._body_time_dependent_frames[0]

    @property
    def _sun_position(self):
        """
        The sun position

        Returns
        -------
        array :
            The sun position vectors relative to the center
            of the target body in the J2000 reference frame
            as a 2d numpy array
        """
        if not hasattr(self, "_sun_position_table"):
            self._init_tables()
        return self._sun_position_table.get('Positions', 'None')

    @property
    def _sun_velocity(self):
        """
        The sun velocity

        Returns
        -------
        array :
            The sun velocity vectors in the J2000 reference
            frame as a 2d numpy array
        """
        if not hasattr(self, "_sun_position_table"):
            self._init_tables()
        return self._sun_position_table.get('Velocities', None)

    @property
    def _sensor_position(self):
        """
        """
        if not hasattr(self, "_inst_position_table"):
            self._init_tables()
        return self._inst_position_table.get('Positions', None)

    @property
    def _sensor_velocity(self):
        """
        The sensor velocity

        Returns
        -------
        array :
            The sensor velocity vectors in the J2000
              reference frame as a 2d numpy array
        """
        if not hasattr(self, "_inst_position_table"):
            self._init_tables()
        return self._inst_position_table.get('Velocities', None)

    @property
    def _sensor_orientation(self):
        """
        The rotation from J2000 to the sensor reference
        frame

        Returns
        -------
        array :
            The sensor rotation quaternions as a numpy
            quaternion array
        """
        if not hasattr(self, "_inst_pointing_table"):
            self._init_tables()
        return self._inst_pointing_table.get('Rotations', None)

    @property
    def body_orientation(self):
        """
        The rotation from J2000 to the target body
        reference frame

        Returns
        -------
        array :
            The body rotation quaternions as a numpy
            quaternion array
        """
        if not hasattr(self, "_body_orientation_table"):
            self._init_tables()
        return self._body_orientation_table.get('Rotations', None)

    @property
    def naif_keywords(self):
        """
        The NaifKeywords group from the file label that
        contains stored values from the original SPICE
        kernels

        Returns
        -------
        PVLModule :
            The stored NAIF keyword values
        """
        if 'NaifKeywords' not in self.label:
            raise ValueError("Could not find NaifKeywords in label.")
        return self.label['NaifKeywords']

    @property
    def _odtk(self):
        return self.label["NaifKeywords"]["INS{}_OD_K".format(self.ikid)]


class RadialDistortion():
    @property
    def optical_distortion(self):
        return {
            "Radial": {
                "coefficients" : self._odtk
            }
        }
