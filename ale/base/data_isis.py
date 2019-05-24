import os
import re
import struct
from glob import glob

import numpy as np
from dateutil import parser

import pvl
import spiceypy as spice
from ale import config

from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation

from ale.base.label_isis import IsisLabel

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
    cubehandle.seek(table_label['StartByte'] - 1)
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
    label : PVLModule
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
    if 'CkTableOriginalSize' in label:
        results['CkTableOriginalSize'] = label['CkTableOriginalSize']
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

def parse_position_table(label, field_data):
    """
    Parses ISIS position table data.

    Parameters
    ----------
    label : PVLModule
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

    if 'SpkTableOriginalSize' in label:
        results['SpkTableOriginalSize'] = label['SpkTableOriginalSize']
    return results



class IsisSpice(IsisLabel):
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

    @property
    def inst_pointing_table(self):
        """
        ISIS Table containing the rotation between the J2000 reference frame
        and the instrument reference frame.
        """
        if not hasattr(self, "_inst_pointing_table"):
            for table in self.label.getlist('Table'):
                if table['Name'] == 'InstrumentPointing':
                    binary_data = read_table_data(table, self._file)
                    field_data = parse_table_data(table, binary_data)
                    self._inst_pointing_table = parse_rotation_table(table, field_data)
        return self._inst_pointing_table

    @property
    def body_orientation_table(self):
        """
        ISIS Table containing the rotation between the J2000 reference frame
        and the target body reference frame.
        """
        if not hasattr(self, "_body_orientation_table"):
            for table in self.label.getlist('Table'):
                if table['Name'] == 'BodyRotation':
                    binary_data = read_table_data(table, self._file)
                    field_data = parse_table_data(table, binary_data)
                    self._body_orientation_table = parse_rotation_table(table, field_data)
        return self._body_orientation_table

    @property
    def inst_position_table(self):
        """
        ISIS Table containing the location of the instrument relative to the
        target body in the J2000 reference frame.
        """
        if not hasattr(self, "_inst_position_table"):
            for table in self.label.getlist('Table'):
                if table['Name'] == 'InstrumentPosition':
                    binary_data = read_table_data(table, self._file)
                    field_data = parse_table_data(table, binary_data)
                    self._inst_position_table = parse_position_table(table, field_data)
        return self._inst_position_table

    @property
    def sun_position_table(self):
        """
        ISIS Table containing the location of the sun relative to the
        target body in the J2000 reference frame.
        """
        if not hasattr(self, "_sun_position_table"):
            for table in self.label.getlist('Table'):
                if table['Name'] == 'SunPosition':
                    binary_data = read_table_data(table, self._file)
                    field_data = parse_table_data(table, binary_data)
                    self._sun_position_table = parse_position_table(table, field_data)
        return self._sun_position_table

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
        return int(self.inst_pointing_table['CkTableOriginalSize'])

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
        return int(self.inst_position_table['SpkTableOriginalSize'])

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
    def ephemeris_start_time(self):
        """
        The image start time in ephemeris time

        Returns
        -------
        float :
            The image start ephemeris time
        """
        return struct.unpack('d', bytes.fromhex(self._sclock_hex_string))[0]

    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return self.naif_keywords.get('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), None)

    @property
    def detector_center_line(self):
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
            if 'NaifFrameCode' not in self._kernels_group:
                raise ValueError("Could not find Instrument NAIF ID in Kernels group.")
            return self._kernels_group['NaifFrameCode']
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
    def focal_length(self):
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
            if re.match(r'BODY-?\d*_RADII', key[0]):
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
        if 'TimeDependentFrames' not in self.body_orientation_table:
            raise ValueError("Could not find body time dependent frames.")
        return self.body_orientation_table['TimeDependentFrames']

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
        return self.sun_position_table.get('Positions', 'None')

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
        return self.sun_position_table.get('Velocities', None)

    @property
    def _sensor_position(self):
        """
        """
        inst_positions_times = np.linspace(self.inst_position_table["Times"][0],
                                           self.inst_position_table["Times"][-1],
                                           self.number_of_ephemerides)

        # interpolate out positions
        if len(self.inst_position_table["Times"]) < 2:
            time_0 = self.inst_position_table["Times"][0]
            position_0 = self.inst_position_table["Positions"][0]
            velocity_0 = self.inst_position_table["Velocities"][0]
            coefs = np.asarray([position_0 - time_0*velocity_0,
                                velocity_0])
            positions = np.polynomial.polynomial.polyval(inst_positions_times, coefs)

        else:
            f_positions_x = interp1d(self.inst_position_table["Times"], self.inst_position_table["Positions"].T[0])
            f_positions_y = interp1d(self.inst_position_table["Times"], self.inst_position_table["Positions"].T[1])
            f_positions_z = interp1d(self.inst_position_table["Times"], self.inst_position_table["Positions"].T[2])

            positions = np.asarray([f_positions_x(inst_positions_times),
                                   f_positions_y(inst_positions_times),
                                   f_positions_z(inst_positions_times)])

        # convert positions to Body-Fixed and scale to meters
        return self._body_j2k2bf_rotation.apply(positions.T)*1000


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
        inst_velocities_times = np.linspace(self.inst_position_table["Times"][0],
                                            self.inst_position_table["Times"][-1],
                                            self.number_of_ephemerides)

        if len(self.inst_position_table["Times"]) < 2:
            velocity_0 = self.inst_position_table["Velocities"][0]
            coefs = np.asarray([velocity_0,
                                [0, 0, 0]])
            velocties = np.polynomial.polynomial.polyval(inst_velocities_times, coefs)

        else:
            f_velocities_x = interp1d(self.inst_position_table["Times"], self.inst_position_table["Velocities"].T[0])
            f_velocities_y = interp1d(self.inst_position_table["Times"], self.inst_position_table["Velocities"].T[1])
            f_velocities_z = interp1d(self.inst_position_table["Times"], self.inst_position_table["Velocities"].T[2])

            velocties = np.asarray([f_velocities_x(inst_velocities_times),
                                   f_velocities_y(inst_velocities_times),
                                   f_velocities_z(inst_velocities_times)])

        # convert positions to Body-Fixed and scale to meters
        return self._body_j2k2bf_rotation.apply(velocties.T)*1000

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
        inst_pointing_times = np.linspace(self.inst_pointing_table["Times"][0],
                                          self.inst_pointing_table["Times"][-1],
                                          self.number_of_quaternions)
        rotations = self.inst_pointing_table["Rotations"]
        rotations = np.roll(np.asarray(rotations), -1, 1) # adjust rotations [0,1,2,3] -> [3,0,1,2]

        if len(self.inst_pointing_table["Times"]) < 2:
            orientations = Rotation.from_quat(rotations[0])
        else:
            orientations = Slerp(self.inst_pointing_table["Times"], Rotation.from_quat(rotations))(inst_pointing_times)

        bf2inst_rotation = (orientations*self._body_j2k2bf_rotation.inv()).as_quat()
        return bf2inst_rotation

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
        return self.body_orientation_table.get('Rotations', None)

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
    def odtk(self):
        return self.label["NaifKeywords"]["INS{}_OD_K".format(self.ikid)]

    @property
    def _body_j2k2bf_rotation(self):
        """
        Returns Mc*Mt where:
        Mt is the time dependant portion of the rotation from j2000 to body fixed
        Mc is contant portion of the rotation from J2000 to body fixed.

        This represents the rotation to get positions from J2000 to body fixed,
        """
        body_rot_times = self.body_orientation_table["Times"]
        body_timed_rots = self.body_orientation_table["Rotations"]
        body_timed_rots = np.roll(np.asarray(body_timed_rots), -1, 1) # adjust quaternions [0,1,2,3] -> [3,0,1,2]

        interp_rot_times = np.linspace(body_rot_times[0],
                                       body_rot_times[-1],
                                       self.number_of_ephemerides)

        if len(self.body_orientation_table["Times"]) < 2:
            rotation_mat = Rotation.from_quat(body_timed_rots[0])
        else:
            rotation_mat = Slerp(body_rot_times, Rotation.from_quat(body_timed_rots))(interp_rot_times)

        # Not all body rotations have a constant component
        if "ConstantRotation" in self.body_orientation_table:
            body_const_rots = self.body_orientation_table["ConstantRotation"]
            rotation_mat = Rotation.from_dcm(body_const_rots)*rotation_mat

        return rotation_mat
