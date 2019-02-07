from glob import glob
import os
import struct
import re

import pvl
import numpy as np
import quaternion

from ale import config
from ale.drivers.base import Driver, Isis3

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
    cube.seek(label['StartByte']-1) # This -1 is straight out of ISIS
    return cube.read(label['Bytes'])

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
        results['Rotations'] = quaternion.as_quat_array( [ [q0, q1, q2, q3] for q0, q1, q2, q3 in zip(field_data['J2000Q0'],field_data['J2000Q1'],field_data['J2000Q2'],field_data['J2000Q3']) ] )
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
        results['ConstantRotation'] = quaternion.from_rotation_matrix(np.reshape(const_rotation_mat, (3, 3)))
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

class IsisSpice(Isis3):

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
                self._label = pvl.load(self._file)
            except:
                raise ValueError("{} is not a valid label".format(self._file))
            for table in self._label.getlist('Table'):
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
        return len(self.sensor_orientation)

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
        return len(self.sensor_position)

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
    def detector_center(self):
        """
        The center of the CCD in detector pixels

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return [
            self.naif_keywords.get('INS{}_BORESIGHT_LINE'.format(self.ikid), None),
            self.naif_keywords.get('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), None)
        ]

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
    def body_radii(self):
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
    def semimajor(self):
        """
        The radius of the target body at its widest
        diameter

        Returns
        -------
        float :
            The radius in kilometers
        """
        return self.body_radii[0]

    @property
    def semiminor(self):
        """
        The radius of the target body perpendicular to its
        widest diameter

        Returns
        -------
        float :
            The radius in kilometers
        """
        return self.body_radii[2]

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
            self.label
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
    def sun_position(self):
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
            self.label
        return self._sun_position_table.get('Positions', 'None')

    @property
    def sun_velocity(self):
        """
        The sun velocity

        Returns
        -------
        array :
            The sun velocity vectors in the J2000 reference
            frame as a 2d numpy array
        """
        if not hasattr(self, "_sun_position_table"):
            self.label
        return self._sun_position_table.get('Velocities', None)

    @property
    def sensor_position(self):
        """
        The sensor position

        Returns
        -------
        array :
            The sensor position vectors relative to the
            center of the target body in the J2000
            reference frame as a 2d numpy array
        """
        if not hasattr(self, "_inst_position_table"):
            self.label
        return self._inst_position_table.get('Positions', None)

    @property
    def sensor_velocity(self):
        """
        The sensor velocity

        Returns
        -------
        array :
            The sensor velocity vectors in the J2000
              reference frame as a 2d numpy array
        """
        if not hasattr(self, "_inst_position_table"):
            self.label
        return self._inst_position_table.get('Velocities', None)

    @property
    def sensor_orientation(self):
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
            self.label
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
            self.label
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
