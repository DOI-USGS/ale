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


class IsisSpice():
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

        Returns
        -------
        : dict
          Instrument pointing table
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

        Returns
        -------
        : dict
          Body orientation table
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

        Returns
        -------
        : dict
          Instrument position table
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

        Returns
        -------
        : dict
          Sun position table
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


    @property
    def _sclock_hex_string(self):
        """
        The hex encoded image start time computed from the
        spacecraft clock count
        Expects isis_naif_keywords to be defined. This should be a dict containing
        Naif keyworkds from the label.

        Returns
        -------
        str :
            The hex string representation of the image
            start time as a double
        """
        regex = re.compile('CLOCK_ET_.*_COMPUTED')
        for key in self.isis_naif_keywords:
            if re.match(regex, key[0]):
                # If the hex string is only numbers and contains leading 0s,
                # the PVL library strips them off (ie. 0000000000002040 becomes
                # 2040). Pad to 16 in case this happens.
                return str(key[1]).zfill(16)
        raise ValueError("No computed spacecraft clock time found in NaifKeywords.")

    @property
    def ephemeris_start_time(self):
        """
        The image start time in ephemeris time
        Expects sclock_hex_string to be defined. This should be a string
        containing the hex start time of the image

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
        Expects ikid to be defined. this should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return self.isis_naif_keywords.get('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), None)

    @property
    def detector_center_line(self):
        """
        The center of the CCD in detector pixels
        Expects ikid to be defined. this should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        list :
            The center of the CCD formatted as line, sample
        """
        return self.isis_naif_keywords.get('INS{}_BORESIGHT_LINE'.format(self.ikid), None)


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
        Expects cube_label to be defined. This should be a PVLModule containing
        the ISIS cube label.
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
        Expects kernels_group to be defined. This should be a PVLModule
        containing the kernels group.

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
        Expects isis_naif_keywords to be defined. This should be a dict containing
        Naif keyworkds from the label.
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument.

        Returns
        -------
        list :
            The coefficients of the affine transformation
            formatted as constant, x, y
        """
        return self.isis_naif_keywords.get('INS{}_ITRANSL'.format(self.ikid), None)

    @property
    def focal2pixel_samples(self):
        """
        The sample component of the affine transformation
        from focal plane coordinates to centered ccd pixels
        Expects isis_naif_keywords to be defined. This should be a dict containing
        Naif keyworkds from the label.
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument.

        Returns
        -------
        list :
            The coefficients of the affine transformation
            formatted as constant, x, y
        """
        return self.isis_naif_keywords.get('INS{}_ITRANSS'.format(self.ikid), None)

    @property
    def pixel2focal_x(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane x
        """

        return self.isis_naif_keywords.get('INS{}_TRANSX'.format(self.ikid), None)

    @property
    def pixel2focal_y(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane y
        """

        return self.isis_naif_keywords.get('INS{}_TRANSY'.format(self.ikid), None)

    @property
    def focal_length(self):
        """
        The focal length of the instrument
        Expects isis_naif_keywords to be defined. This should be a dict containing
        Naif keyworkds from the label.
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument.

        Returns
        -------
        float :
            The focal length in millimeters
        """
        return self.isis_naif_keywords.get('INS{}_FOCAL_LENGTH'.format(self.ikid), None)

    @property
    def target_body_radii(self):
        """
        The triaxial radii of the target body
        Expects isis_naif_keywords to be defined. This should be a dict containing
        Naif keyworkds from the label.

        Returns
        -------
        list :
            The body radii in kilometers. For most bodies,
            this is formatted as semimajor, semimajor,
            semiminor
        """
        regex = re.compile(r'BODY-?\d*_RADII')
        for key in self.isis_naif_keywords:
            if re.match(regex, key[0]):
                return self.isis_naif_keywords[key[0]]


    @property
    def sun_position(self):
        """
        The sun position
        Expects sun_position_table to be defined. This should be a
        dictionary that contains information about the location of the sun
        relative to the center of the target body.

        Returns
        -------
        array :
            The sun position vectors relative to the center
            of the target body in the J2000 reference frame
            as a tuple of numpy arrays.
        """
        return (self.sun_position_table.get('Positions', 'None'),
                self.sun_position_table.get('Velocities', 'None'),
                self.sun_position_table.get('Times', 'None'))


    @property
    def sensor_position(self):
        """
        Sensor position
        Expects inst_position_table to be defined. This should be a
        dictionary that contains information about the location of the
        sensor relative to the center of the target body.
        Expects number_of_ephemerides to be defined. This should be an integer
        containing the number of instrument position states.

        Returns
        -------
        : (positions, velocities, times)
          a tuple containing a list of positions, a list of velocities, and a list of times
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
    def isis_naif_keywords(self):
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
        """
        Returns optical distortion coefficients
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument

        Returns
        -------
        : list
          optical distortion coefficients
        """
        return self.isis_naif_keywords["INS{}_OD_K".format(self.ikid)]

