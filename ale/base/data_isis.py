import os
import re
import struct
from glob import glob

import numpy as np
from numpy.polynomial.polynomial import polyval, polyder
from dateutil import parser

import pvl
import spiceypy as spice
from ale.rotation import ConstantRotation, TimeDependentRotation
from ale.transformation import FrameChain

from scipy.interpolate import interp1d, BPoly

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

def parse_table(table_label, data):
    """
    Parse an ISIS table into a dictionary.

    Parameters
    ----------
    table_label : PVLModule
                  The ISIS table label
    data : bytes
           The binary component of the ISIS table

    Returns
    -------
    dict :
           The table as a dictionary with the keywords from the label and the
           binary data
    """
    data_sizes = {'Integer' : 4,
                  'Double'  : 8,
                  'Real'    : 4,
                  'Text'    : 1}
    data_formats = {'Integer' : 'i',
                    'Double'  : 'd',
                    'Real'    : 'f'}

    # Parse the binary data
    fields = table_label.getall('Field')
    results = {field['Name']:[] for field in fields}
    offset = 0
    for record in range(table_label['Records']):
        for field in fields:
            if field['Type'] == 'Text':
                field_data = data[offset:offset+field['Size']].decode(encoding='latin_1')
            else:
                data_format = data_formats[field['Type']] * field['Size']
                field_data = struct.unpack_from(data_format, data[offset:])
                if len(field_data) == 1:
                    field_data = field_data[0]

            results[field['Name']].append(field_data)
            offset += data_sizes[field['Type']] * field['Size']

    # Parse the keywords from the label
    results.update({key : value for key, value in table_label.items() if not isinstance(value, pvl.collections.PVLGroup)})

    return results

def rotate_state(table, rotation):
    """
    Rotate the positions and velocities in an ISIS position Table.

    If the table stores states as a function, then it will re compute them
    based on the original size of the table.

    Parameters
    ----------
    table : dict
            The position table as a dictionary
    rotation : TimeDependentRotation
               The rotation to rotate the positions by

    Returns
    -------
    : 2darray
      Array of rotated positions
    : 2darray
      Array of rotated velocities. Returns None if no velocities are in the table.
    : array
      Array of times for the states
    """
    positions = None
    velocities = None
    ephemeris_times = None
    # Case 1, the table has states at discrete times
    if 'J2000X' in table:
        ephemeris_times = table['ET']
        positions = 1000 * np.array([table['J2000X'],
                                     table['J2000Y'],
                                     table['J2000Z']]).T
        if 'J2000XV' in table:
            ephemeris_times = table['ET']
            velocities = 1000 * np.array([table['J2000XV'],
                                          table['J2000YV'],
                                          table['J2000ZV']]).T

    # Case 2, the table has coefficients of polynomials for the states
    elif 'J2000SVX' in table:
        ephemeris_times = np.linspace(table['SpkTableStartTime'],
                                      table['SpkTableEndTime'],
                                      table['SpkTableOriginalSize'])
        base_time = table['J2000SVX'][-1]
        time_scale = table['J2000SVY'][-1]
        scaled_times = (ephemeris_times - base_time) / time_scale
        coeffs = np.array([table['J2000SVX'][:-1],
                           table['J2000SVY'][:-1],
                           table['J2000SVZ'][:-1]])
        positions = 1000 * polyval(scaled_times, coeffs.T).T
        scaled_vel = 1000 * polyval(scaled_times,  polyder(coeffs,axis=1).T).T
        # We took a derivative in scaled time, so we have to multiply by our
        # scale in order to get the derivative in real time
        velocities = scaled_vel / time_scale
    else:
        raise ValueError('No positions are available in the input table.')

    rotated_pos = rotation.apply_at(positions, ephemeris_times)
    if velocities is not None:
        rotated_vel = rotation.rotate_velocity_at(positions, velocities, ephemeris_times)
    else:
        rotated_vel = None
    return rotated_pos, rotated_vel, ephemeris_times

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
            tables = []
            if "Table" in self.label:
                tables = self.label.getall('Table')

            for table in tables:
                if table['Name'] == 'InstrumentPointing':
                    binary_data = read_table_data(table, self._file)
                    self._inst_pointing_table = parse_table(table, binary_data)
                    return self._inst_pointing_table
            raise ValueError(f'Could not find InstrumentPointing table on file {self._file}')
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
            tables = []
            if "Table" in self.label:
                tables = self.label.getall('Table')
                
            for table in tables:
                if table['Name'] == 'BodyRotation':
                    binary_data = read_table_data(table, self._file)
                    self._body_orientation_table = parse_table(table, binary_data)
                    return self._body_orientation_table
            raise ValueError(f'Could not find BodyRotation table on file {self._file}')
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
            tables = []
            if "Table" in self.label:
                tables = self.label.getall('Table')

            for table in tables:
                if table['Name'] == 'InstrumentPosition':
                    binary_data = read_table_data(table, self._file)
                    self._inst_position_table = parse_table(table, binary_data)
                    return self._inst_position_table
            raise ValueError(f'Could not find InstrumentPosition table on file {self._file}')
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
            tables = []
            if "Table" in self.label:
                tables = self.label.getall('Table')

            for table in tables:
                if table['Name'] == 'SunPosition':
                    binary_data = read_table_data(table, self._file)
                    self._sun_position_table = parse_table(table, binary_data)
                    return self._sun_position_table
            raise ValueError(f'Could not find SunPosition table on file {self._file}')
        return self._sun_position_table

    def __enter__(self):
        """
        Stub method to conform with how other driver mixins
        are used.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context goes out of scope. This is
        implemented to conform to the context manager paradigm
        used by other data mix ins.
        """
        pass


    @property
    def _sclock_hex_string(self):
        """
        The hex encoded image start time computed from the
        spacecraft clock count
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keywords from the label.

        Returns
        -------
        str :
            The hex string representation of the image
            start time as a double
        """
        regex = re.compile('CLOCK_ET_.*_COMPUTED')
        for key in self.naif_keywords:
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
        return self.naif_keywords.get('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), None)

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
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keywords from the label.
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument.

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
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keywords from the label.
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument.

        Returns
        -------
        list :
            The coefficients of the affine transformation
            formatted as constant, x, y
        """
        return self.naif_keywords.get('INS{}_ITRANSS'.format(self.ikid), None)

    @property
    def pixel2focal_x(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane x
        """

        return self.naif_keywords.get('INS{}_TRANSX'.format(self.ikid), None)

    @property
    def pixel2focal_y(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
        detector to focal plane y
        """

        return self.naif_keywords.get('INS{}_TRANSY'.format(self.ikid), None)

    @property
    def focal_length(self):
        """
        The focal length of the instrument
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keywords from the label.
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument.

        Returns
        -------
        float :
            The focal length in millimeters
        """
        return self.naif_keywords.get('INS{}_FOCAL_LENGTH'.format(self.ikid), None)

    @property
    def target_body_radii(self):
        """
        The triaxial radii of the target body
        Expects naif_keywords to be defined. This should be a dict containing
        Naif keywords from the label.

        Returns
        -------
        list :
            The body radii in kilometers. For most bodies,
            this is formatted as semimajor, semimajor,
            semiminor
        """
        regex = re.compile(r'BODY-?\d*_RADII')
        for key in self.naif_keywords:
            if re.match(regex, key[0]):
                return self.naif_keywords[key[0]]

    @property
    def frame_chain(self):
        """
        Return the root node of the rotation frame tree/chain.

        The root node is the J2000 reference frame. The other nodes in the
        tree can be accessed via the methods in the FrameNode class.

        Returns
        -------
        FrameNode
            The root node of the frame tree. This will always be the J2000 reference frame.
        """
        if not hasattr(self, '_frame_chain'):
            self._frame_chain = FrameChain.from_isis_tables(
                    inst_pointing = self.inst_pointing_table,
                    body_orientation = self.body_orientation_table)
        return self._frame_chain


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
        j2000_to_target = self.frame_chain.compute_rotation(1, self.target_frame_id)
        positions, velocities, times = rotate_state(self.sun_position_table, j2000_to_target)
        return positions, velocities, times


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
        j2000_to_target = self.frame_chain.compute_rotation(1, self.target_frame_id)
        positions, velocities, times = rotate_state(self.inst_position_table, j2000_to_target)
        return positions, velocities, times


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
        """
        Returns optical distortion coefficients
        Expects ikid to be defined. This should be the integer Naif ID code
        for the instrument

        Returns
        -------
        : list
          optical distortion coefficients
        """
        return self.naif_keywords["INS{}_OD_K".format(self.ikid)]

    @property
    def sensor_frame_id(self):
        if 'ConstantFrames' in self.inst_pointing_table:
            return self.inst_pointing_table['ConstantFrames'][0]
        else:
            return self.inst_pointing_table['TimeDependentFrames'][0]


    @property
    def target_frame_id(self):
        if 'ConstantFrames' in self.body_orientation_table:
            return self.body_orientation_table['ConstantFrames'][0]
        else:
            return self.body_orientation_table['TimeDependentFrames'][0]
