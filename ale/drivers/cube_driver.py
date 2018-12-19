from glob import glob
import os
import struct
import re

import pvl
import spiceypy as spice
import numpy as np
import quaternion

from ale import config
from ale.drivers.base import Base

def read_table_data(table_label, file):
    file.seek(label['StartByte']-1)
    return file.read(label['Bytes'])

def field_size(field_label):
    data_sizes = {
        'Integer' : 4,
        'Double'  : 8,
        'Real'    : 4,
        'Text'    : 1
    }
    return data_sizes[field_label['Type']] * field_label['Size']

def field_format(field_label):
    data_formats = {
        'Integer' : 'i',
        'Double'  : 'd',
        'Real'    : 'f'
    }
    return data_formats[field_label['Type']] * field_label['Size']

def parse_field(field_label, data):
    if field_label['Type'] == 'Text':
        results[field_label['Name']].append(data[:field_label['Size']].decode(encoding='latin_1'))
    else:
        data_format = field_format(field_label)
        field_data = struct.unpack_from(data_format, data)
        if len(field_data) == 1:
            field_data = field_data[0]
    return field_data

def parse_table_data(table_label, data):
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
    results = {}
    if all (key in field_data for key in ['J2000Q0','J2000Q1','J2000Q2','J2000Q3']):
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

class Cube(Base):

    def __init__(self, file, *args, **kwargs):
        super(Cube, self).__init__('')
        self.label = pvl.load(file)
        for table in self.label.getlist('Table'):
            binary_data = read_table_data(table, file)
            field_data = parse_table_data(table, binary_data)
            if table['Name'] == 'InstrumentPointing':
                self.inst_pointing_table = parse_rotation_table(table, field_data)
            elif table['Name'] == 'BodyRotation':
                self.body_orientation_table = parse_rotation_table(table, field_data)
            elif table['Name'] == 'InstrumentPosition':
                self.inst_position_table = parse_position_table(field_data)
            elif table['Name'] == 'SunPosition':
                self.sun_position_table = parse_position_table(field_data)

    @property
    def instrument_id(self):
        return self.label['IsisCube']['Instrument']['InstrumentId']

    @property
    def start_time(self):
        return self.label['IsisCube']['Instrument']['StartTime']

    @property
    def image_lines(self):
        return self.label['IsisCube']['Core']['Dimensions']['Lines']

    @property
    def image_samples(self):
        return self.label['IsisCube']['Core']['Dimensions']['Samples']

    @property
    def interpolation_method(self):
        return 'hermite'

    @property
    def number_of_quaternions(self):
        return len(self.sensor_orientation)

    @property
    def number_of_ephemerides(self):
        return len(self.sensor_position)

    @property
    def target_name(self):
        return self.label['IsisCube']['Instrument']['TargetName']

    @property
    def starting_ephemeris_time(self):
        return self.inst_position_table['Times'][0]

    @property
    def ending_ephemeris_time(self):
        return self.inst_position_table['Times'][-1]

    @property
    def detector_center(self):
        return [
            self.label['NaifKeywords']['INS{}_BORESIGHT_LINE'.format(self.ikid)],
            self.label['NaifKeywords']['INS{}_BORESIGHT_SAMPLE'.format(self.ikid)]
        ]

    @property
    def spacecraft_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def ikid(self):
        return self.label['IsisCube']['Kernels']['NaifIkCode']

    @property
    def fikid(self):
        pass

    @property
    def focal2pixel_lines(self):
        return self.label['NaifKeywords']['INS{}_ITRANSL'.format(self.ikid)]

    @property
    def focal2pixel_samples(self):
        return self.label['NaifKeywords']['INS{}_ITRANSS'.format(self.ikid)]

    @property
    def focal_length(self):
        return self.label['NaifKeywords']['INS{}_FOCAL_LENGTH'.format(self.ikid)]

    @property
    def body_radii(self):
        for key in self.label['NaifKeywords']:
            if re.match('BODY-?\d*_RADII', key[0]):
                return self.label['NaifKeywords'][key[0]]

    @property
    def semimajor(self):
        return self.body_radii[0]

    @property
    def semiminor(self):
        return self.body_radii[2]

    @property
    def reference_frame(self):
        return self.body_orientation_table['TimeDependentFrames'][0]

    @property
    def sun_position(self):
        return self.sun_position_table['Positions']

    @property
    def sun_velocity(self):
        return self.sun_position_table['Velocities']

    @property
    def sensor_position(self):
        return self.inst_position_table['Positions']

    @property
    def sensor_velocity(self):
        return self.inst_position_table['Velocities']

    @property
    def sensor_orientation(self):
        return self.inst_pointing_table['Rotations']

    @property
    def body_orientation(self):
        return self.body_orientation_table['Rotations']
