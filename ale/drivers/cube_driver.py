from glob import glob
import os
import struct

import pvl
import spiceypy as spice
import numpy as np
import quaternion

from ale import config
from ale.drivers.base import Base

def read_table(label, file):
    data_types = {
        'Integer' : {'format':'i', 'size':4},
        'Double'  : {'format':'d', 'size':8},
        'Real'    : {'format':'f', 'size':4},
        'Text'    : {'format':'c', 'size':1}
    }

    file.seek(label['StartByte']-1)
    data = file.read(label['Bytes'])

    fields = label.getlist('Field')
    results = {field['Name']:[] for field in fields}
    offset = 0
    for record in range(label['Records']):
        for field in fields:
            count = field['Size']
            if field['Type'] == 'Text':
                results[field['Name']].append(data[offset:offset+count].decode(encoding='latin_1'))
            else:
                data_format = data_types[field['Type']]['format'] * count
                field_data = struct.unpack_from(data_format, data, offset)
                if len(field_data) == 1:
                    results[field['Name']].append(field_data[0])
                else:
                    results[field['Name']].append(field_data)
            offset += data_types[field['Type']]['size'] * count

    return results

def read_rotation_table(label, file):
    bin_data = read_table(label, file)
    results = {}
    if all (key in bin_data for key in ('J2000Q0','J2000Q1','J2000Q2','J2000Q3')):
        results['Rotations'] = quaternion.as_quat_array( [ [q0, q1, q2, q2] for q0, q1, q2, q2 in zip(bin_data['J2000Q0'],bin_data['J2000Q1'],bin_data['J2000Q2'],bin_data['J2000Q3']) ] )
    if all (key in bin_data for key in ('AV1','AV2','AV3')):
        results['AngularVelocities'] = np.array( [ [av1, av2, av3] for av1, av2, av3 in zip(bin_data['AV1'],bin_data['AV2'],bin_data['AV3']) ] )
    if 'ET' in bin_data:
        results['Times'] = np.array(bin_data['ET'])
    if all (key in bin_data for key in ('J2000Ang1','J2000Ang2','J2000Ang3')):
        results['EulerCoefficients'] = np.array([bin_data['J2000Ang1'],bin_data['J2000Ang2'],bin_data['J2000Ang3']])
        results['BaseTime'] = bin_data['J2000Ang1'][-1]
        results['TimeScale'] = bin_data['J2000Ang2'][-1]
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

def read_position_table(label, file):
    bin_data = read_table(label, file)
    results = {}
    if all (key in bin_data for key in ('J2000X','J2000Y','J2000Z')):
        results['Positions'] = np.array( [ [x, y, z] for x, y, z in zip(bin_data['J2000X'],bin_data['J2000Y'],bin_data['J2000Z']) ] )
    if 'ET' in bin_data:
        results['Times'] = np.array(bin_data['ET'])
    if all (key in bin_data for key in ('J2000XV','J2000YV','J2000ZV')):
        results['Velocities'] = np.array( [ [x, y, z] for x, y, z in zip(bin_data['J2000XV'],bin_data['J2000YV'],bin_data['J2000ZV']) ] )
    if all (key in bin_data for key in ('J2000SVX','J2000SVY','J2000SVZ')):
        results['PositionCoefficients'] = np.array( [bin_data['J2000SVX'][:-1],bin_data['J2000SVY'][:-1],bin_data['J2000SVZ'][:-1]] )
        results['BaseTime'] = bin_data['J2000SVX'][-1]
        results['TimeScale'] = bin_data['J2000SVY'][-1]
    return results

class Cube(Base):

    def __init__(self, file, *args, **kwargs):
        super(Cube, self).__init__('')
        self.label = pvl.load(file)
        for table in self.label.getlist('Table'):
            if table['Name'] == 'InstrumentPointing':
                self.inst_pointing_table = read_rotation_table(table, file)
            elif table['Name'] == 'BodyRotation':
                self.body_orientation_table = read_rotation_table(table, file)
            elif table['Name'] == 'InstrumentPosition':
                self.inst_position_table = read_position_table(table, file)
            elif table['Name'] == 'SunPosition':
                self.sun_position_table = read_position_table(table, file)

    def _parse_naif_keywords(self):
        """
        Helper function to parse the NaifKeywords object on the cube label.
        This object contains all values queried from NAIF kernels via g*pool calls.
        """
        if 'NaifKeywords' in self.label:
            self._naif_keywords = self.label['NaifKeywords']
            for key in self.label['NaifKeywords'].keys():
                if key.endswith('_RADII'):
                    self._radii = self.label['NaifKeywords'][key]
                elif key.endswith('_FOCAL_LENGTH'):
                    self._focal_length = self.label['NaifKeywords'][key]
                elif key.endswith('_BORESIGHT_SAMPLE'):
                    self._boresight_sample = self.label['NaifKeywords'][key]
                elif key.endswith('_BORESIGHT_LINE'):
                    self._boresight_line = self.label['NaifKeywords'][key]
                elif key.endswith('_ITRANSS'):
                    self._focal2pixels_sample = self.label['NaifKeywords'][key]
                elif key.endswith('_ITRANSL'):
                    self._focal2pixels_line = self.label['NaifKeywords'][key]

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
        if self._sensor_velocity:
            return 'hermite'
        else:
            return 'linear'

    @property
    def number_of_ephemerides(self):
        if 'Positions' in self.inst_position_table:
            return epehem_count = self.inst_position_table['Positions']
        else:
            pass

    @property
    def target_name(self):
        return self.label['IsisCube']['Instrument']['TargetName']

    @property
    def starting_ephemeris_time(self):
        if 'Positions' in self.inst_position_table:
            return epehem_count = self.inst_position_table['Times'][0]
        else:
            pass

    @property
    def ending_ephemeris_time(self):
        if 'Positions' in self.inst_position_table:
            return epehem_count = self.inst_position_table['Times'][-1]
        else:
            pass

    @property
    def detector_center(self):
        if not hasattr(self, '_naif_keywords'):
            self._parse_naif_keywords()
        if all([hasattr(self, key) for key in ['_boresight_sample','_boresight_line']]):
            return [self._boresight_line, self._boresight_sample]
        return [self.image_lines/2, self.image_samples/2]

    @property
    def spacecraft_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def ikid(self):
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def fikid(self):
        pass

    @property
    def spacecraft_id(self):
        return self.label['IsisCube']['Instrument']['SpacecraftId']

    @property
    def focal2pixel_lines(self):
        if not hasattr(self, '_naif_keywords'):
            self._parse_naif_keywords()
        if hasattr(self, '_focal2pixels_line'):
            return self._focal2pixels_line
        return [0, 0, 0] #TODO what should be returned here?

    @property
    def focal2pixel_samples(self):
        if not hasattr(self, '_naif_keywords'):
            self._parse_naif_keywords()
        if hasattr(self, '_focal2pixels_sample'):
            return self._focal2pixels_sample
        return [0, 0, 0] #TODO what should be returned here?

    @property
    def focal_length(self):
        if not hasattr(self, '_naif_keywords'):
            self._parse_naif_keywords()
        if hasattr(self, '_focal_length'):
            return self._focal_length
        return 0 #TODO what should be returned here?

    @property
    def detector_line_summing(self):
        # This is going to take mission by mission code
        return 1

    @property
    def semimajor(self):
        if not hasattr(self, '_naif_keywords'):
            self._parse_naif_keywords()
        if hasattr(self, '_radii'):
            return self._radii[0]
        return 0 #TODO what should be returned here?

    @property
    def semiminor(self):
        if not hasattr(self, '_naif_keywords'):
            self._parse_naif_keywords()
        if hasattr(self, '_radii'):
            return self._radii[2]
        return 0 #TODO what should be returned here?

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
