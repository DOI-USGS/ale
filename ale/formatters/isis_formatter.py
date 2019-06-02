import json

from ale.transformation import FrameNode
from ale.rotation import ConstantRotation, TimeDependentRotation
from ale.encoders import NumpyEncoder

def to_isis(driver):
    """
    Formatter to create ISIS sensor model meta data from a driver.

    Parameters
    ----------
    driver : Driver
        Concrete driver for the image that meta data is being generated for.

    Returns
    -------
    string
        The ISIS compatible meta data as a JSON encoded string.
    """
    meta_data = {}

    meta_data['CameraVersion'] = driver.sensor_model_version

    meta_data['NaifKeywords'] = driver.isis_naif_keywords

    j2000 = driver.frame_chain

    instrument_pointing = {}
    sensor_frame = j2000.find_child_frame(driver.sensor_frame_id)
    time_dependent_sensor_frame = j2000.last_time_dependent_frame_between(sensor_frame)
    if time_dependent_sensor_frame != j2000:
        forward_path, reverse_path = j2000.path_to(time_dependent_sensor_frame)
        # Reverse the frame order because ISIS orders frames as
        # (destination, intermediate, ..., intermediate, source)
        instrument_pointing['TimeDependentFrames'] = [frame.id for frame in (forward_path + reverse_path)[::-1]]
        time_dependent_rotation = j2000.rotation_to(time_dependent_sensor_frame)
        instrument_pointing['CkTableStartTime'] = time_dependent_rotation.times[0]
        instrument_pointing['CkTableEndTime'] = time_dependent_rotation.times[-1]
        instrument_pointing['CkTableOriginalSize'] = len(time_dependent_rotation.times)
        instrument_pointing['EphemerisTimes'] = time_dependent_rotation.times
        instrument_pointing['Quaternions'] = time_dependent_rotation.quats
    if time_dependent_sensor_frame != sensor_frame:
        forward_path, reverse_path = time_dependent_sensor_frame.path_to(sensor_frame)
        # Reverse the frame order because ISIS orders frames as
        # (destination, intermediate, ..., intermediate, source)
        instrument_pointing['ConstantFrames'] = [frame.id for frame in (forward_path + reverse_path)[::-1]]
        constant_rotation = time_dependent_sensor_frame.rotation_to(sensor_frame)
        instrument_pointing['ConstantRotation'] = constant_rotation.rotation_matrix()
    meta_data['InstrumentPointing'] = instrument_pointing

    body_rotation = {}
    target_frame = j2000.find_child_frame(driver.target_frame_id)
    time_dependent_target_frame = j2000.last_time_dependent_frame_between(target_frame)
    if time_dependent_target_frame != j2000:
        forward_path, reverse_path = j2000.path_to(time_dependent_target_frame)
        # Reverse the frame order because ISIS orders frames as
        # (destination, intermediate, ..., intermediate, source)
        body_rotation['TimeDependentFrames'] = [frame.id for frame in (forward_path + reverse_path)[::-1]]
        time_dependent_rotation = j2000.rotation_to(time_dependent_target_frame)
        body_rotation['CkTableStartTime'] = time_dependent_rotation.times[0]
        body_rotation['CkTableEndTime'] = time_dependent_rotation.times[-1]
        body_rotation['CkTableOriginalSize'] = len(time_dependent_rotation.times)
        body_rotation['EphemerisTimes'] = time_dependent_rotation.times
        body_rotation['Quaternions'] = time_dependent_rotation.quats
    if time_dependent_target_frame != target_frame:
        forward_path, reverse_path = time_dependent_target_frame.path_to(target_frame)
        # Reverse the frame order because ISIS orders frames as
        # (destination, intermediate, ..., intermediate, source)
        body_rotation['ConstantFrames'] = [frame.id for frame in (forward_path + reverse_path)[::-1]]
        constant_rotation = time_dependent_target_frame.rotation_to(target_frame)
        body_rotation['ConstantRotation'] = constant_rotation.rotation_matrix()
    meta_data['BodyRotation'] = body_rotation

    instrument_position = {}
    positions, velocities, times = driver.sensor_position
    instrument_position['SpkTableStartTime'] = times[0]
    instrument_position['SpkTableEndTime'] = times[-1]
    instrument_position['SpkTableOriginalSize'] = len(times)
    instrument_position['EphemerisTimes'] = times
    instrument_position['Positions'] = positions
    instrument_position['Velocities'] = velocities
    meta_data['InstrumentPosition'] = instrument_position

    sun_position = {}
    positions, velocities, times = driver.sun_position
    sun_position['SpkTableStartTime'] = times[0]
    sun_position['SpkTableEndTime'] = times[-1]
    sun_position['SpkTableOriginalSize'] = len(times)
    sun_position['EphemerisTimes'] = times
    sun_position['Positions'] = positions
    sun_position['Velocities'] = velocities
    meta_data['SunPosition'] = sun_position

    return json.dumps(meta_data, cls=NumpyEncoder)
