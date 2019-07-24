import json

from ale.transformation import FrameChain

from ale.base.type_sensor import LineScanner, Framer
from ale.encoders import NumpyEncoder
from ale.rotation import ConstantRotation, TimeDependentRotation

def to_usgscsm(driver):
    """
    Formatter to create USGSCSM meta data from a driver.

    Parameters
    ----------
    driver : Driver
        Concrete driver for the image that meta data is being generated for.

    Returns
    -------
    string
        The USGSCSM compatible meta data as a JSON encoded string.
    """
    isd_data = {}

    # exterior orientation
    body_radii = driver.target_body_radii
    isd_data['radii'] = {
        'semimajor' : body_radii[0],
        'semiminor' : body_radii[2],
        'unit' : 'km'
    }
    positions, velocities, position_times = driver.sensor_position
    isd_data['sensor_position'] = {
        'positions' : positions,
        'velocities' : velocities,
        'unit' : 'm'
    }
    sun_positions, sun_velocities, _ = driver.sun_position
    isd_data['sun_position'] = {
        'positions' : sun_positions,
        'velocities' : sun_velocities,
        'unit' : 'm'
    }

    frame_chain = driver.frame_chain
    sensor_to_target = frame_chain.compute_rotation(driver.sensor_frame_id, driver.target_frame_id)
    quaternions = sensor_to_target.quats
    rotation_times = sensor_to_target.times
    isd_data['sensor_orientation'] = {
        'quaternions' : quaternions
    }

    # interior orientation
    isd_data['detector_sample_summing'] = driver.sample_summing
    isd_data['detector_line_summing'] = driver.line_summing
    isd_data['focal_length_model'] = {
        'focal_length' : driver.focal_length
    }
    isd_data['detector_center'] = {
        'line' : driver.detector_center_line,
        'sample' : driver.detector_center_sample
    }
    isd_data['starting_detector_line'] = driver.detector_start_line
    isd_data['starting_detector_sample'] = driver.detector_start_sample
    isd_data['focal2pixel_lines'] = driver.focal2pixel_lines
    isd_data['focal2pixel_samples'] = driver.focal2pixel_samples
    isd_data['optical_distortion'] = driver.usgscsm_distortion_model

    # general information
    isd_data['image_lines'] = driver.image_lines
    isd_data['image_samples'] = driver.image_samples
    isd_data['name_platform'] = driver.platform_name
    isd_data['name_sensor'] = driver.sensor_name
    isd_data['reference_height'] = {
        "maxheight": 1000,
        "minheight": -1000,
        "unit": "m"
    }

    # line scan sensor model specifics
    if isinstance(driver, LineScanner):
        isd_data['name_model'] = 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'
        isd_data['interpolation_method'] = 'lagrange'
        start_lines, start_times, scan_rates = driver.line_scan_rate
        center_time = driver.center_ephemeris_time
        isd_data['line_scan_rate'] = [[line, time, rate] for line, time, rate in zip(start_lines, start_times, scan_rates)]
        isd_data['starting_ephemeris_time'] = driver.ephemeris_start_time
        isd_data['center_ephemeris_time'] = center_time
        isd_data['t0_ephemeris'] = position_times[0] - center_time
        if len(position_times) > 1:
            isd_data['dt_ephemeris'] = (position_times[-1] - position_times[0]) / (len(position_times) - 1)
        else:
            isd_data['dt_ephemeris'] = 0
        isd_data['t0_quaternion'] = rotation_times[0] - center_time
        if len(rotation_times) > 1:
            isd_data['dt_quaternion'] = (rotation_times[-1] - rotation_times[0]) / (len(rotation_times) - 1)
        else:
            isd_data['dt_quaternion'] = 0


    # frame sensor model specifics
    if isinstance(driver, Framer):
        isd_data['name_model'] = 'USGS_ASTRO_FRAME_SENSOR_MODEL'
        isd_data['center_ephemeris_time'] = driver.center_ephemeris_time

    # check that there is a valid sensor model name
    if 'name_model' not in isd_data:
        raise Exception('No CSM sensor model name found!')

    # Convert to JSON object
    return json.dumps(isd_data, cls=NumpyEncoder)
