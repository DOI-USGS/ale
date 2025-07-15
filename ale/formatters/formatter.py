from networkx.algorithms.shortest_paths.generic import shortest_path

import json 

from ale.base.type_sensor import LineScanner, Framer, Radar, PushFrame
from ale import logger

def to_isd(driver):
    """
    Formatter to create sensor model meta data from a driver.

    Parameters
    ----------
    driver : Driver
        Concrete driver for the image that meta data is being generated for.

    Returns
    -------
    string
        The ISIS compatible meta data as a JSON encoded string.
    """
     
    driver_data = driver.to_dict()
    logger.debug(f"driver_data:\n{driver_data}")
    isd = {}
    isd['isis_camera_version'] = driver_data["sensor_model_version"]

    # general information
    isd['image_lines'] = driver_data["image_lines"]
    isd['image_samples'] = driver_data["image_samples"]
    isd['name_platform'] = driver_data["platform_name"]
    isd['name_sensor'] = driver_data["sensor_name"]
    isd['reference_height'] = {
        "maxheight": 1000,
        "minheight": -1000,
        "unit": "m"
    }

    # line scan sensor model specifics
    if isinstance(driver, LineScanner):
        isd['name_model'] = 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'
        isd['interpolation_method'] = 'lagrange'
        
        start_lines, start_times, scan_rates = driver_data["line_scan_rate"]
        isd['line_scan_rate'] = [[line, time, rate] for line, time, rate in zip(start_lines, start_times, scan_rates)]
        isd['starting_ephemeris_time'] = driver_data["ephemeris_start_time"]
        isd['center_ephemeris_time'] = driver_data["center_ephemeris_time"]

    # frame sensor model specifics
    if isinstance(driver, Framer):
        isd['name_model'] = 'USGS_ASTRO_FRAME_SENSOR_MODEL'
        isd['center_ephemeris_time'] = driver_data["center_ephemeris_time"]

    if isinstance(driver, PushFrame):
        isd['name_model'] = 'USGS_ASTRO_PUSH_FRAME_SENSOR_MODEL'
        isd['starting_ephemeris_time'] = driver_data["ephemeris_start_time"]
        isd['ending_ephemeris_time'] = driver_data["ephemeris_stop_time"]
        isd['center_ephemeris_time'] = driver_data["center_ephemeris_time"]
        isd['exposure_duration'] = driver_data["exposure_duration"]
        isd['interframe_delay'] = driver_data["interframe_delay"]
        isd['framelet_order_reversed'] = driver_data["framelet_order_reversed"]
        isd['framelets_flipped'] = driver_data["framelets_flipped"]
        isd['framelet_height'] = driver_data["framelet_height"]
        isd['num_lines_overlap'] = driver_data["num_lines_overlap"]

    # SAR sensor model specifics
    if isinstance(driver, Radar):
        isd['name_model'] = 'USGS_ASTRO_SAR_SENSOR_MODEL'
        isd['starting_ephemeris_time'] = driver_data["ephemeris_start_time"]
        isd['ending_ephemeris_time'] = driver_data["ephemeris_stop_time"]
        isd['center_ephemeris_time'] = driver_data["center_ephemeris_time"]
        isd['wavelength'] = driver_data["wavelength"]
        isd['line_exposure_duration'] = driver_data["line_exposure_duration"]
        isd['scaled_pixel_width'] = driver_data["scaled_pixel_width"]
        isd['range_conversion_times'] = driver_data["range_conversion_times"]
        isd['range_conversion_coefficients'] = driver_data["range_conversion_coefficients"]
        isd['look_direction'] = driver_data["look_direction"]

    # Target body
    body_radii = driver.target_body_radii
    isd['radii'] = {
        'semimajor' : body_radii[0],
        'semiminor' : body_radii[2],
        'unit' : 'km'
    }

    frame_chain = driver_data["frame_chain"]
    target_frame = driver_data["target_frame_id"]

    J2000 = 1 # J2000 frame id
    body_rotation = {}
    source_frame, destination_frame, time_dependent_target_frame = frame_chain.last_time_dependent_frame_between(target_frame, J2000)
    
    if source_frame != J2000:
        # Reverse the frame order because ISIS orders frames as
        # (destination, intermediate, ..., intermediate, source)
        body_rotation['time_dependent_frames'] = shortest_path(frame_chain, source_frame, J2000)
        time_dependent_rotation = frame_chain.compute_rotation(J2000, source_frame)
        body_rotation['ck_table_start_time'] = time_dependent_rotation.times[0]
        body_rotation['ck_table_end_time'] = time_dependent_rotation.times[-1]
        body_rotation['ck_table_original_size'] = len(time_dependent_rotation.times)
        body_rotation['ephemeris_times'] = time_dependent_rotation.times
        body_rotation['quaternions'] = time_dependent_rotation.quats[:, [3, 0, 1, 2]]
        body_rotation['angular_velocities'] = time_dependent_rotation.av

    if source_frame != target_frame:
        # Reverse the frame order because ISIS orders frames as
        # (destination, intermediate, ..., intermediate, source)
        body_rotation['constant_frames'] = shortest_path(frame_chain, target_frame, source_frame)
        constant_rotation = frame_chain.compute_rotation(source_frame, target_frame)
        body_rotation['constant_rotation'] = constant_rotation.rotation_matrix().flatten()

    body_rotation["reference_frame"] = destination_frame
    isd['body_rotation'] = body_rotation

    # sensor orientation
    sensor_frame = driver_data["sensor_frame_id"]

    instrument_pointing = {}
    source_frame, destination_frame, _ = frame_chain.last_time_dependent_frame_between(1, sensor_frame)

    # Reverse the frame order because ISIS orders frames as
    # (destination, intermediate, ..., intermediate, source)
    instrument_pointing['time_dependent_frames'] = shortest_path(frame_chain, destination_frame, J2000)
    time_dependent_rotation = frame_chain.compute_rotation(J2000, destination_frame)
    instrument_pointing['ck_table_start_time'] = time_dependent_rotation.times[0]
    instrument_pointing['ck_table_end_time'] = time_dependent_rotation.times[-1]
    instrument_pointing['ck_table_original_size'] = len(time_dependent_rotation.times)
    instrument_pointing['ephemeris_times'] = time_dependent_rotation.times
    instrument_pointing['quaternions'] = time_dependent_rotation.quats[:, [3, 0, 1, 2]]
    instrument_pointing['angular_velocities'] = time_dependent_rotation.av

    # reference frame should be the last frame in the chain
    instrument_pointing["reference_frame"] = instrument_pointing['time_dependent_frames'][-1]

    # Reverse the frame order because ISIS orders frames as
    # (destination, intermediate, ..., intermediate, source)
    instrument_pointing['constant_frames'] = shortest_path(frame_chain, sensor_frame, destination_frame)
    constant_rotation = frame_chain.compute_rotation(destination_frame, sensor_frame)
    instrument_pointing['constant_rotation'] = constant_rotation.rotation_matrix().flatten()
    isd['instrument_pointing'] = instrument_pointing

    # interior orientation
    isd['naif_keywords'] = driver_data["naif_keywords"]

    if isinstance(driver,LineScanner) or isinstance(driver, Framer) or isinstance(driver, PushFrame):

        isd['detector_sample_summing'] = driver_data["sample_summing"]
        isd['detector_line_summing'] = driver_data["line_summing"]

        isd['focal_length_model'] = {
            'focal_length' : driver_data["focal_length"]
        }
        isd['detector_center'] = {
            'line' : driver_data["detector_center_line"],
            'sample' : driver_data["detector_center_sample"]
        }
        isd['focal2pixel_lines'] = driver_data["focal2pixel_lines"]
        isd['focal2pixel_samples'] = driver_data["focal2pixel_samples"]
        isd['optical_distortion'] = driver_data["usgscsm_distortion_model"]

        isd['starting_detector_line'] = driver_data["detector_start_line"]
        isd['starting_detector_sample'] = driver_data["detector_start_sample"]

    j2000_rotation = frame_chain.compute_rotation(target_frame, J2000)

    instrument_position = {}
    positions, velocities, times = driver_data["sensor_position"]
    instrument_position['spk_table_start_time'] = times[0]
    instrument_position['spk_table_end_time'] = times[-1]
    instrument_position['spk_table_original_size'] = len(times)
    instrument_position['ephemeris_times'] = times
    # Rotate positions and velocities into J2000 then scale into kilometers
    rotated_positions = j2000_rotation.apply_at(positions, times)/1000
    instrument_position['positions'] = rotated_positions
    # If velocities are provided, then rotate and add to ISD
    if velocities is not None:
        velocities = j2000_rotation.rotate_velocity_at(positions, velocities, times)/1000
        instrument_position['velocities'] = velocities
    instrument_position["reference_frame"] = j2000_rotation.dest
    
    isd['instrument_position'] = instrument_position
    
    sun_position = {}
    positions, velocities, times = driver_data["sun_position"]
    sun_position['spk_table_start_time'] = times[0]
    sun_position['spk_table_end_time'] = times[-1]
    sun_position['spk_table_original_size'] = len(times)
    sun_position['ephemeris_times'] = times
    # Rotate positions and velocities into J2000 then scale into kilometers
    rotated_positions = j2000_rotation.apply_at(positions, times)/1000
    sun_position['positions'] = rotated_positions
    # If velocities are provided, then rotate and add to ISD
    if velocities is not None:
        velocities = j2000_rotation.rotate_velocity_at(positions, velocities, times)/1000
        sun_position['velocities'] = velocities
    sun_position["reference_frame"] = j2000_rotation.dest

    isd['sun_position'] = sun_position

    if (driver.projection != ""):
        isd["projection"] = driver_data["projection"]
        isd["geotransform"] = driver_data["geotransform"]

    # check that there is a valid sensor model name
    if 'name_model' not in isd:
        raise Exception('No CSM sensor model name found!')

    # remove extra qualities
    # TODO: Rewuires SpiceQL API update to get relative kernels
    # if driver.kernels and isinstance(driver.kernels, dict): 
    #     isd["kernels"] = {k: v for k, v in driver.kernels.items() if not "_quality" in k or driver.spiceql_mission in k }
    # elif driver.kernels and isinstance(driver.kernels, list): 
    #     isd["kernels"] = driver.kernels
    # else: 
    #     isd["kernels"] = {}

    return isd
