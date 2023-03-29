import json
import numpy as np
from scipy.interpolate import interp1d, BPoly

from ale.transformation import FrameChain

from ale.base.type_sensor import LineScanner, Framer, Radar
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


    # general information
    isd_data['image_lines'] = driver.image_lines
    isd_data['image_samples'] = driver.image_samples
    isd_data['name_platform'] = driver.platform_name
    isd_data['name_sensor'] = driver.sensor_name

    # shared exterior orientation
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

    isd_data["projection"] = driver.projection
    isd_data["geotransform"] = driver.geotransform

    # shared isd keywords for Framer and Linescanner
    if isinstance(driver, LineScanner) or isinstance(driver, Framer):
        # exterior orientation for just Framer and LineScanner
        frame_chain = driver.frame_chain
        sensor_to_target = frame_chain.compute_rotation(driver.sensor_frame_id, driver.target_frame_id)
        quaternions = sensor_to_target.quats
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
        isd_data['reference_height'] = {
            "maxheight": 1000,
            "minheight": -1000,
            "unit": "m"
        }

    # shared interpolation needed for LineScanner and Radar
    if isinstance(driver, LineScanner) or isinstance(driver, Radar):
        interp_times = np.linspace(position_times[0],
                                   position_times[-1],
                                   int(driver.image_lines / 64))

        if velocities is not None:
            positions = np.asarray(positions)
            velocities = np.asarray(velocities)
            pos_x, pos_y, pos_z = np.asarray(positions).T
            vel_x, vel_y, vel_z = np.asarray(velocities).T
            x_interp = BPoly.from_derivatives(position_times,
                                              np.vstack((pos_x, vel_x)).T,
                                              extrapolate=True)
            y_interp = BPoly.from_derivatives(position_times,
                                              np.vstack((pos_y, vel_y)).T,
                                              extrapolate=True)
            z_interp = BPoly.from_derivatives(position_times,
                                              np.vstack((pos_z, vel_z)).T,
                                              extrapolate=True)
            interp_pos = np.vstack((x_interp(interp_times),
                                    y_interp(interp_times),
                                    z_interp(interp_times))).T
            interp_vel = np.vstack((x_interp(interp_times, nu=1),
                                    y_interp(interp_times, nu=1),
                                    z_interp(interp_times, nu=1))).T
        else:
            position_interp = interp1d(position_times, positions)
            interp_pos = position_interp(interp_times)
            interp_vel = None
        isd_data['sensor_position'] = {
            'positions' : interp_pos,
            'velocities' : interp_vel,
            'unit' : 'm'
        }
        if len(interp_times) > 1:
            isd_data['dt_ephemeris'] = (interp_times[-1] - interp_times[0]) / (len(interp_times) - 1)
        else:
            isd_data['dt_ephemeris'] = 0

        isd_data['t0_ephemeris'] = interp_times[0]

    # line scan sensor model specifics
    if isinstance(driver, LineScanner):
        isd_data['name_model'] = 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'
        isd_data['interpolation_method'] = 'lagrange'

        start_lines, start_times, scan_rates = driver.line_scan_rate
        center_time = driver.center_ephemeris_time
        isd_data['line_scan_rate'] = [[line, time, rate] for line, time, rate in zip(start_lines, start_times, scan_rates)]
        isd_data['starting_ephemeris_time'] = driver.ephemeris_start_time
        isd_data['center_ephemeris_time'] = center_time

        rotation_interp = sensor_to_target.reinterpolate(interp_times)
        isd_data['sensor_orientation'] = {
            'quaternions' : rotation_interp.quats
        }

        isd_data['t0_ephemeris'] = interp_times[0] - center_time

        isd_data['t0_quaternion'] = isd_data['t0_ephemeris']
        isd_data['dt_quaternion'] = isd_data['dt_ephemeris']


    # frame sensor model specifics
    if isinstance(driver, Framer):
        isd_data['name_model'] = 'USGS_ASTRO_FRAME_SENSOR_MODEL'
        isd_data['center_ephemeris_time'] = driver.center_ephemeris_time

    # radar sensor model specifics
    if isinstance(driver, Radar):
        isd_data['name_model'] = 'USGS_ASTRO_SAR_SENSOR_MODEL'
        isd_data['starting_ephemeris_time'] = driver.ephemeris_start_time
        isd_data['center_ephemeris_time'] = driver.center_ephemeris_time
        isd_data['ending_ephemeris_time'] = driver.ephemeris_stop_time
        isd_data['wavelength'] = driver.wavelength
        isd_data['line_exposure_duration'] = driver.line_exposure_duration
        isd_data['scaled_pixel_width'] = driver.scaled_pixel_width
        isd_data['range_conversion_times'] = driver.range_conversion_times
        isd_data['range_conversion_coefficients'] = driver.range_conversion_coefficients
        isd_data['look_direction'] = driver.look_direction

    # check that there is a valid sensor model name
    if 'name_model' not in isd_data:
        raise Exception('No CSM sensor model name found!')

    # Convert to JSON object
    return isd_data
