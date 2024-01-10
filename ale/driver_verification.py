#!/usr/bin/env python
import ale
from ale import drivers
from ale.base.base import Driver
from ale.base.data_isis import IsisSpice
from ale.base.label_isis import IsisLabel

class ReadIsis(IsisSpice, IsisLabel, Driver):
    def sensor_model_version(self):
        return 0

from networkx.algorithms.shortest_paths.generic import shortest_path
from importlib import reload
import argparse
import importlib
import pvl
import pkgutil
import os
ale_root = os.environ.get('ALEROOT')
import shutil
import subprocess
import difflib
from pathlib import Path
import numpy as np

# function to check if driver attribute exists in any given driver class file
def driver_exists(driver_name):
    # Find the package path
    package_path = ale.drivers.__path__

    # Iterate over all modules in the ale.drivers package
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"ale.drivers.{module_name}"
        module = importlib.import_module(full_module_name)
        # Check if the driver class exists in the module
        if hasattr(module, driver_name):
            return getattr(module, driver_name)
    print(f"Driver {driver_name} does not exist.")
    return None

# Define the function to run spiceinit with ISIS
def run_spiceinit_isis(image_path):
    if ale_root is None:
        raise EnvironmentError("The environment variable 'ALEROOT' is not set.")
    # Move ALE drivers to a temporary subfolder
    ale_drivers_path = Path(ale_root) / 'ale' / 'drivers'
    temp_folder = ale_drivers_path.parent / 'temp_drivers'
    temp_folder.mkdir(exist_ok=True)
    for driver in ale_drivers_path.glob('*'): # this globs wrong
        shutil.move(str(driver), str(temp_folder))
    
    # Run spiceinit with ISIS
    try:
        subprocess.run(['spiceinit', f'from={image_path}']) # I believe this is where the crashes are coming from
    except:
        pass
    
    # Move the drivers back
    for driver in temp_folder.glob('*'):
        shutil.move(str(driver), str(ale_drivers_path))
    temp_folder.rmdir()

# Define the function to run spiceinit with ALE
def run_spiceinit_ale(image_path):
    # Run spiceinit with ALE
    subprocess.run(['spiceinit', f'from={image_path}'])

def generate_body_rotation(driver, target_frame_id):
    frame_chain = driver.frame_chain
    target_frame = target_frame_id

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
    return body_rotation

def generate_instrument_rotation(driver, sensor_frame_id):
    # sensor orientation
    frame_chain = driver.frame_chain
    sensor_frame = sensor_frame_id

    J2000 = 1 # J2000 frame id
    instrument_pointing = {}
    source_frame, destination_frame, _ = frame_chain.last_time_dependent_frame_between(J2000, sensor_frame)

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
    
    return instrument_pointing

def generate_instrument_position(driver):
    instrument_position = {}
    positions, velocities, times = driver.sensor_position
    instrument_position['spk_table_start_time'] = times[0]
    instrument_position['spk_table_end_time'] = times[-1]
    instrument_position['spk_table_original_size'] = len(times)
    instrument_position['ephemeris_times'] = times
    # Rotate positions and velocities into J2000 then scale into kilometers
    # velocities = j2000_rotation.rotate_velocity_at(positions, velocities, times)/1000
    # positions = j2000_rotation.apply_at(positions, times)/1000
    instrument_position['positions'] = positions
    instrument_position['velocities'] = velocities
    return instrument_position

def generate_sun_position(driver):
    sun_position = {}
    positions, velocities, times = driver.sun_position
    sun_position['spk_table_start_time'] = times[0]
    sun_position['spk_table_end_time'] = times[-1]
    sun_position['spk_table_original_size'] = len(times)
    sun_position['ephemeris_times'] = times
    # Rotate positions and velocities into J2000 then scale into kilometers
    # velocities = j2000_rotation.rotate_velocity_at(positions, velocities, times)/1000
    # positions = j2000_rotation.apply_at(positions, times)/1000
    sun_position['positions'] = positions
    sun_position['velocities'] = velocities
    # sun_position["reference_frame"] = j2000_rotation.dest
    return sun_position

def create_json_dump(driver, sensor_frame_id, target_frame_id):
    json_dump = {}
    json_dump["instrument_rotation"] = generate_instrument_rotation(driver, sensor_frame_id)
    json_dump["body_rotation"] = generate_body_rotation(driver, target_frame_id)
    json_dump["instrument_position"] = generate_instrument_position(driver)
    json_dump["sun_position"] = generate_sun_position(driver)
    return json_dump

def diff_and_describe(json1, json2, key_array):
    for key in key_array:
        json1 = json1[key]
        json2 = json2[key]
    diff = json1 - json2
    print(len(diff), np.mean(diff, axis=(0)), np.median(diff, axis=(0)))


# Define the function to compare ISDs
def compare_isds(json1, json2):
    diff_and_describe(json1, json2, ["instrument_position", "positions"])
    diff_and_describe(json1, json2, ["instrument_position", "velocities"])
    diff_and_describe(json1, json2, ["sun_position", "positions"])
    diff_and_describe(json1, json2, ["sun_position", "velocities"])
    diff_and_describe(json1, json2, ["instrument_rotation", "quaternions"])
    diff_and_describe(json1, json2, ["instrument_rotation", "angular_velocities"])
    diff_and_describe(json1, json2, ["body_rotation", "quaternions"])
    diff_and_describe(json1, json2, ["body_rotation", "angular_velocities"])

# Define the main function
def main(driver, image):
    
    # Check if the mission name is valid
    driver_class = driver_exists(driver)
    if driver_class == None:
        raise ValueError("Invalid driver name provided.")
    
    # Duplicate the image for ALE and ISIS processing
    image_ale_path = Path(f"{image}_ALE.cub")
    image_isis_path = Path(f"{image}_ISIS.cub")
    shutil.copy(image, image_ale_path)
    shutil.copy(image, image_isis_path)

    # Run spiceinit with ISIS
    run_spiceinit_isis(image_isis_path)

    # try ale.loads
    isis_kerns = ale.util.generate_kernels_from_cube(image_isis_path, expand=True)
    ale.loads(image_isis_path, props={"kernels": isis_kerns}, only_naif_spice=True)
    
    # Run spiceinit with ALE
    run_spiceinit_ale(image_ale_path)

    # try ale.loads
    ale_kerns = ale.util.generate_kernels_from_cube(image_ale_path, expand=True)
    ale.loads(image_ale_path, props={"kernels": ale_kerns}, only_naif_spice=True)
    
    # Generate ISD for both ALE and ISIS
    read_ale_driver = ReadIsis(image_ale_path)
    ale_json_dump = create_json_dump(read_ale_driver, read_ale_driver.sensor_frame_id, read_ale_driver.target_frame_id)
    read_isis_driver = ReadIsis(image_isis_path)
    isis_json_dump = create_json_dump(read_isis_driver, read_isis_driver.sensor_frame_id, read_isis_driver.target_frame_id)
    # because there isn't isis spice drivers for every driver
    # it is probably better to use tabledump
        
    # Compare the ISDs
    compare_isds(ale_json_dump, isis_json_dump)

# Set up argparse to handle command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to compare ALE driver and ISIS3 driver against an image.")
    parser.add_argument('driver', type=str, help='Name of the driver to utilize.')
    parser.add_argument('image', type=str, help='Image to process.')
    args = parser.parse_args()

    # Call the main function
    main(args.driver, args.image)

