import ale.drivers
import argparse
import importlib
import pkgutil
import os
import shutil
import subprocess
import difflib
from pathlib import Path

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
            return True
    print(f"Driver {driver_name} does not exist.")
    return False

# Define the function to run spiceinit with ISIS
def run_spiceinit_isis(image_path):
    ale_root = os.environ.get('ALEROOT')
    if ale_root is None:
        raise EnvironmentError("The environment variable 'ALEROOT' is not set.")
    # Move ALE drivers to a temporary subfolder
    ale_drivers_path = Path(ale_root) / 'ale' / 'temp_drivers'
    temp_folder = ale_drivers_path.parent / 'temp_drivers'
    temp_folder.mkdir(exist_ok=True)
    for driver in ale_drivers_path.glob('*'): # this globs wrong
        shutil.move(str(driver), str(temp_folder))
    
    # Run spiceinit with ISIS
    subprocess.run(['spiceinit', f'from={image_path}']) # I believe this is where the crashes are coming from
    
    # Move the drivers back
    for driver in temp_folder.glob('*'):
        shutil.move(str(driver), str(ale_drivers_path))
    temp_folder.rmdir()

# Define the function to run spiceinit with ALE
def run_spiceinit_ale(image_path):
    # Run spiceinit with ALE
    subprocess.run(['spiceinit', f'from={image_path}'])

# Define the function to compare ISDs
def compare_isds(isd1, isd2):
    # Read the contents of the ISD files
    with open(isd1, 'r') as file_1:
        isd1_content = file_1.readlines()

    with open(isd2, 'r') as file_2:
        isd2_content = file_2.readlines()

    # Use difflib to get the differences
    diff = difflib.unified_diff(isd1_content, isd2_content, lineterm='')

    # Count the number of differing lines
    diff_count = sum(1 for line in diff if line.startswith('+') or line.startswith('-'))

    return diff_count

# Define the main function
def main(driver, image):
    
    # Check if the mission name is valid
    if driver_exists(driver):
        pass
    else:
        raise ValueError("Invalid driver name provided.")
    
    # Duplicate the image for ALE and ISIS processing
    image_ale_path = Path(f"{image}_ALE.cub")
    image_isis_path = Path(f"{image}_ISIS.cub")
    shutil.copy(image, image_ale_path)
    shutil.copy(image, image_isis_path)
        
    # Run spiceinit with ISIS
    run_spiceinit_isis(image_isis_path)
        
    # Run spiceinit with ALE
    run_spiceinit_ale(image_ale_path)
        
    # Generate ISD for both ALE and ISIS
    subprocess.run(['isd_generate', f'input={image_ale_path}'])
    subprocess.run(['isd_generate', f'input={image_isis_path}'])
    # because there isn't isis spice drivers for every driver
    # it is probably better to use tabledump
        
    # Compare the ISDs
    compare_isds(image_ale_path.with_suffix('.json'), image_isis_path.with_suffix('.json'))

    # Clean up duplicated images if needed
    image_ale_path.unlink()
    image_isis_path.unlink()

# Set up argparse to handle command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to compare ALE driver and ISIS3 driver against an image.")
    parser.add_argument('driver', type=str, help='Name of the driver to utilize.')
    parser.add_argument('image', type=str, help='Image to process.')
    args = parser.parse_args()

    # Call the main function
    main(args.driver, args.image)
