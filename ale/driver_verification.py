import argparse
import os
import shutil
import subprocess
import difflib
from pathlib import Path

# Define the function to run spiceinit with ISIS
def run_spiceinit_isis(image_path):
    # Move ALE drivers to a temporary subfolder
    ale_drivers_path = Path('$ALEROOT/ale/drivers')
    temp_folder = ale_drivers_path.parent / 'temp_drivers'
    temp_folder.mkdir(exist_ok=True)
    for driver in ale_drivers_path.glob('*'): # this globs wrong
        shutil.move(str(driver), str(temp_folder))
    
    # Run spiceinit with ISIS
    subprocess.run(['spiceinit', f'from={image_path}'])
    
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
    with open(isd1, 'r') as file1:
        isd1_content = file1.readlines()

    with open(isd2, 'r') as file2:
        isd2_content = file2.readlines()

    # Use difflib to get the differences
    diff = difflib.unified_diff(isd1_content, isd2_content, lineterm='')

    # Count the number of differing lines
    diff_count = sum(1 for line in diff if line.startswith('+') or line.startswith('-'))

    return diff_count

# Define the main function
def main(driver, image):
    
    # Check if the mission name is valid
    if driver not in drivers:
        raise ValueError("Invalid driver name provided.")
    
    # Duplicate the image for ALE and ISIS processing
    image_ale_path = Path(f"{image.stem}_ALE{image.suffix}")
    image_isis_path = Path(f"{image.stem}_ISIS{image.suffix}")
    shutil.copy(image, image_ale_path)
    shutil.copy(image, image_isis_path)
        
    # Run spiceinit with ISIS
    run_spiceinit_isis(image_isis_path)
        
    # Run spiceinit with ALE
    run_spiceinit_ale(image_ale_path)
        
    # Generate ISD for both ALE and ISIS
    subprocess.run(['python3 isd_generate.py', f'input={image_ale_path}'])
    subprocess.run(['python3 isd_generate.py', f'input={image_isis_path}'])
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
    parser.add_argument('image', type=str, nargs='+', help='Image to process.')
    args = parser.parse_args()

    # Call the main function
    main(args.driver, images)
