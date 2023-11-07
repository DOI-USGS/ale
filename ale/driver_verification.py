import argparse
import os
import shutil
import subprocess
from pathlib import Path

# Define the function to run spiceinit with ISIS
def run_spiceinit_isis(image_path):
    # Move ALE drivers to a temporary subfolder
    ale_drivers_path = Path('path/to/ale/drivers')
    temp_folder = ale_drivers_path.parent / 'temp_drivers'
    temp_folder.mkdir(exist_ok=True)
    for driver in ale_drivers_path.glob('*'):
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

# Define the function to generate ISD using the provided script
def generate_isd(image_path, driver_type):
    isd_path = f"{image_path.stem}-{driver_type}.isd"
    subprocess.run(['python', 'isd_generate_script.py', '-o', isd_path, 'input', str(image_path)])

# Define the function to compare ISDs
def compare_isds(isd1, isd2):
    # Implement the comparison logic here
    pass

# Define the main function
def main(mission_name, images_list):
    # Map mission names to drivers
    mission_to_driver = {
        'mission1': 'driver1',
        'mission2': 'driver2',
        # Add more mappings as needed
    }
    
    # Check if the mission name is valid
    if mission_name not in mission_to_driver:
        raise ValueError("Invalid mission name provided.")
    
    # Process each image
    for image_path in images_list:
        # Duplicate the image for ALE and ISIS processing
        image_ale_path = Path(f"{image_path.stem}_ALE{image_path.suffix}")
        image_isis_path = Path(f"{image_path.stem}_ISIS{image_path.suffix}")
        shutil.copy(image_path, image_ale_path)
        shutil.copy(image_path, image_isis_path)
        
        # Run spiceinit with ISIS
        run_spiceinit_isis(image_isis_path)
        
        # Run spiceinit with ALE
        run_spiceinit_ale(image_ale_path)
        
        # Generate ISD for both ALE and ISIS
        generate_isd(image_ale_path, 'ALE')
        generate_isd(image_isis_path, 'ISIS')
        
        # Compare the ISDs
        compare_isds(image_ale_path.with_suffix('.isd'), image_isis_path.with_suffix('.isd'))

        # Clean up duplicated images if needed
        image_ale_path.unlink()
        image_isis_path.unlink()

# Set up argparse to handle command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to compare ALE and ISIS3 drivers.")
    parser.add_argument('mission_name', type=str, help='Name of the mission to determine the driver.')
    parser.add_argument('images', type=str, nargs='+', help='List of image files to process.') #I am not convicned this is the best way to read images in yet
    args = parser.parse_args()
    
    # Convert image paths to Path objects
    images_list = [Path(image) for image in args.images]
    
    # Call the main function
    main(args.mission_name, images_list)

