import os

# Original data is in our Google drive foulder (Please update as needed)
MAIN_DIRECTORY = "XXX"

# Current working directory
PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


# Define the directories to be created
directories = {
    "data": {"main": "data"},
}

def create_directory(path):
    """This function checks if a directory exists at the given path,
    and if it doesn't, it creates one."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Create directories
# Loop through the main directories and their subdirectories defined in the 'directories' dictionary.
# For each directory, create a path using the current working directory and the directory name, then call the 'create_directory' function.
for main_dir, sub_dirs in directories.items():
    main_path = os.path.join(PROJECT_DIRECTORY, main_dir)
    create_directory(main_path)
    # if sub_dirs:
    #     for sub_dir in sub_dirs["sub"]:
    #         sub_path = os.path.join(main_path, sub_dir)
    #         create_directory(sub_path)

# Define the paths for later use in the code.
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "data")
MRT_SENSOR_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "princeton mrt sensor")


EXP_RECORD_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "experimental record")
