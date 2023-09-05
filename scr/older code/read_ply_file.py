import os
import re
from datetime import datetime
from congigration import *

# Directory where your PLY files are stored
directory_path = os.path.join(MRT_SENSOR_DATA_DIRECTORY, "sample ply data")
print(f"Scanning directory: {directory_path}")

# Regular expression pattern for filename
pattern = re.compile(r'ThermalArray_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})_\d+.PLY')

# Date-time range to filter the files (can be adjusted as needed)
start_time = datetime.strptime('08-28-2023 09:00:00', '%m-%d-%Y %H:%M:%S')
end_time = datetime.strptime('09-01-2023 17:00:00', '%m-%d-%Y %H:%M:%S')
print(f"Filtering files between {start_time} and {end_time}")

# List to hold the sorted and filtered ply paths
sorted_filtered_ply_paths = []

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    # print(f"Checking file: {filename}")
    match = pattern.match(filename)
    if match:
        # Extract the timestamp from the filename
        timestamp_str = match.group(1)
        # print(f"Timestamp found: {timestamp_str}")
        timestamp = datetime.strptime(timestamp_str, '%m-%d-%Y_%H-%M-%S')

        # Filter based on the date-time range
        if start_time <= timestamp <= end_time:
            sorted_filtered_ply_paths.append({
                'time': timestamp,
                'path': os.path.join(directory_path, filename)
            })


# Sort the list based on the timestamp
sorted_filtered_ply_paths.sort(key=lambda x: x['time'])

# Now, sorted_filtered_ply_paths contains the paths to your sorted and filtered PLY files
print("PLY files have been read successfully!")
print(sorted_filtered_ply_paths)