# Importing the required libraries for demonstration
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from read_ply_file import sorted_filtered_ply_paths

# Define ply paths
ply_paths = sorted_filtered_ply_paths
frame_count = len(ply_paths)
print(f"Number of frames: {frame_count}")

# Sample function to simulate the behavior of 'adapted_plot_with_same_color_scale'
def adapted_plot_with_same_color_scale(frame, ax):
    ax.clear()
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x + frame * 0.1)
    ax.plot(x, y)
    ax.set_title(f"Frame {frame}")

# Create a figure and an axis
fig, ax = plt.subplots()

# Create an animation
animation = FuncAnimation(fig=fig, func=adapted_plot_with_same_color_scale, frames=range(frame_count), fargs=(ax,))

# To display the animation inline (may not work in all environments)
plt.show()

# To save the animation
# animation.save('sine_wave_animation.mp4', writer='ffmpeg')

from collections import defaultdict

# Initialize storage for intersection points for all times
all_intersection_data = {}

for timestamp, ply_path in ply_paths.items():
    print(f"Processing PLY file for timestamp: {timestamp}")
    print(f"Path: {ply_path}")

    # Read the PLY file
    ply_data = plyfile.PlyData.read(ply_path)
    vertex_data = np.vstack([ply_data['vertex'][axis] for axis in ['x', 'y', 'z']]).T
    temperatures = ply_data['vertex']['temperature']

    # Define lab dimensions and center
    chamber_width, chamber_length, chamber_height = 5500, 5500, 2500
    chamber_center = np.array([chamber_width / 2, chamber_length / 2, chamber_height / 2])

    # Adjust vertex coordinates to be relative to lab center
    adjusted_vertices = vertex_data + chamber_center

    # Define face equations
    faces = {
        "Ceiling": [0, 0, -1, (0, 0, chamber_height)],
        "Floor": [0, 0, 1, (chamber_center[0], chamber_center[1], 0)],
        "Front wall": [0, -1, 0, (chamber_center[0], chamber_length, chamber_center[2])],
        "Back wall": [0, 1, 0, (chamber_center[0], 0, chamber_center[2])],
        "Left wall": [-1, 0, 0, (chamber_width, chamber_center[1], chamber_center[2])],
        "Right wall": [1, 0, 0, (0, chamber_center[1], chamber_center[2])],
    }

    # Initialize storage for intersection points for this timestamp
    intersection_data = defaultdict(list)

    # Your code to populate intersection_data goes here.

    # Store the intersection data for this timestamp
    all_intersection_data[timestamp] = intersection_data

# Now, all_intersection_data contains the intersection data for all timestamps.
