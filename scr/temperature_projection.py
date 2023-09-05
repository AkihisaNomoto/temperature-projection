import os
import csv
import re
from datetime import datetime
import numpy as np
import plyfile
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

from congigration import *


# Directory where your PLY files are stored
directory_path = os.path.join(MRT_SENSOR_DATA_DIRECTORY, "sample ply data")
start_time_str = "08-28-2023 09:00:00"
end_time_str = "09-01-2023 17:00:00"

# Define panel coordinates
panel_zone_coordinates = {
    "Zone1": {
        "Top Left": (1835, 0),
        "Top Right": (3665, 0),
        "Bottom Left": (1835, 1220),
        "Bottom Right": (3665, 1220),
    },
    "Zone2": {
        "Top Left": (610, 1220),
        "Top Right": (2445, 1220),
        "Bottom Left": (615, 3660),
        "Bottom Right": (2445, 3660),
    },
    "Zone3": {
        "Top Left": (1835, 3660),
        "Top Right": (3665, 3660),
        "Bottom Left": (1835, 4880),
        "Bottom Right": (3665, 4880),
    },
    "Zone4": {
        "Top Left": (3055, 1220),
        "Top Right": (4885, 1220),
        "Bottom Left": (3055, 3660),
        "Bottom Right": (4885, 3660),
    },
}

# Define a flag outside the function to keep track of whether the colorbar has been added.
colorbar_added = False


def read_ply_data(ply_path):
    """
    Read data from a PLY file.

    Parameters:
        ply_path (str): The path to the PLY file.

    Returns:
        np.ndarray: The vertex data.
        np.ndarray: The temperature data.
    """
    # Read the PLY file
    ply_data = plyfile.PlyData.read(ply_path)
    vertex_data = np.vstack([ply_data["vertex"][axis] for axis in ["x", "y", "z"]]).T
    temperatures = ply_data["vertex"]["temperature"]
    return vertex_data, temperatures


def get_sorted_filtered_ply_dict(directory_path, start_time_str, end_time_str):
    """
    Reads PLY files from a directory, filters them based on timestamps, and sorts them.

    Parameters:
        directory_path (str): The directory where the PLY files are stored.
        start_time_str (str): The start time for filtering files in the format 'MM-DD-YYYY HH:MM:SS'.
        end_time_str (str): The end time for filtering files in the format 'MM-DD-YYYY HH:MM:SS'.

    Returns:
        dict: A dictionary containing sorted and filtered PLY files.
    """

    pattern = re.compile(r"ThermalArray_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})_\d+.PLY")
    start_time = datetime.strptime(start_time_str, "%m-%d-%Y %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%m-%d-%Y %H:%M:%S")

    sorted_filtered_ply_dict = {}

    for filename in os.listdir(directory_path):
        match = pattern.match(filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%m-%d-%Y_%H-%M-%S")
            if start_time <= timestamp <= end_time:
                sorted_filtered_ply_dict[timestamp] = os.path.join(
                    directory_path, filename
                )

    return sorted_filtered_ply_dict


def line_plane_intersection(point_on_line, line_direction, plane_eq):
    """
    Calculate the intersection point between a line and a plane in 3D space.

    Parameters:
    - point_on_line (list or array): A point on the line defined as [x, y, z].
    - line_direction (list or array): The direction vector of the line defined as [dx, dy, dz].
    - plane_eq (list or array): The equation of the plane defined as [a, b, c, x0, y0, z0],
      where [a, b, c] is the normal vector of the plane, and [x0, y0, z0] is a point on the plane.

    Returns:
    - numpy array: The intersection point [x, y, z].
    """
    # Convert input lists to numpy arrays for easier manipulation
    point_on_line = np.array(point_on_line)
    line_direction = np.array(line_direction)
    plane_normal = np.array(plane_eq[:3])
    plane_point = np.array(plane_eq[3:])

    # Calculate the intersection point using the formula
    # \[ d = \frac{(plane_point - point_on_line) \cdot plane_normal}{line_direction \cdot plane_normal} \]
    # \[ intersection = point_on_line + d * line_direction \]
    d = np.dot(plane_point - point_on_line, plane_normal) / np.dot(
        line_direction, plane_normal
    )
    intersection = point_on_line + d * line_direction

    return intersection


def define_walls_and_panels(chamber_dimensions):
    """
    Define the walls and panel zones of a chamber based on its dimensions.

    Parameters:
    - chamber_dimensions (tuple): The dimensions of the chamber defined as (width, length, height).

    Returns:
    - dict: A dictionary containing the coordinates of the walls' corners.
    - dict: A dictionary containing the coordinates of the panel zones' corners.
    """
    # Extract width, length, and height from chamber_dimensions tuple
    w, l, h = chamber_dimensions

    # Define walls with their corner coordinates
    # The coordinates are defined in a counter-clockwise manner starting from the bottom-left corner when facing the wall.
    walls = {
        "Ceiling": [(0, 0, h), (w, 0, h), (w, l, h), (0, l, h)],
        "Floor": [(0, 0, 0), (w, 0, 0), (w, l, 0), (0, l, 0)],
        "Front wall": [(0, l, 0), (w, l, 0), (w, l, h), (0, l, h)],
        "Back wall": [(0, 0, 0), (w, 0, 0), (w, 0, h), (0, 0, h)],
        "Left wall": [(w, 0, 0), (w, l, 0), (w, l, h), (w, 0, h)],
        "Right wall": [(0, 0, 0), (0, l, 0), (0, l, h), (0, 0, h)],
    }

    # Define panel zones with their corner coordinates
    # These zones are located on the ceiling and are defined by their corner coordinates.
    panel_zones = {
        "Zone1": [(1835, 0, h), (3665, 0, h), (3665, 1220, h), (1835, 1220, h)],
        "Zone2": [(610, 1220, h), (2445, 1220, h), (2445, 3660, h), (610, 3660, h)],
        "Zone3": [(1835, 3660, h), (3665, 3660, h), (3665, 4880, h), (1835, 4880, h)],
        "Zone4": [(3055, 1220, h), (4885, 1220, h), (4885, 3660, h), (3055, 3660, h)],
    }

    return walls, panel_zones


def gather_intersection_and_panel_data(
    adjusted_vertices,
    temperatures,
    faces,
    panel_zone_coordinates,
    chamber_dimensions,
    chamber_center,
):
    """
    Gather intersection and panel data, and calculate statistics for each zone within the chamber.

    Parameters:
    - adjusted_vertices (np.ndarray): An array of adjusted vertex coordinates, each defined as [x, y, z].
    - temperatures (np.ndarray): An array of temperatures corresponding to each vertex.
    - faces (dict): A dictionary containing the equations for each face (wall or panel zone).
    - panel_zone_coordinates (dict): The coordinates of the panel zones.
    - chamber_dimensions (tuple): The dimensions of the chamber defined as (width, length, height).
    - chamber_center (list or array): The coordinates of the chamber's center [x, y, z].

    Returns:
    - dict: A dictionary containing the statistics (average and median temperature) for each zone.
    """

    # Extract width, length, and height from chamber_dimensions tuple
    w, l, h = chamber_dimensions

    # Initialize dictionaries to hold the intersection data and panel data for each zone
    intersection_data = defaultdict(list)
    panel_intersection_data = defaultdict(list)

    # Helper function: To calculate zone statistics (average and median temperature)
    def calculate_zone_statistics(intersection_data):
        zone_statistics = {}
        for zone_name, data in intersection_data.items():
            temps = [temp for _, temp in data]
            avg_temp = np.mean(temps)
            median_temp = np.median(temps)
            zone_statistics[zone_name] = {"average": avg_temp, "median": median_temp}
        return zone_statistics

    # Helper function: To check if a point is within a given panel zone
    def is_point_in_zone(x, y, top_left, top_right, bottom_left, chamber_height, z):
        return (
            top_left[0] <= x <= top_right[0]
            and top_left[1] <= y <= bottom_left[1]
            and z == chamber_height
        )

    # Loop through each vertex and its corresponding temperature
    for point, temp in zip(adjusted_vertices, temperatures):
        # Calculate intersection data for walls and other faces
        for face_name, face_eq in faces.items():
            intersection_point = line_plane_intersection(
                chamber_center, point - chamber_center, face_eq
            )
            x, y, z = intersection_point
            if 0 <= x <= w and 0 <= y <= l and 0 <= z <= h:
                vector_from_center_to_point = point - chamber_center
                vector_from_center_to_intersection = intersection_point - chamber_center
                if (
                    np.dot(
                        vector_from_center_to_point, vector_from_center_to_intersection
                    )
                    > 0
                ):
                    intersection_data[face_name].append((intersection_point, temp))

        # Calculate intersection data for panel zones
        for panel_name, panel_coords in panel_zone_coordinates.items():
            top_left, top_right, bottom_left, bottom_right = panel_coords.values()
            panel_face_eq = [0, 0, -1, (0, 0, h)]
            intersection_point = line_plane_intersection(
                chamber_center, point - chamber_center, panel_face_eq
            )
            x, y, z = intersection_point
            if is_point_in_zone(x, y, top_left, top_right, bottom_left, h, z):
                vector_from_center_to_point = point - chamber_center
                vector_from_center_to_intersection = intersection_point - chamber_center
                if (
                    np.dot(
                        vector_from_center_to_point, vector_from_center_to_intersection
                    )
                    >= 0
                ):
                    panel_intersection_data[panel_name].append(
                        (intersection_point, temp)
                    )

    # Calculate statistics
    wall_statistics = calculate_zone_statistics(intersection_data)
    panel_statistics = calculate_zone_statistics(panel_intersection_data)

    # Combine wall and panel statistics into one dictionary
    all_statistics = {**wall_statistics, **panel_statistics}

    # Sort the dictionary by zone names
    sorted_statistics = OrderedDict(sorted(all_statistics.items(), key=lambda x: x[0]))

    return sorted_statistics


def adapted_plot_with_same_color_scale(
    ax,
    all_statistics,
    chamber_dimensions,
    frame_name,
    min_temp=15,
    max_temp=30,
    show_temp_point=False,
):
    """
    Create a plot with the same color scale.

    Parameters:
        all_statistics (dict): The aggregated statistics for each zone at a specific timestamp.
        chamber_dimensions (tuple): The dimensions of the chamber (width, length, height).
        min_temp (float): The minimum temperature value for the color scale.
        max_temp (float): The maximum temperature value for the color scale.
        frame_name (str): The name of the frame (e.g., timestamp).

    Returns:
        None
    """
    global colorbar_added  # Use this line if you're using a global variable for the flag

    def plot_face(ax, corners, color, alpha=1.0):
        """Plots a single face on a given matplotlib 3D axis (ax)."""
        poly3d = [[list(corner) for corner in corners]]
        ax.add_collection3d(
            Poly3DCollection(
                poly3d, facecolors=color, linewidths=1, edgecolors="k", alpha=alpha
            )
        )

    def add_colorbar(ax, cmap, min_temp, max_temp):
        sc = ax.scatter([], [], [], c=[], cmap=cmap, vmin=min_temp, vmax=max_temp)
        cbar = plt.colorbar(sc, label="Temperature (°C)")
        cbar.set_label("Temperature (°C)", rotation=270, labelpad=15)

    def add_median_temps(ax, all_statistics):
        text_position = [-0.4, 0.5]
        legend_text = "Median Temperatures (°C)\n\n"
        for name, stats in all_statistics.items():
            temp = round(stats.get("median", 0), 2)
            temp_str = "{:.2f}".format(
                temp
            )  # Explicitly control the number of decimal places
            legend_text += f"{name}: {temp_str}\n"
        ax.text2D(
            text_position[0],
            text_position[1],
            legend_text.rstrip(),
            transform=ax.transAxes,
            fontsize=10,
        )

    def add_frame_info(ax, frame_name):
        text_position = [-0.4, 1]
        ax.text2D(
            text_position[0],
            text_position[1],
            f'Time: {frame_name}',
            transform=ax.transAxes,
            fontsize=10,
        )

    def set_plot_labels_and_limits(ax, chamber_dimensions):
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_xlim([0, chamber_dimensions[0]])
        ax.set_ylim([0, chamber_dimensions[1]])
        ax.set_zlim([0, chamber_dimensions[2]])

    # Clear the previous plot elements from the ax
    ax.clear()

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')

    walls, panel_zones = define_walls_and_panels(chamber_dimensions)

    cmap = cm.seismic
    norm = Normalize(vmin=min_temp, vmax=max_temp)

    for name_type, entities in [("Panel", panel_zones), ("Wall", walls)]:
        for name, corners in entities.items():
            temp = all_statistics.get(name, {}).get("median", 0)
            color = cmap(norm(temp))
            plot_face(ax, corners, color, alpha=0.7 if name_type == "Panel" else 0.5)

    if show_temp_point:
        # If you still want to show intersection points, you may need to pass them separately or get them from all_statistics
        pass

    # Only add the colorbar if it hasn't been added yet
    if not colorbar_added:
        add_colorbar(ax, cmap, min_temp, max_temp)
        colorbar_added = True  # Set the flag to True after adding the colorbar

    add_median_temps(ax, all_statistics)
    add_frame_info(ax, frame_name)
    set_plot_labels_and_limits(ax, chamber_dimensions)

    # No need for animation
    # plt.show()


# def save_statistics_to_csv(all_statistics, csv_file_path):
#     """
#     Save the collected statistics to a CSV file, with each row containing all statistics for a single timestamp.
#
#     Parameters:
#     - all_statistics (dict): A dictionary containing the statistics (average and median temperature) for each zone at different timestamps.
#     - csv_file_path (str): The path where the CSV file will be saved.
#     """
#     # Open the CSV file for writing
#     with open(csv_file_path, "w", newline="") as csvfile:
#         # Identify the unique zones involved
#         example_timestamp = next(iter(all_statistics.keys()))
#         unique_zones = [zone for zone in all_statistics[example_timestamp].keys() if zone not in ['scene_index', 'elapsed_time']]
#
#         # Initialize the CSV writer
#         csvwriter = csv.writer(csvfile)
#
#         # Write the header row
#         header = ["Timestamp", "Scene Index", "Elapsed Time (min)"]
#         for zone in unique_zones:
#             header.extend([f"{zone}_mean_C", f"{zone}_median_C"])
#         csvwriter.writerow(header)
#
#         # Write the statistics data
#         for timestamp, stats in all_statistics.items():
#             row = [timestamp, stats.get('scene_index', None), stats.get('elapsed_time', None)]
#             for zone in unique_zones:
#                 if zone in stats:
#                     row.extend([stats[zone].get("average", None), stats[zone].get("median", None)])
#                 else:
#                     row.extend([None, None])
#             csvwriter.writerow(row)


def save_statistics_to_csv(all_statistics, csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        example_timestamp = next(iter(all_statistics.keys()))
        unique_zones = [zone for zone in all_statistics[example_timestamp].keys() if
                        zone not in ['scene_index', 'elapsed_time_in_min']]

        csvwriter = csv.writer(csvfile)

        header = ["Timestamp", "Scene Index", "Elapsed Time (min)"]
        for zone in unique_zones:
            header.extend([f"{zone}_mean_C", f"{zone}_median_C"])
        csvwriter.writerow(header)

    for timestamp, stats in all_statistics.items():
        row = [timestamp, stats.get('scene_index', None), stats.get('elapsed_time_in_min', None)]
        for zone in unique_zones:
            if zone in stats:
                # Check if stats[zone] is a dictionary before calling .get()
                if isinstance(stats[zone], dict):
                    row.extend([stats[zone].get("average", None), stats[zone].get("median", None)])
                else:
                    row.extend([None, None])  # if it's not a dictionary, extend the row with None
            else:
                row.extend([None, None])


def main():
    # Initialize an empty dictionary to hold all temperature statistics
    all_statistics = {}
    # Initialize variables to keep track of the scene index and starting time
    scene_index = 0
    start_time = None

    # Define the dimensions and center of the chamber
    chamber_width, chamber_length, chamber_height = 5500, 5500, 2500
    chamber_center = np.array([chamber_width / 2, chamber_length / 2, chamber_height / 2])
    chamber_dimensions = (chamber_width, chamber_length, chamber_height)

    # Define the equations for each face of the chamber
    faces = {
        "Ceiling": [0, 0, -1, (0, 0, chamber_height)],
        "Floor": [0, 0, 1, (chamber_center[0], chamber_center[1], 0)],
        "Front wall": [0, -1, 0, (chamber_center[0], chamber_length, chamber_center[2])],
        "Back wall": [0, 1, 0, (chamber_center[0], 0, chamber_center[2])],
        "Left wall": [-1, 0, 0, (chamber_width, chamber_center[1], chamber_center[2])],
        "Right wall": [1, 0, 0, (0, chamber_center[1], chamber_center[2])],
    }

    # Get a sorted list of PLY files to process (assuming the function get_sorted_filtered_ply_dict is defined)
    sorted_filtered_ply_dict = get_sorted_filtered_ply_dict(directory_path, start_time_str, end_time_str)

    # Loop through each timestamp and corresponding PLY file
    for timestamp, ply_path in sorted_filtered_ply_dict.items():

        vertex_data, temperatures = read_ply_data(ply_path=ply_path)
        adjusted_vertices = vertex_data + chamber_center

        # Gather all statistics for the current timestamp
        current_statistics = gather_intersection_and_panel_data(
            adjusted_vertices=adjusted_vertices,
            temperatures=temperatures,
            faces=faces,
            panel_zone_coordinates=panel_zone_coordinates,
            chamber_dimensions=chamber_dimensions,
            chamber_center=chamber_center
        )

        # Increment the scene index
        scene_index += 1

        # Initialize the start_time during the first iteration
        if start_time is None:
            start_time = timestamp

        # Calculate the elapsed time
        elapsed_time = timestamp - start_time

        # Store additional info in the dictionary
        current_statistics['scene_index'] = scene_index
        current_statistics['elapsed_time_in_min'] = elapsed_time.total_seconds() / 60  # in minutes


        # Store the statistics in the dictionary
        all_statistics[timestamp] = current_statistics

    print(all_statistics)


    # Save the statistics to a CSV file
    csv_file_path = 'all_statistics.csv'  # Modify the path as needed
    save_statistics_to_csv(all_statistics, csv_file_path)

    # Initialize the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the update function for the animation
    def update(frame):
        timestamps = list(all_statistics.keys())
        timestamp = timestamps[frame]

        # Check if the current timestamp exists in the statistics dictionary
        if timestamp in all_statistics:
            adapted_plot_with_same_color_scale(
                ax,
                all_statistics[timestamp],
                chamber_dimensions,
                min_temp=15,
                max_temp=30,
                frame_name=timestamp,
                show_temp_point=False
            )
        else:
            print(f"Warning: {timestamp} not found in all_statistics.")

    # Create and save the animation
    ani = FuncAnimation(fig, update, frames=len(all_statistics.keys()))
    ani.save('temperature_projection_animation.gif', writer='pillow')
    plt.show()


if __name__ == "__main__":
    main()
