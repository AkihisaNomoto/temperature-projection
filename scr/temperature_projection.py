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
start_time_str = '08-28-2023 09:00:00'
end_time_str = '09-01-2023 17:00:00'

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


# # Regular expression pattern for filename
# pattern = re.compile(r'ThermalArray_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})_\d+.PLY')
#
# # Date-time range to filter the files (can be adjusted as needed)
# start_time = datetime.strptime('08-28-2023 09:00:00', '%m-%d-%Y %H:%M:%S')
# end_time = datetime.strptime('09-01-2023 17:00:00', '%m-%d-%Y %H:%M:%S')
# print(f"Filtering files between {start_time} and {end_time}")
#
# # Dictionary to hold the sorted and filtered ply paths
# sorted_filtered_ply_dict = {}
#
# # Iterate over each file in the directory
# for filename in os.listdir(directory_path):
#     match = pattern.match(filename)
#     if match:
#         # Extract the timestamp from the filename
#         timestamp_str = match.group(1)
#         timestamp = datetime.strptime(timestamp_str, '%m-%d-%Y_%H-%M-%S')
#
#         # Filter based on the date-time range
#         if start_time <= timestamp <= end_time:
#             # Use the timestamp as the key
#             sorted_filtered_ply_dict[timestamp] = os.path.join(directory_path, filename)
#
# # Now, sorted_filtered_ply_dict contains the paths to your sorted and filtered PLY files
# # The dictionary is already sorted by the timestamp if you're using Python 3.7+
# print("PLY files have been read successfully!")
# print(sorted_filtered_ply_dict)
#
# # Define frame
# number_of_frame = len(sorted_filtered_ply_dict)
# print("Number of frame", number_of_frame)


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

    pattern = re.compile(r'ThermalArray_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})_\d+.PLY')
    start_time = datetime.strptime(start_time_str, '%m-%d-%Y %H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%m-%d-%Y %H:%M:%S')

    sorted_filtered_ply_dict = {}

    for filename in os.listdir(directory_path):
        match = pattern.match(filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%m-%d-%Y_%H-%M-%S')
            if start_time <= timestamp <= end_time:
                sorted_filtered_ply_dict[timestamp] = os.path.join(directory_path, filename)

    return sorted_filtered_ply_dict

def line_plane_intersection(point_on_line, line_direction, plane_eq):
    point_on_line = np.array(point_on_line)
    line_direction = np.array(line_direction)
    plane_normal = np.array(plane_eq[:3])
    plane_point = np.array(plane_eq[3:])
    d = np.dot(plane_point - point_on_line, plane_normal) / np.dot(line_direction, plane_normal)
    intersection = point_on_line + d * line_direction
    return intersection


# Main function to gather panel intersection data and statistics
def gather_panel_data(adjusted_vertices, temperatures, panel_zone_coordinates, chamber_center, chamber_height):
    """
    Gather panel data at a given time step.

    Parameters:
        adjusted_vertices (np.ndarray): The adjusted vertex coordinates.
        temperatures (np.ndarray): The vertex temperatures.
        panel_zone_coordinates (dict): The coordinates of the panel zones.
        chamber_center (np.ndarray): The coordinates of the chamber center.
        chamber_height (float): The height of the chamber.

    Returns:
        dict: The panel statistics.
    """
    panel_intersection_data = defaultdict(list)

    # Helper function: To calculate panel statistics (average and median temperature)
    def calculate_panel_statistics(panel_intersection_data):
        panel_statistics = {}
        for panel_name, data in panel_intersection_data.items():
            temperatures = [temp for _, temp in data]
            avg_temp = round(np.mean(temperatures), 2)
            median_temp = round(np.median(temperatures), 2)
            panel_statistics[panel_name] = {"average": avg_temp, "median": median_temp}
        return panel_statistics

    # Helper function: To check if a point is within a given panel zone
    def is_point_in_zone(x, y, top_left, top_right, bottom_left, chamber_height, z):
        return top_left[0] <= x <= top_right[0] and top_left[1] <= y <= bottom_left[1] and z == chamber_height

    for point, temp in zip(adjusted_vertices, temperatures):
        for panel_name, panel_coords in panel_zone_coordinates.items():
            top_left, top_right, bottom_left, bottom_right = panel_coords.values()

            # Using the ceiling's plane equation: [0, 0, -1, (0, 0, chamber_height)]
            panel_face_eq = [0, 0, -1, (0, 0, chamber_height)]

            intersection_point = line_plane_intersection(chamber_center, point - chamber_center, panel_face_eq)
            x, y, z = intersection_point

            if is_point_in_zone(x, y, top_left, top_right, bottom_left, chamber_height, z):
                vector_from_center_to_point = point - chamber_center
                vector_from_center_to_intersection = intersection_point - chamber_center

                if np.dot(vector_from_center_to_point, vector_from_center_to_intersection) >= 0:
                    panel_intersection_data[panel_name].append((intersection_point, temp))

    panel_statistics = calculate_panel_statistics(panel_intersection_data)
    sorted_panel_statistics = OrderedDict(sorted(panel_statistics.items(), key=lambda x: x[0]))

    return sorted_panel_statistics


# # Assuming adjusted_vertices, temperatures, panel_zone_coordinates, chamber_center, chamber_height are defined
# panel_statistics = gather_panel_data(adjusted_vertices, temperatures, panel_zone_coordinates, chamber_center,
#                                      chamber_height)
# print(panel_statistics)


def adapted_plot_with_same_color_scale(intersection_data, median_surface_temperatures, panel_statistics,
                                       chamber_dimensions, min_temp, max_temp, frame_name, show_temp_point=False):
    """
    Create a plot with the same color scale.

    Parameters:
        intersection_data (dict): The intersection data.
        median_surface_temperatures (dict): The median surface temperatures.
        panel_statistics (dict): The panel statistics.
        chamber_dimensions (tuple): The dimensions of the chamber (width, length, height).
        vmin (float): The minimum temperature value for the color scale.
        vmax (float): The maximum temperature value for the color scale.
        frame_name (str): The name of the frame (e.g., timestamp).

    Returns:
        None
    """

    def plot_face(ax, corners, color, alpha=1.0):
        """Plots a single face on a given matplotlib 3D axis (ax)."""
        poly3d = [[list(corner) for corner in corners]]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))

    def plot_intersection_points(ax, intersection_data, cmap, min_temp, max_temp):
        for face_name, data in intersection_data.items():
            points, temps = zip(*data)
            points = np.array(points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=temps, cmap=cmap, vmin=min_temp, vmax=max_temp,
                       marker='o', s=5, alpha=0.3)

    def add_colorbar(ax, cmap, min_temp, max_temp):
        sc = ax.scatter([], [], [], c=[], cmap=cmap, vmin=min_temp, vmax=max_temp)
        cbar = plt.colorbar(sc, label='Temperature (°C)')
        cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

    def add_median_temps(ax, panel_statistics, median_surface_temperatures):
        text_position = [-0.4, 0.5]
        legend_text = 'Median Temperatures (°C)\n\n'
        for entity_stats, title in [(panel_statistics, ''), (median_surface_temperatures, 'Surface')]:
            for name, stats in entity_stats.items():
                temp = round(stats.get('median', stats) if title == '' else stats, 2)
                temp_str = "{:.2f}".format(temp)  # Explicitly control the number of decimal places
                legend_text += f"{title} {name}: {temp_str}\n"
        ax.text2D(text_position[0], text_position[1], legend_text.rstrip(), transform=ax.transAxes, fontsize=10)

    def add_frame_info(ax, frame_name):
        text_position = [-0.4, 1]
        ax.text2D(text_position[0], text_position[1], frame_name, transform=ax.transAxes, fontsize=10)

    def set_plot_labels_and_limits(ax, chamber_dimensions):
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_xlim([0, chamber_dimensions[0]])
        ax.set_ylim([0, chamber_dimensions[1]])
        ax.set_zlim([0, chamber_dimensions[2]])
        ax.set_title(f'Temperature Projection on Chamber Surface')

        # Adding cardinal directions (N, W, E, S)
        offset = 100  # Define an offset distance for the labels
        ax.text(chamber_dimensions[0] / 2, chamber_dimensions[1] + offset, chamber_dimensions[2] / 2, 'North',
                fontsize=12)
        ax.text(chamber_dimensions[0] / 2, -offset, chamber_dimensions[2] / 2, 'South', fontsize=12)
        ax.text(chamber_dimensions[0] + offset, chamber_dimensions[1] / 2, chamber_dimensions[2] / 2, 'East',
                fontsize=12)
        ax.text(-offset, chamber_dimensions[1] / 2, chamber_dimensions[2] / 2, 'West', fontsize=12)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    walls, panel_zones = define_walls_and_panels(chamber_dimensions)

    cmap = cm.seismic
    norm = Normalize(vmin=min_temp, vmax=max_temp)

    for name_type, entities, stats in [('Panel', panel_zones, panel_statistics),
                                       ('Wall', walls, median_surface_temperatures)]:
        for name, corners in entities.items():
            temp = stats.get(name, {}).get('average', 0) if name_type == 'Panel' else stats.get(name, 0)
            color = cmap(norm(temp))
            plot_face(ax, corners, color, alpha=0.7 if name_type == 'Panel' else 0.5)

    if show_temp_point:
        plot_intersection_points(ax, intersection_data, cmap, min_temp, max_temp)

    add_colorbar(ax, cmap, min_temp, max_temp)
    add_median_temps(ax, panel_statistics, median_surface_temperatures)
    add_frame_info(ax, frame_name)
    set_plot_labels_and_limits(ax, chamber_dimensions)
    plt.show()


def define_walls_and_panels(chamber_dimensions):
    w, l, h = chamber_dimensions
    # Define walls
    walls = {
        "Ceiling": [(0, 0, h), (w, 0, h), (w, l, h), (0, l, h)],
        "Floor": [(0, 0, 0), (w, 0, 0), (w, l, 0), (0, l, 0)],
        "Front wall": [(0, l, 0), (w, l, 0), (w, l, h), (0, l, h)],
        "Back wall": [(0, 0, 0), (w, 0, 0), (w, 0, h), (0, 0, h)],
        "Left wall": [(w, 0, 0), (w, l, 0), (w, l, h), (w, 0, h)],
        "Right wall": [(0, 0, 0), (0, l, 0), (0, l, h), (0, 0, h)],
    }

    # Define panel zones
    panel_zones = {
        "Zone1": [(1835, 0, h), (3665, 0, h), (3665, 1220, h), (1835, 1220, h)],
        "Zone2": [(610, 1220, h), (2445, 1220, h), (2445, 3660, h), (610, 3660, h)],
        "Zone3": [(1835, 3660, h), (3665, 3660, h), (3665, 4880, h), (1835, 4880, h)],
        "Zone4": [(3055, 1220, h), (4885, 1220, h), (4885, 3660, h), (3055, 3660, h)],
    }

    return walls, panel_zones



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
    vertex_data = np.vstack([ply_data['vertex'][axis] for axis in ['x', 'y', 'z']]).T
    temperatures = ply_data['vertex']['temperature']
    return vertex_data, temperatures

def calculate_intersection_data(adjusted_vertices, temperatures, faces, chamber_center):
    """
    Calculate the intersection data.

    Parameters:
        adjusted_vertices (np.ndarray): The adjusted vertex coordinates.
        temperatures (np.ndarray): The vertex temperatures.
        faces (dict): The face equations.

    Returns:
        dict: The intersection data.
    """
    intersection_data = defaultdict(list)
    for point, temp in zip(adjusted_vertices, temperatures):
        for face_name, face_eq in faces.items():
            intersection_point = line_plane_intersection(chamber_center, point - chamber_center, face_eq)
            x, y, z = intersection_point
            if 0 <= x <= 5500 and 0 <= y <= 5500 and 0 <= z <= 2500:
                vector_from_center_to_point = point - chamber_center
                vector_from_center_to_intersection = intersection_point - chamber_center
                if np.dot(vector_from_center_to_point, vector_from_center_to_intersection) > 0:
                    intersection_data[face_name].append((intersection_point, temp))
    return intersection_data

# # Initialize storage for intersection points for all times
# all_intersection_data = {}
#
# # Initialize storage for panel statistics for all times
# all_panel_statistics = {}

# for timestamp, ply_path in sorted_filtered_ply_dict.items():
#     vertex_data, temperatures = read_ply_data(ply_path)
#     print(f"Processing PLY file for timestamp: {timestamp}")
#     frame_name = ply_path



    # # Initialize storage for intersection points
    # intersection_data = defaultdict(list)
    #
    # # Store the intersection data for this timestamp
    # all_intersection_data[timestamp] = intersection_data
    #
    # # Define panel statistics at a given time step
    # panel_statistics = gather_panel_data(adjusted_vertices, temperatures, panel_zone_coordinates, chamber_center,
    #                                      chamber_height)
    #
    # # Store the panel statistics data for this timestamp
    # all_panel_statistics[timestamp] = panel_statistics

    # # Loop over each point in the PLY data
    # for point, temp in zip(adjusted_vertices, temperatures):
    #     for face_name, face_eq in faces.items():
    #         # print("face_name", face_name)
    #         intersection_point = line_plane_intersection(chamber_center, point - chamber_center, face_eq)
    #         x, y, z = intersection_point
    #         if 0 <= x <= chamber_width and 0 <= y <= chamber_length and 0 <= z <= chamber_height:
    #             vector_from_center_to_point = point - chamber_center
    #             vector_from_center_to_intersection = intersection_point - chamber_center
    #             if np.dot(vector_from_center_to_point, vector_from_center_to_intersection) > 0:
    #                 intersection_data[face_name].append((intersection_point, temp))
    #                 # print(intersection_data)
    #                 break


    # Calculate the average temperature for each face and store it in a dictionary
    # median_surface_temperatures = {face_name: np.median([temp for _, temp in data]) for face_name, data in intersection_data.items()}
    # print("mean_surface_temperatures", median_surface_temperatures)

    # # Plotting
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(projection='3d')
    #
    # # Plot edges of the lab
    # edges = [
    #     [(0, 0, 0), (chamber_width, 0, 0)],
    #     [(chamber_width, 0, 0), (chamber_width, chamber_length, 0)],
    #     [(chamber_width, chamber_length, 0), (0, chamber_length, 0)],
    #     [(0, chamber_length, 0), (0, 0, 0)],
    #     [(0, 0, chamber_height), (chamber_width, 0, chamber_height)],
    #     [(chamber_width, 0, chamber_height), (chamber_width, chamber_length, chamber_height)],
    #     [(chamber_width, chamber_length, chamber_height), (0, chamber_length, chamber_height)],
    #     [(0, chamber_length, chamber_height), (0, 0, chamber_height)],
    #     [(0, 0, 0), (0, 0, chamber_height)],
    #     [(chamber_width, 0, 0), (chamber_width, 0, chamber_height)],
    #     [(chamber_width, chamber_length, 0), (chamber_width, chamber_length, chamber_height)],
    #     [(0, chamber_length, 0), (0, chamber_length, chamber_height)],
    # ]
    #
    # for edge in edges:
    #     ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], c='k')
    #
    # # Plot the intersection points
    # scatters = []
    # for face_name, data in intersection_data.items():
    #     points, temps = zip(*data)
    #     points = np.array(points)
    #     sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=temps, cmap='inferno', marker='o', s=5, label=f"{face_name} (n={len(data)})")
    #     scatters.append(sc)
    #
    # # Add color bar
    # if scatters:
    #     cbar = plt.colorbar(scatters[-1], label='Temperature')
    #     cbar.set_label('Temperature', rotation=270, labelpad=15)
    #
    # # Set axis labels and title
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # plt.title('Temperature Projection onto Lab Faces')
    # # plt.show()


# def update(frame):
#     # Get the list of timestamps
#     timestamps = list(all_intersection_data.keys())
#
#     # Pick the timestamp corresponding to the current frame
#     timestamp = timestamps[frame]
#
#     # Fetch the intersection data and panel statistics for this timestamp
#     intersection_data = all_intersection_data[timestamp]
#     panel_statistics = all_panel_statistics[timestamp]  # Newly Added
#
#     # Define median_surface_temperatures arbitrarily (needs to be computed properly in your actual code)
#     median_surface_temperatures = {}
#
#     # Call the plotting function to create each frame of the animation
#     adapted_plot_with_same_color_scale(
#         intersection_data,
#         median_surface_temperatures,
#         panel_statistics,
#         (chamber_width, chamber_length, chamber_height),
#         15, 30,
#         frame_name=str(timestamp)
#     )
#     plt.clf()  # Clear the current figure (very important for animation)


def main():
    all_intersection_data = {}
    all_panel_statistics = {}
    all_chamber_surface_temperature_statistics = {}
    # Define lab dimensions and center
    chamber_width, chamber_length, chamber_height = 5500, 5500, 2500
    chamber_center = np.array([chamber_width / 2, chamber_length / 2, chamber_height / 2])

    # Define face equations
    faces = {
        "Ceiling": [0, 0, -1, (0, 0, chamber_height)],
        "Floor": [0, 0, 1, (chamber_center[0], chamber_center[1], 0)],
        "Front wall": [0, -1, 0, (chamber_center[0], chamber_length, chamber_center[2])],
        "Back wall": [0, 1, 0, (chamber_center[0], 0, chamber_center[2])],
        "Left wall": [-1, 0, 0, (chamber_width, chamber_center[1], chamber_center[2])],
        "Right wall": [1, 0, 0, (0, chamber_center[1], chamber_center[2])],
    }
    # 関数を呼び出してPLYファイルの辞書を取得
    sorted_filtered_ply_dict = get_sorted_filtered_ply_dict(directory_path, start_time_str, end_time_str)

    # Initialize storage for intersection points
    intersection_data = defaultdict(list)

    # # Store the intersection data for this timestamp
    # all_intersection_data[timestamp] = intersection_data
    #
    # # Define panel statistics at a given time step
    # panel_statistics = gather_panel_data(adjusted_vertices, temperatures, panel_zone_coordinates, chamber_center,
    #                                      chamber_height)
    #
    # # Store the panel statistics data for this timestamp
    # all_panel_statistics[timestamp] = panel_statistics
    for timestamp, ply_path in sorted_filtered_ply_dict.items():
        vertex_data, temperatures = read_ply_data(ply_path)
        adjusted_vertices = vertex_data + chamber_center
        intersection_data = calculate_intersection_data(adjusted_vertices, temperatures, faces, chamber_center)
        panel_statistics = gather_panel_data(adjusted_vertices, temperatures, panel_zone_coordinates, chamber_center,
                                             chamber_height)
        median_surface_temperatures = {face_name: np.median([temp for _, temp in data]) for face_name, data in
                                       intersection_data.items()}

        all_intersection_data[timestamp] = intersection_data
        all_panel_statistics[timestamp] = panel_statistics
        all_chamber_surface_temperature_statistics[timestamp] = median_surface_temperatures

        # Calculate the average temperature for each face and store it in a dictionary

        # print("mean_surface_temperatures", median_surface_temperatures)
    print(all_panel_statistics)
    print(all_chamber_surface_temperature_statistics)
    adapted_plot_with_same_color_scale(
        intersection_data,
        median_surface_temperatures,
        panel_statistics,
        (chamber_width, chamber_length, chamber_height),
        15, 30,
        frame_name=str(timestamp)
    )
    # def update(frame):
    #     timestamps = list(all_intersection_data.keys())
    #     timestamp = timestamps[frame]
    #     intersection_data = all_intersection_data[timestamp]
    #     panel_statistics = all_panel_statistics[timestamp]
    #     # median_surface_temperatures =
    #
    #     adapted_plot_with_same_color_scale(
    #         intersection_data,
    #         median_surface_temperatures,
    #         panel_statistics,
    #         (chamber_width, chamber_length, chamber_height),
    #         15, 30,
    #         frame_name=str(timestamp)
    #     )
    #     plt.clf()
    #
    # # アニメーション作成
    # fig, ax = plt.subplots()
    # ani = FuncAnimation(fig, update, frames=len(all_intersection_data))
    # ani.save('temperature_projection_animation.gif', writer='pillow')
    # plt.show()


if __name__ == "__main__":
    main()


# # To display the animation
# plt.show()


# To save the animation
# ani.save('sine_wave_animation.mp4', writer='ffmpeg')

# # CSV file path
# csv_file_path = os.path.join(directory_path, "sorted_filtered_ply_paths.csv")
#
# # Write to CSV file
# with open(csv_file_path, mode='w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#
#     # Write header
#     csv_writer.writerow(['Time', 'Path'])
#
#     # Write rows
#     for time, path in sorted_filtered_ply_dict.items():
#         csv_writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), path])
#
# print(csv_file_path)