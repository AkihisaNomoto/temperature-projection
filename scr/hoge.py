# Adapting the original plotting function to use the same color scale for both the point cloud and the walls
def adapted_plot_with_same_color_scale(intersection_data, median_surface_temperatures, panel_statistics, chamber_dimensions, min_temp, max_temp, show_temp_point=False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define walls with their corners (based on lab dimensions)
    w, l, h = chamber_dimensions
    walls = {
        "ceiling": [(0, 0, h), (w, 0, h), (w, l, h), (0, l, h)],
        "floor": [(0, 0, 0), (w, 0, 0), (w, l, 0), (0, l, 0)],
        "front_wall": [(0, l, 0), (w, l, 0), (w, l, h), (0, l, h)],
        "back_wall": [(0, 0, 0), (w, 0, 0), (w, 0, h), (0, 0, h)],
        "left_wall": [(w, 0, 0), (w, l, 0), (w, l, h), (w, 0, h)],
        "right_wall": [(0, 0, 0), (0, l, 0), (0, l, h), (0, 0, h)],
    }

    # Define the panel zones with their corners (based on given coordinates and lab height)
    panel_zones = {
        "Zone1": [(1835, 0, h), (3665, 0, h), (3665, 1220, h), (1835, 1220, h)],
        "Zone2": [(610, 1220, h), (2445, 1220, h), (2445, 3660, h), (610, 3660, h)],
        "Zone3": [(1835, 3660, h), (3665, 3660, h), (3665, 4880, h), (1835, 4880, h)],
        "Zone4": [(3055, 1220, h), (4885, 1220, h), (4885, 3660, h), (3055, 3660, h)],
    }

    # Use the same colormap for both the point cloud and the walls
    # (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    cmap = cm.bwr
    cmap = cm.seismic
    # cmap = cm.coolwarm

    # Normalization function for the color map
    norm = Normalize(vmin=min_temp, vmax=max_temp)

    # Function to plot a face based on its corners and color
    def plot_face(ax, corners, color, alpha=1.0):
        """
        Plots a single face on a given matplotlib 3D axis (ax)

        Parameters:
            ax : matplotlib 3D axis
            corners : list of corner coordinates for the face
            color : face color
            alpha : transparency
        """
        poly3d = [[list(corner) for corner in corners]]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))

    # Plot each panel with color based on its average temperature
    for panel, stats in panel_statistics.items():
        color = cmap((stats['average'] - min_temp) / (max_temp - min_temp))
        # panel_name = f'Radiant Panel {panel}'
        plot_face(ax, panel_zones[panel], color, alpha=0.7)

    # Plot each wall with color based on its average temperature
    for wall, corners in walls.items():
        color = cmap((median_surface_temperatures[wall] - min_temp) / (max_temp - min_temp))
        poly3d = [[list(corner) for corner in corners]]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='k', alpha=0.5))

    # Plot the intersection points using the same color scale
    if show_temp_point:
        for face_name, data in intersection_data.items():
            points, temps = zip(*data)
            points = np.array(points)

            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=temps, cmap=cmap, vmin=min_temp, vmax=max_temp,
                       marker='o', s=5, alpha=0.3)

    # Add color bar using the same color scale
    sc = ax.scatter([], [], [], c=[], cmap=cmap, vmin=min_temp, vmax=max_temp)
    cbar = plt.colorbar(sc, label='Temperature (°C)')
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

    # Add median temperatures for panels, floor, walls, and ceiling as text beside the plot
    text_position = [-0.4, 0.5]
    legend_text = 'Median Temperatures (°C)\n'

    # Panels
    for zone, stats in panel_statistics.items():
        median_temp_round = round(stats.get('median'), 2)
        legend_text += f"{zone}: {median_temp_round}\n"

    # Floor, Walls, and Ceiling
    for surface, temp in median_surface_temperatures.items():
        legend_text += f"{surface.capitalize()}: {round(temp, 2)}\n"

    # Add the legend-like box
    ax.text2D(text_position[0], text_position[1], legend_text.rstrip(), transform=ax.transAxes, fontsize=10,)
              # bbox=dict(facecolor='white', edgecolor='black'))

    # Set axis labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    # Set axis limits
    ax.set_xlim([0, chamber_dimensions[0]])
    ax.set_ylim([0, chamber_dimensions[1]])
    ax.set_zlim([0, chamber_dimensions[2]])
    ax.set_title('Temperature Projection on Chamber Surface')

    plt.show()