import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime, timedelta


def round_10_min(dt):
    # Find the number of minutes past the hour
    minutes = dt.minute
    
    # Calculate the rounding direction
    if minutes % 10 >= 5:
        # Round up
        rounded_minute = (minutes // 10 + 1) * 10
    else:
        # Round down
        rounded_minute = (minutes // 10) * 10
    
    # Handle overflow to the next hour
    if rounded_minute == 60:
        dt = dt.replace(minute=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=rounded_minute)
    
    # Zero out seconds and microseconds
    dt = dt.replace(second=0, microsecond=0)
    
    return dt

def plot_sun(time_array, altitude_array, azimuth_array, irradiance_frames, filename):

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_array, altitude_array, label='Altitude', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Altitude (degrees)')
    plt.title('Solar Altitude Over Time')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time_array, azimuth_array, label='Azimuth', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Azimuth (degrees)')
    plt.title('Solar Azimuth Over Time')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_array, irradiance_frames, label='Irradiance', color='red')
    plt.xlabel('Time')
    plt.ylabel(r"Irradiance $\left(\frac{W}{m^{2}}\right)$")
    plt.title('Solar Irradiance Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def set_axes_equal(ax):
    ''' 
    Set the 3D plot axes to have equal scale so that spheres appear as spheres,
    and cubes as cubes, etc. This is one possible solution to Matplotlib's lack
    of a set_aspect method for 3D plots.
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = np.array([x_range, y_range, z_range]).max()

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range / 2, mid_x + max_range / 2])
    ax.set_ylim3d([mid_y - max_range / 2, mid_y + max_range / 2])
    ax.set_zlim3d([mid_z - max_range / 2, mid_z + max_range / 2])

def plot_surface(X, Y, Z, normals_unit, X_ground, Y_ground, Z_ground, normals_unit_ground, sun_vec):
    # Normal vectors at the center points
    U = normals_unit[0]
    V = normals_unit[1]
    W = normals_unit[2]
    U_ground = normals_unit_ground[0]
    V_ground = normals_unit_ground[1]
    W_ground = normals_unit_ground[2]
    sun_dir_x = sun_vec[0]
    sun_dir_y = sun_vec[1]
    sun_dir_z = sun_vec[2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surfaces with custom colors
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.4)  # Polytunnel surface
    ax.plot_surface(X_ground, Y_ground, Z_ground, color='brown', alpha=0.6)  # Ground surface

    # Plot normal vectors
    ax.quiver(X_ground, Y_ground, Z_ground, U_ground, V_ground, W_ground, length=1, color='green')
    #ax.quiver(X, Y, Z, U, V, W, length=1, color='black')

    # Plot sun rays
    #ax.quiver(X, Y, Z, sun_dir_x, sun_dir_y, sun_dir_z, color='red')

    # Remove grid and axes
    ax.grid(False)
    ax.axis('off')

    # Set equal aspect ratio for the axes
    set_axes_equal(ax)

    plt.show()

def plot_irradiance(ground_grid_x, ground_grid_y, irradiance_array):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Initialize the contour plot
    contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array, levels=100, cmap='hot')
    cbar = fig.colorbar(contour, ax=ax, label='Irradiance')

    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_title('Irradiance Pattern on the Polytunnel surface')
    plt.show()

def animate_irradiance(time_array, ground_grid_x, ground_grid_y, irradiance_array, filename):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

   # Find the global min and max of the irradiance array
    irradiance_min = 0
    irradiance_max = np.max(irradiance_array)
    list_index = next(i for i, sublist in enumerate(irradiance_array) if irradiance_max in sublist)

    # Initialize the contour plot with the first frame, using the global min and max for color limits
    contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array[list_index], levels=100, cmap='hot', vmin=irradiance_min, vmax=irradiance_max)
    cbar = fig.colorbar(contour, ax=ax, label='Irradiance')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_title('Irradiance')

    def update(frame):
        ax.clear()
        contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array[frame], levels=100, cmap='hot', vmin=irradiance_min, vmax=irradiance_max)
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Length (m)')
        ax.set_title(f'Irradiance Grid - Time {round_10_min(time_array[frame])}')
        return contour,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(irradiance_array), interval=50, blit=True)

    # Save the animation
    anim.save(filename, writer='ffmpeg')

    plt.show()

def plot_power(time_array, power_surface, power_ground_diffuse, power_ground_direct, power_ground, filename):
   
    plt.figure(figsize=(10, 7))
    plt.plot(time_array, power_surface, label = 'Surface (Global)')
    plt.plot(time_array, power_ground_diffuse, label = 'Ground (Diffuse)')
    plt.plot(time_array, power_ground_direct, label = 'Ground (Direct)')
    plt.plot(time_array, power_ground, label = 'Ground (Global)')

    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.savefig(filename)
    plt.show()

