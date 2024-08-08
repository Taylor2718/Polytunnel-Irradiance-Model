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

def plot_sun_positions(time_array, altitude_array, azimuth_array):

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time_array, altitude_array, label='Altitude', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Altitude (degrees)')
    plt.title('Solar Altitude Over Time')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_array, azimuth_array, label='Azimuth', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Azimuth (degrees)')
    plt.title('Solar Azimuth Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("figures/solar-positions.png")
    plt.show()

def plot_surface(X, Y, Z, normals_unit, sun_vec):
        # Normal vectors at the center points
    U = normals_unit[0]
    V = normals_unit[1]
    W = normals_unit[2]
    sun_dir_x = sun_vec[0]
    sun_dir_y = sun_vec[1]
    sun_dir_z = sun_vec[2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    ax.quiver(X, Y, Z, U, V, W, length=1, color='r')
    ax.quiver(X, Y, Z, 
          sun_dir_x, sun_dir_y, sun_dir_z, 
          color='b', label='Sun Rays')
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

def animate_irradiance(time_array, ground_grid_x, ground_grid_y, irradiance_array):
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
    ax.set_title('Irradiance Pattern on the Polytunnel surface')

    def update(frame):
        ax.clear()
        contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array[frame], levels=100, cmap='hot', vmin=irradiance_min, vmax=irradiance_max)
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Length (m)')
        ax.set_title(f'Irradiance Pattern on the Ground Surface of the Poly Tunnel - Time {round_10_min(time_array[frame])}')
        return contour,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(irradiance_array), interval=50, blit=True)

    # Save the animation
    anim.save('figures/irradiance-tunnel-surface-animation.mp4', writer='ffmpeg')

    plt.show()

def plot_spectra(time_array, spectra):
   
    plt.figure()

    for i in range(len(spectra)):
        plt.plot((spectra[0][i])['wavelength'], (spectra[0][i])['poa_global'])
    
    plt.show()

    plt.figure()
    plt.plot(time_array, spectra[1])
    plt.show()