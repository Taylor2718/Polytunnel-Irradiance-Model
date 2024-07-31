import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def plot_irradiance(ground_grid_x, ground_grid_y, irradiance):

    plt.figure(figsize=(10, 5))
    contour = plt.contourf(ground_grid_x, ground_grid_y, irradiance, levels=100, cmap='hot')
    plt.colorbar(contour, label='Irradiance')
    plt.xlabel('Length (m)')
    plt.ylabel('Width (m)')
    plt.title('Irradiance Pattern on the Ground Surface of the Poly Tunnel')
    plt.savefig("figures/irradiance-tunnel-surface.png")
    plt.show()

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

def animate_irradiance(time_array, ground_grid_x, ground_grid_y, irradiance_array):

        # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Initialize the contour plot
    contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array[0], levels=100, cmap='hot')
    cbar = fig.colorbar(contour, ax=ax, label='Irradiance')

    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    ax.set_title('Irradiance Pattern on the Ground Surface of the Poly Tunnel')

    def update(frame):
        ax.clear()
        contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array[frame], levels=100, cmap='hot')
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.set_title(f'Irradiance Pattern on the Ground Surface of the Poly Tunnel - Time {round_10_min(time_array[frame])}')
        return contour,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(irradiance_array), interval=50, blit=True)

    # Save the animation
    anim.save('figures/irradiance-tunnel-surface-animation.mp4', writer='ffmpeg')

    plt.show()