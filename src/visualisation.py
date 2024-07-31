import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_irradiance(ground_grid_x, ground_grid_y, irradiance):

    plt.figure(figsize=(10, 5))
    contour = plt.contourf(ground_grid_x, ground_grid_y, irradiance, levels=100, cmap='hot')
    plt.colorbar(contour, label='Irradiance')
    plt.xlabel('Length (m)')
    plt.ylabel('Width (m)')
    plt.title('Irradiance Pattern on the Ground Surface of the Poly Tunnel')
    plt.savefig("irradiance-tunnel-surface.png")
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
    plt.savefig("solar-positions.png")
    plt.show()