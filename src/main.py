# main.py
import numpy as np
import matplotlib.pyplot as plt
from geometry import Polytunnel
from sun import Sun
from irradiance import TunnelIrradiance
import visualisation as viz
from tracing import Tracing
import tmm as tmm
import tmm_fast as vectmm
import torch

__all__ = ("compute_surface_grid",)

#coh = tmm.coh_tmm()

def compute_surface_grid():
    """Returns the surface grid"""



def main(start_time_str='2024-07-30T00:00:00Z', end_time_str='2024-07-30T23:59:59Z', latitude=51.1950, longitude=0.2757, res_minutes = 1, n_points = 1000, length = 20, radius = 5, xy_angle = 0, z_angle = 0, transmissivity = 1):
    
    tunnel = Polytunnel(radius=radius, length=length, n_points=n_points, xy_angle=xy_angle, z_angle=z_angle, theta_margin = 3, cell_thickness=1, cell_gap=2)
    tunnel_l = Polytunnel(radius=radius, length=length, n_points=n_points, xy_angle=xy_angle, z_angle=z_angle, x_shift = 3.0)
    tunnel_r = Polytunnel(radius=radius, length=length, n_points=n_points, xy_angle=xy_angle, z_angle=z_angle, x_shift = -3.0)

    ground_grid = tunnel.generate_ground_grid()
    ground_grid_x, ground_grid_y = (ground_grid[0], ground_grid[1])

    sun = Sun(start_time=start_time_str, end_time=end_time_str, latitude=latitude, longitude=longitude, resolution_minutes=res_minutes)
    altitude_array, azimuth_array = sun.generate_sun_positions()

    normals_unit_surface, areas_surface = tunnel.surface_element_unit_vectors()
    normals_unit_ground, areas_ground = tunnel.ground_element_unit_vectors()
    tilts_unit = tunnel.surface_tilt(normals_unit_surface)

    surface_grid, solar_cells = tunnel.generate_surface()
    surface_grid_x, surface_grid_y = (surface_grid[0], surface_grid[1])
    print(len(surface_grid_x))

    #surface_grid_l = tunnel_l.generate_surface()
    #surface_grid_r = tunnel_r.generate_surface()
    d = 2*radius
    
    distance_grid, separation_unit_vector_grid = tunnel.generate_distances_grid(ground_grid, surface_grid)
    time_array = sun.get_times()

    irradiance = TunnelIrradiance(tunnel, radius, length)
    sun_positions = list(zip(altitude_array, azimuth_array))
    sun_vecs = sun.generate_sun_vecs(sun_positions)
    sun_incident = sun.sunvec_tilts_grid(sun_vecs, tilts_unit)

    tracer = Tracing(tunnel, sun_vecs, surface_grid, tilts_unit, d, radius)
    gradient_grid, angle_grid, surface_gradient_grid, surface_angle_grid = tracer.find_tangent_gradient()
    solid_angle_grid = tracer.solid_angle_grid(angle_grid, surface_angle_grid)

    exposure_maps = irradiance.shading_exposure_map(angle_grid, surface_angle_grid, sun_vecs)
    optical_wavelengths, optical_intensities, spectra_frames = sun.get_spectra(300, 700) #zero tilt

    material_list = ['ag', 'moo3', 'zno', 'moo3', 'zno', 'ito']
    d_list = np.array(['inf', 30, 7, 35, 15, 25, 110, 'inf'])
    complex_array = tracer.n_list_wavelength(material_list, optical_wavelengths)
    #t_grid_frames = irradiance.t_grid(sun_incident, optical_wavelengths, complex_array, d_list)
    #solar_cell_spectra = irradiance.solar_cells_irradiance_rays(optical_intensities, t_grid_frames, solar_cells)
    #solar_cell_irradiance_frames = irradiance.int_spectra(optical_wavelengths, solar_cell_spectra)
    #print(time_array[30])
    #print(solar_cell_irradiance_frames[30])
    irradiance_frames = irradiance.irradiance_rays(normals_unit_surface, sun_positions, sun_vecs, spectra_frames)
    #shaded_irradiance_frames = irradiance.shaded_irradiance_rays(irradiance_frames, exposure_maps)
    
    #print(solar_cells)
    #print(time_array[54])
    #print(coh)
    #print(coh['T'])

    optical_wavelengths = np.linspace(300, 700, 1000)

    complex_array = tracer.n_list_wavelength(material_list, optical_wavelengths)
    print(complex_array[10])

    complex_array = np.array(complex_array)  # Refractive index array
    d_list = np.array(d_list)  # Thickness array
    d_list = d_list.astype(np.float64)  # Convert from string to float
    sun_incident = np.array(sun_incident[72][:, 0])  # Ensure sun_incident is a numpy array
    optical_wavelengths = np.array(optical_wavelengths) 

        # Assuming complex_array is your refractive index array (N) and d_list is the thickness array (T)
    N = np.array(complex_array, dtype=np.complex128)  # Convert N to numpy array
    T = np.array(d_list, dtype=np.float64)  # Convert T to numpy array

    # Swap the axes of N to have shape [8, 38] instead of [38, 8]
    N = np.swapaxes(N, 0, 1)  # Swap axes so that layers come first (N shape should be [8, 38])

    # Incident angles and wavelengths as numpy arrays
    Theta = np.linspace(-np.pi/2, np.pi/2, 1000)  # Incident angles (Theta)
    optical_wavelengths = np.array(optical_wavelengths, dtype=np.float64)  # Wavelengths

        # Check the shapes and data types after conversion
    print("N shape:", N.shape)  # Should be (8, 38)
    print("T shape:", T.shape)  # Should be (8)
    print("Theta shape:", Theta.shape)  # Check the shape of incident angles array
    print("optical_wavelengths shape:", optical_wavelengths.shape)  # Check the shape of wavelength array

    # If you were previously using torch-based functions, replace them with numpy equivalents or modify the function you're using accordingly.
    O = vectmm.coh_tmm('s', N, T, Theta, optical_wavelengths)  # Assuming vectmm.coh_tmm works with numpy arrays
    t = O['T']  # Transmission powers
    t_amp = np.sqrt(t)

    # Output the shape of the result
    print(t.shape)
    print(np.max(t))
    print(np.max(t_amp))

    plt.figure(figsize=(8, 6))

    # Create a heatmap using pcolormesh
    plt.pcolormesh(optical_wavelengths, Theta, t_amp, cmap='viridis', shading='auto', vmin=0, vmax=np.max(t_amp))

    # Add a color bar to show the transmission amplitude scale (0 to 1)
    cbar = plt.colorbar()
    cbar.set_label('Transmission Amplitude')

    # Label the axes
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Tilt Angle (Radians)')

    # Set a title for the plot
    plt.title('TMM of Transmission Amplitudes for Tilt Angles and Wavelengths')

    plt.savefig('figures/tmm-transmission.png', dpi=300)

    # Display the plot
    plt.show()
    
    #diffuse_irradiance_frames = irradiance.diffuse_irradiance_ground(distance_grid, separation_unit_vector_grid, normals_unit_ground, normals_unit_surface, areas_surface, irradiance_frames, transmissivity)
    #direct_irradiance_frames = irradiance.direct_irradiance_ground(normals_unit_ground, sun_vecs, spectra_frames, transmissivity)
    #direct_irradiance_frames = irradiance.ray_trace_to_surface(ground_grid, normals_unit_ground, surface_grid, distance_grid, sun_vecs, irradiance_frames, transmissivity)
    #global_irradiance_frames = irradiance.global_irradiance_ground(direct_irradiance_frames, diffuse_irradiance_frames)

    #power_total_ground_diffuse = irradiance.power(areas_ground, diffuse_irradiance_frames)
    #power_total_surface = irradiance.power(areas_surface, irradiance_frames)
    #power_total_ground_direct = irradiance.power(areas_ground, direct_irradiance_frames)
    #power_total_out = np.array(power_total_ground_diffuse) + np.array(power_total_ground_direct)

    #viz.plot_sun(time_array, altitude_array, azimuth_array, spectra_frames, "figures/sun.png")

    #viz.plot_surface(surface_grid_x, surface_grid_y, surface_grid_z, normals_unit_surface, ground_grid_x, ground_grid_y, ground_grid_z, normals_unit_ground, sun_vec=sun_vecs[72])

    #viz.plot_irradiance(surface_grid_x, surface_grid_y, irradiance_frames[60])
    
    #viz.animate_irradiance(time_array, surface_grid_x, surface_grid_y, direct_irradiance_frames, "figures/direct-irradiance-surface-animation.mp4")
    
    #viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, diffuse_irradiance_frames, "figures/diffuse-irradiance-ground-animation.mp4")

    #viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, direct_irradiance_frames, "figures/direct-irradiance-ground-animation.mp4")

    #viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, global_irradiance_frames, "figures/global-irradiance-ground-animation.mp4")

    #viz.plot_power(time_array, power_total_ground_diffuse, power_total_ground_direct, power_total_out, 'figures/power-received.png')
    
    #viz.plot_coverage(surface_grid, visible_solid_angle)
    # Compute visible solid angles

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No command-line arguments, use default values
        main()
    elif len(sys.argv) == 12:
        # Command-line arguments provided
        start_time_str = sys.argv[1]
        end_time_str = sys.argv[2]
        latitude = float(sys.argv[3])
        longitude = float(sys.argv[4])
        res_minutes = float(sys.argv[5])
        n_points = float(sys.argv[6])
        length = float(sys.argv[7])
        radius = float(sys.argv[8])
        xy_angle = float(sys.argv[9])
        z_angle = float(sys.argv[10])
        transmissivity = float(sys.argv[11])

        main(start_time_str, end_time_str, latitude, longitude, res_minutes, n_points, length, radius, xy_angle, z_angle, transmissivity)
    else:
        print("Usage: python main.py [<start_time> <end_time> <latitude> <longitude>]")
        print("Default values will be used if no arguments are provided.")
        sys.exit(1)
