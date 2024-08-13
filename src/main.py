# main.py
import numpy as np
from geometry import Polytunnel
from sun import Sun
from ray_tracing import RayTracer
import visualisation as viz


def main(start_time_str='2024-07-30T00:00:00Z', end_time_str='2024-07-30T23:59:59Z', latitude=51.1950, longitude=0.2757, res_minutes = 1, n_points = 1000, length = 20, radius = 5, xy_angle = 0, z_angle = 0, transmissivity = 1):
    
    tunnel = Polytunnel(radius=radius, length=length, n_points=n_points, xy_angle=xy_angle, z_angle=z_angle)
    ground_grid = tunnel.generate_ground_grid()

    sun = Sun(start_time=start_time_str, end_time=end_time_str, latitude=latitude, longitude=longitude, resolution_minutes=res_minutes)
    altitude_array, azimuth_array = sun.generate_sun_positions()
    ground_grid_x = tunnel.generate_ground_grid()[0]
    ground_grid_y = tunnel.generate_ground_grid()[1]
    ground_grid_z = tunnel.generate_ground_grid()[2]

    surface_grid_x = tunnel.generate_surface()[0]
    surface_grid_y = tunnel.generate_surface()[1]
    surface_grid_z = tunnel.generate_surface()[2]


    normals_unit_surface, areas_surface = tunnel.surface_element_unit_vectors()
    normals_unit_ground, areas_ground = tunnel.ground_element_unit_vectors()
    tilts_unit = tunnel.surface_tilt(normals_unit_surface)

    surface_grid = tunnel.generate_surface()
    distance_grid, separation_unit_vector_grid = tunnel.generate_distances_grid(ground_grid, surface_grid)
    time_array = sun.get_times()

    ray_tracer = RayTracer(tunnel)
    sun_positions = list(zip(altitude_array, azimuth_array))
    sun_vecs = sun.generate_sun_vecs(sun_positions)

    spectra_frames = sun.get_spectra(tilts_unit[0][0], 400, 700)
    irradiance_frames = ray_tracer.irradiance_rays(normals_unit_surface, sun_positions, sun_vecs, spectra_frames)
    
    diffuse_irradiance_frames = ray_tracer.diffuse_irradiance_ground(distance_grid, separation_unit_vector_grid, normals_unit_ground, normals_unit_surface, areas_surface, irradiance_frames, transmissivity)
    print(f"Number of frames: {len(irradiance_frames)}")

    ground_projection = np.dot(normals_unit_ground[:, 0, 0], separation_unit_vector_grid[0][0][5][5])
    surface_projection = np.dot(normals_unit_surface[:, 5, 5], separation_unit_vector_grid[0][0][5][5])

    print(normals_unit_surface[:, 5, 5])
    print(normals_unit_ground[:, 0, 0])
    print(ground_projection)
    print(surface_projection)
    print(separation_unit_vector_grid[0][0][5][5])

        
    viz.plot_sun(time_array, altitude_array, azimuth_array, spectra_frames, "figures/sun.png")

    viz.plot_surface(surface_grid_x, surface_grid_y, surface_grid_z, normals_unit_surface, ground_grid_x, ground_grid_y, ground_grid_z, normals_unit_ground, sun_vecs[-1])

    #viz.plot_irradiance(surface_grid_x, surface_grid_y, irradiance_frames[60])
    
    #viz.animate_irradiance(time_array, surface_grid_x, surface_grid_y, irradiance_frames, "figures/direct-irradiance-surface-animation.mp4")
    
    #viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, diffuse_irradiance_frames, "figures/diffuse-irradiance-ground-animation.mp4")

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
