# main.py
import numpy as np
from geometry import Polytunnel
from sun import Sun
from ray_tracing import RayTracer
import visualisation as viz


def main(start_time_str='2024-07-30T00:00:00Z', end_time_str='2024-07-30T23:59:59Z', latitude=51.1950, longitude=0.2757, res_minutes = 1, n_points = 1000, length = 20, radius = 5):
    
    # Create Polytunnel instance
    tunnel = Polytunnel(radius=radius, length=length, n_points=n_points)
    ground_grid_x = tunnel.generate_surface()[0]
    ground_grid_y = tunnel.generate_surface()[1]
    ground_grid_z = tunnel.generate_surface()[2]

    # Create Sun instance and generate sun positions
    sun = Sun(start_time=start_time_str, end_time=end_time_str, latitude=latitude, longitude=longitude, resolution_minutes=res_minutes)
    altitude_array, azimuth_array = sun.generate_sun_positions()
    distance_array = sun.generate_sun_distance()
    ground_grid = tunnel.generate_ground_grid()
    surface_grid = tunnel.generate_surface()
    distance_grid = tunnel.generate_distances_grid(ground_grid, surface_grid)
    vecs = tunnel.vec_grid(surface_grid)
    time_array = sun.get_times()
    # Perform ray tracing to calculate irradiance on the ground
    ray_tracer = RayTracer(tunnel)
    sun_positions = list(zip(altitude_array, azimuth_array))
    sun_vecs = sun.generate_sun_vecs(sun_positions)
    normals_unit = tunnel.surface_element_unit_vectors()
    tilts_unit = tunnel.surface_tilt(normals_unit)
    trans_sun_vecs = sun.generate_sun_vec_grids(sun_vecs, surface_grid, distance_array)

    spectra = sun.get_spectra(tilts_unit[0][0], 400, 700)
    irradiance_frames = ray_tracer.irradiance_rays(normals_unit, sun_positions, sun_vecs, spectra[1])
    print(f"Number of frames: {len(irradiance_frames)}")
    
    viz.plot_spectra(time_array, spectra)
    
    #viz.plot_sun_positions(time_array, altitude_array, azimuth_array)

    #viz.plot_surface(ground_grid_x, ground_grid_y, ground_grid_z, normals_unit, sun_vecs[-1])

    viz.plot_irradiance(ground_grid_x, ground_grid_y, irradiance_frames[-1])
    
    viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, irradiance_frames)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No command-line arguments, use default values
        main()
    elif len(sys.argv) == 9:
        # Command-line arguments provided
        start_time_str = sys.argv[1]
        end_time_str = sys.argv[2]
        latitude = float(sys.argv[3])
        longitude = float(sys.argv[4])
        res_minutes = float(sys.argv[5])
        n_points = float(sys.argv[6])
        length = float(sys.argv[7])
        radius = float(sys.argv[8])

        main(start_time_str, end_time_str, latitude, longitude, res_minutes, n_points, length, radius)
    else:
        print("Usage: python main.py [<start_time> <end_time> <latitude> <longitude>]")
        print("Default values will be used if no arguments are provided.")
        sys.exit(1)
