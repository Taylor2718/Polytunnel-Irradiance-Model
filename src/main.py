# main.py
import numpy as np
from geometry import Polytunnel
from sun import Sun
from ray_tracing import RayTracer
import visualisation as viz
from PIL import Image


def main(start_time_str='2024-07-30T00:00:00Z', end_time_str='2024-07-30T23:59:59Z', latitude=51.1950, longitude=0.2757, res_minutes = 1):
    # Create Polytunnel instance
    tunnel = Polytunnel(radius=5, length=20, resolution=0.1)
    ground_grid_x, ground_grid_y = tunnel.generate_ground_grid()

    # Create Sun instance and generate sun positions
    sun = Sun(start_time=start_time_str, end_time=end_time_str, latitude=latitude, longitude=longitude, resolution_minutes=res_minutes)
    altitude_array, azimuth_array = sun.generate_sun_positions()
    time_array = sun.get_times()

    # Perform ray tracing to calculate irradiance on the ground
    ray_tracer = RayTracer(tunnel)
    sun_positions = list(zip(altitude_array, azimuth_array))
    irradiance = ray_tracer.trace_rays(ground_grid_x, ground_grid_y, sun_positions)

    irradiance_frames = []
    for position in sun_positions:
        irradiance = ray_tracer.trace_rays(ground_grid_x, ground_grid_y, [position])
        irradiance_frames.append(irradiance)

    print("Irradiance frames generated:", len(irradiance_frames))

    # Check if frames are generated
    if not irradiance_frames:
        print("No irradiance frames generated")
        return

    # Visualize the sun positions
    viz.plot_sun_positions(time_array, altitude_array, azimuth_array)

    viz.plot_irradiance(ground_grid_x, ground_grid_y, irradiance_frames[300])


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No command-line arguments, use default values
        main()
    elif len(sys.argv) == 6:
        # Command-line arguments provided
        start_time_str = sys.argv[1]
        end_time_str = sys.argv[2]
        latitude = float(sys.argv[3])
        longitude = float(sys.argv[4])
        res_minutes = float(sys.argv[5])

        main(start_time_str, end_time_str, latitude, longitude, res_minutes)
    else:
        print("Usage: python main.py [<start_time> <end_time> <latitude> <longitude>]")
        print("Default values will be used if no arguments are provided.")
        sys.exit(1)
