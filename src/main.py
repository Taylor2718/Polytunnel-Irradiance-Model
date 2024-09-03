# main.py
import numpy as np
from geometry import Polytunnel
from sun import Sun
from irradiance import TunnelIrradiance
import visualisation as viz
from tracing import Tracing

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

    #surface_grid_l = tunnel_l.generate_surface()
    #surface_grid_r = tunnel_r.generate_surface()
    d = 2*radius
    

    distance_grid, separation_unit_vector_grid = tunnel.generate_distances_grid(ground_grid, surface_grid)
    time_array = sun.get_times()

    print(solar_cells)
    print(surface_grid_x[0,:])

    #irradiance = TunnelIrradiance(tunnel, radius, length)
    #sun_positions = list(zip(altitude_array, azimuth_array))
    #sun_vecs = sun.generate_sun_vecs(sun_positions)
    #tracer = Tracing(tunnel, sun_vecs, surface_grid, tilts_unit, d, radius)
    #gradient_grid, angle_grid, surface_gradient_grid, surface_angle_grid = tracer.find_tangent_gradient()

    #exposure_maps = irradiance.shading_exposure_map(angle_grid, surface_angle_grid, sun_vecs)

    #print(surface_grid[0])
    #print(gradient_grid)
    #print(surface_angle_grid)
    #print(angle_grid)

    #tracer = Tracing(tunnel, sun_vecs, tunnel_l, tunnel_r)
    #exposure_maps = tracer.calculate_light_exposure()

    #i = 118
    #print(len(exposure_maps))
    #print(time_array[i])
    #print(altitude_array[i])
    #print(azimuth_array[i])
    #print(sun_vecs[i])
    #print(exposure_maps[i])

    #spectra_frames = sun.get_spectra(tilts_unit[0][0], 400, 700)
    #irradiance_frames = irradiance.irradiance_rays(normals_unit_surface, sun_positions, sun_vecs, spectra_frames)
    #shaded_irradiance_frames = irradiance.shaded_irradiance_rays(irradiance_frames, exposure_maps)

    #diffuse_irradiance_frames = irradiance.diffuse_irradiance_ground(distance_grid, separation_unit_vector_grid, normals_unit_ground, normals_unit_surface, areas_surface, irradiance_frames, transmissivity)
    #direct_irradiance_frames = irradiance.direct_irradiance_ground(normals_unit_ground, sun_vecs, spectra_frames, transmissivity)
    #direct_irradiance_frames = irradiance.ray_trace_to_surface(ground_grid, normals_unit_ground, surface_grid, distance_grid, sun_vecs, irradiance_frames, transmissivity)
    #global_irradiance_frames = irradiance.global_irradiance_ground(direct_irradiance_frames, diffuse_irradiance_frames)

    #power_total_ground_diffuse = irradiance.power(areas_ground, diffuse_irradiance_frames)
    #power_total_surface = irradiance.power(areas_surface, irradiance_frames)
    #power_total_ground_direct = irradiance.power(areas_ground, direct_irradiance_frames)
    #power_total_out = np.array(power_total_ground_diffuse) + np.array(power_total_ground_direct)

    #shading = Angle(
    #solid_angle = shading.compute_visible_solid_angle(surface_grid, surface_normal, None, None))

    #viz.plot_sun(time_array, altitude_array, azimuth_array, spectra_frames, "figures/sun.png")

    #viz.plot_surface(surface_grid_x, surface_grid_y, surface_grid_z, normals_unit_surface, ground_grid_x, ground_grid_y, ground_grid_z, normals_unit_ground, sun_vec=sun_vecs[72])

    #viz.plot_irradiance(surface_grid_x, surface_grid_y, irradiance_frames[60])
    
    #viz.animate_irradiance(time_array, surface_grid_x, surface_grid_y, shaded_irradiance_frames, "figures/direct-irradiance-surface-animation.mp4")
    
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
