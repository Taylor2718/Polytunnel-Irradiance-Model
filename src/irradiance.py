import numpy as np
import tmm as tmm


class TunnelIrradiance:
    def __init__(self, polytunnel, radius, length):
        """
        Initializes the RayTracer class for calculating solar irradiance on the polytunnel ground.
        
        :param polytunnel: Instance of the Polytunnel class.
        """
        self.polytunnel = polytunnel
        self.radius = radius
        self.length = length

    def irradiance_rays(self, normals_unit, sun_positions, sun_vecs, irradiance_frames):

        irradiance = []

        for i in range(len(sun_positions)):
            sun_vec = sun_vecs[i]
            calc = self.calculate_irradiance(normals_unit, sun_vec, irradiance_frames[i])
            irradiance.append(calc)
        return np.array(irradiance)
    
    def calculate_irradiance(self, normals_unit, sun_vec, spectral_irradiance):
        
        irradiance_surface = np.clip((spectral_irradiance * np.tensordot(normals_unit, sun_vec, axes=(0, 0))), a_min = 0, a_max = None)
        
        return irradiance_surface
    
    def shading_exposure_map(self, angle_grid1, angle_grid2, sun_vecs):
        
        exposure_maps = []
        for i in range(len(sun_vecs)):
            x = sun_vecs[i][0]
            z = sun_vecs[i][2]
            theta = np.degrees(np.arctan2(z,x))
                    # Initialize exposure map with zeros
            exposure_map = np.zeros(angle_grid1.shape, dtype=int)

            # Iterate over each cell in the grids
            for i in range(angle_grid1.shape[0]):
                for j in range(angle_grid1.shape[1]):
                    # Define the interval boundaries
                    theta1 = angle_grid1[i, j]
                    theta2 = angle_grid2[i, j]

                    if theta1 > theta2:
                        #theta1 is obtuse
                        if theta2 <= theta <= theta1:
                            exposure_map[i, j] = 1


                    elif theta2 > theta1:
                        #theta2 is obtuse
                        if theta1 <= theta <= theta2:
                            exposure_map[i, j] = 1
                            
            exposure_maps.append(exposure_map)

        return exposure_maps

    
    def diffuse_irradiance_ground(self, distances_grid, separation_unit_vector_grid, normals_unit_ground, normals_unit_surface, surface_areas, irradiance_frames, transmissivity):

        diffuse_irradiance_frames = []

        ground_shape = (len(distances_grid), len(distances_grid[0]))
        
        for j in range(len(irradiance_frames)):
            irradiance_ground = np.zeros(ground_shape)

            # Iterate over each point on the ground grid
            for p in range(ground_shape[0]):
                for q in range(ground_shape[1]):
                    
                    total_irradiance = 0.0
                    
                    # Iterate over each point on the surface grid
                    surface_shape = irradiance_frames[j].shape
                    for k in range(surface_shape[0]):
                        for l in range(surface_shape[1]):
                            # Sum the irradiance contribution from each surface point
                            distance = distances_grid[p][q][k][l]
                            ground_projection = np.dot(normals_unit_ground[:, p, q], separation_unit_vector_grid[p][q][k][l])
                            surface_projection = np.dot(normals_unit_surface[:, k, l], separation_unit_vector_grid[p][q][k][l])

                            irradiance = (1-transmissivity) * ground_projection * surface_projection * surface_areas[k][l] * irradiance_frames[j][k][l] / (2*np.pi*(distance)**2)
                            total_irradiance += irradiance
                    
                    # Store the result in the irradiance_ground array
                    irradiance_ground[p][q] = total_irradiance
        
            diffuse_irradiance_frames.append(irradiance_ground)

        return diffuse_irradiance_frames

    def direct_irradiance_ground(self, normals_unit_ground, sun_vecs, irradiance_frames, transmissivity):

        direct_irradiance_frames = []
        
        for i in range(len(irradiance_frames)):
            
            sun_vec = sun_vecs[i]
            calc = np.clip((transmissivity * irradiance_frames[i] * np.tensordot(normals_unit_ground, sun_vec, axes=(0, 0))), a_min = 0, a_max = None)
            direct_irradiance_frames.append(calc)

        return direct_irradiance_frames
    
    def ray_trace_to_surface(self, ground_grid, normals_unit_ground, surface_grid, distances_grid, sun_vecs, irradiance_frames, transmissivity):
            
        ground_shape = (len(distances_grid), len(distances_grid[0]))
        irradiance_traced = []

        for i in range(len(irradiance_frames)):
            irradiance_grid = np.zeros((ground_shape[0], ground_shape[1]))

            # Calculate intersections and closest points
            for j in range(ground_shape[0]):
                for k in range(ground_shape[1]):
                    # Get ground point
                    x_g = ground_grid[0][j][k]
                    y_g = ground_grid[1][j][k]
                    z_g = ground_grid[2][j][k]

                    # Normalize sun vector
                    sun_vec = np.array(sun_vecs[i])
                    sun_vec = sun_vec / np.linalg.norm(sun_vec)

                    # Compute t for intersection with the cylindrical surface
                    a = sun_vec[0]**2 + sun_vec[2]**2
                    b = 2 * (x_g * sun_vec[0] + z_g * sun_vec[2])
                    c = x_g**2 + z_g**2 - self.radius**2

                    discriminant = b**2 - 4 * a * c

                    if discriminant >= 0:
                        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
                        
                        # Choose the correct t (positive and smallest)
                        t = min(t1, t2) if min(t1, t2) > 0 else max(t1, t2)
                        
                        # Calculate intersection point on the cylindrical surface
                        x_int = x_g + t * sun_vec[0]
                        y_int = y_g + t * sun_vec[1]
                        z_int = z_g + t * sun_vec[2]

                        # Check if the intersection is within the tunnel bounds
                        if -self.length <= y_int <= self.length:
                            # Find the closest surface grid point
                            distances = np.sqrt((surface_grid[0] - x_int)**2 + 
                                                (surface_grid[1] - y_int)**2 + 
                                                (surface_grid[2] - z_int)**2)
                            closest_j, closest_k = np.unravel_index(np.argmin(distances), distances.shape)
                            irradiance_point = irradiance_frames[i][closest_j][closest_k]
                        else:
                            irradiance_point = 0

                    else:
                        irradiance_point = 0

                    # Apply transmissivity and set the irradiance in the grid
                    irradiance_grid[j][k] = transmissivity * irradiance_point * np.dot(sun_vecs[i], normals_unit_ground[:, j, k])

            irradiance_traced.append(irradiance_grid)
                    
        return irradiance_traced

    def global_irradiance_ground(self, direct_irradiance_ground, diffuse_irradiance_ground):

        global_irradiance_frames = []

        for i in range(len(direct_irradiance_ground)):

            calc = direct_irradiance_ground[i] + diffuse_irradiance_ground[i]

            global_irradiance_frames.append(calc)

        return global_irradiance_frames
    
    def power(self, area_grid, irradiance_frames):
        
        power_total = []

        ground_shape = (len(area_grid), len(area_grid[0]))
        
        for j in range(len(irradiance_frames)):
            power_ground = np.zeros(ground_shape)
            total_power = 0
            # Iterate over each point on the ground grid
            for p in range(ground_shape[0]):
                for q in range(ground_shape[1]):
                                
                    # Store the result in the irradiance_ground array
                    power_calc = irradiance_frames[j][p][q] * area_grid[p][q]
                    power_ground[p][q] = power_calc

                    total_power += power_calc
            
            power_total.append(total_power)

        return power_total
    
    def shaded_irradiance_rays(self, irradiance_frames, shaded_exposure_maps):

        shaded_irradiance_frames = []

        for i in range(len(irradiance_frames)):
            new_map = np.where(shaded_exposure_maps[i] == 1, irradiance_frames[i], 0)
            shaded_irradiance_frames.append(new_map)

        return shaded_irradiance_frames
    
    def solar_cells_irradiance_rays(self, irradiance_frames, solar_cell_exposure_maps, pol, n_list, d_list, incident_grid, wavelengths):

        solar_cells_irradiance_frames = []

        for i in range(len(irradiance_frames)):
            
            
            for j in range(len(wavelengths)):
                
                tmm.coh_tmm(pol, n_list, d_list, incident_grid)
                tau = 1
            
            new_map = np.where(solar_cell_exposure_maps[i] == 1, irradiance_frames[i], 0)
            solar_cells_irradiance_frames.append(new_map)




        
        



   