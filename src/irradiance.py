import numpy as np
from sun import Sun
class TunnelIrradiance:
    def __init__(self, polytunnel):
        """
        Initializes the RayTracer class for calculating solar irradiance on the polytunnel ground.
        
        :param polytunnel: Instance of the Polytunnel class.
        """
        self.polytunnel = polytunnel

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
    
    def ray_trace(self, ground_grid, surface_grid, distances_grid, sun_vecs, irradiance_frames, transmissivity):
        
        ground_shape =  (len(distances_grid), len(distances_grid[0]))
        irradiance_traced = []

        for i in range(len(irradiance_frames)):
        
            irradiance_grid = np.zeros((ground_shape[0], ground_shape[0]))

        # Calculate intersections and closest points
            for j in range(ground_shape[0]):
                for k in range(ground_shape[0]):
                    # Get surface point
                    x_s = surface_grid[0][j][k]
                    y_s = surface_grid[1][j][k]
                    z_s = surface_grid[2][j][k]

                    
                    # Compute t for intersection with ground plane (z=0)
                    if sun_vecs[i][2] != 0:
                        t = z_s / sun_vecs[i][2] #sun_vec is outwards
                        
                        # Compute intersection point
                        x_int = x_s - t * sun_vecs[i][0]
                        y_int = y_s - t * sun_vecs[i][1]
                        z_int = 0  # Since we're intersecting with the ground plane
                        print(f"intersection found at {x_int, y_int, z_int}")
                        # Find the closest ground point
                        distances = np.sqrt((ground_grid[0] - x_int)**2 + 
                                            (ground_grid[1] - y_int)**2 + 
                                            (ground_grid[2] - z_int)**2)
                        
                        if (x_int >= ground_grid[0].min() and x_int <= ground_grid[0].max() and
                        y_int >= ground_grid[1].min() and y_int <= ground_grid[1].max()):
                            closest_j, closest_k = np.unravel_index(np.argmin(distances), distances.shape)
                            irradiance_point = irradiance_frames[i][j][k]

                        else:
                            irradiance_point = 0
                            closest_j = 0
                            closest_k = 0

                    else: 
                        irradiance_point = 0
                        closest_j = 0
                        closest_k = 0

                    irradiance_grid[closest_j][closest_k] =+ transmissivity * irradiance_point

            irradiance_traced.append(irradiance_grid)
                    
        return irradiance_traced

    def global_irradiance_ground(self, direct_irradiance_ground, diffuse_irradiance_ground):

        global_irradiance_frames = []

        for i in range(len(direct_irradiance_ground)):

            calc = direct_irradiance_ground[i] + diffuse_irradiance_ground[i]

            global_irradiance_frames.append(calc)

        return global_irradiance_frames
    
    def power(self, area_grid, irradiance_frames):
        
        power_frames = []

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
            power_frames.append(power_ground)

        return power_frames, power_total

        
        



   