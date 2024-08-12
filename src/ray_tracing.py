import numpy as np
from sun import Sun
class RayTracer:
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
        return irradiance
    
    def calculate_irradiance(self, normals_unit, sun_vec, spectral_irradiance):
        
        irradiance_surface = np.clip((spectral_irradiance * np.tensordot(normals_unit, sun_vec, axes=(0, 0))), a_min = 0, a_max = None)
        
        return irradiance_surface
    
    def diffuse_irradiance_ground(self, distances_grid, irradiance_frames, diffusivity_ratio):

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
                            if distance > 0:  # Avoid division by zero
                                total_irradiance += diffusivity_ratio*irradiance_frames[j][k][l] / distance
                    
                    # Store the result in the irradiance_ground array
                    irradiance_ground[p][q] = total_irradiance
        
            diffuse_irradiance_frames.append(irradiance_ground)

        return diffuse_irradiance_frames

        

        
    
   