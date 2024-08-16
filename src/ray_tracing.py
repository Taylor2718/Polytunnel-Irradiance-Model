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
    
    def find_intersection(self, surface_point, unit_vector, ground_plane_z):
        """
        Find the intersection of a ray with a horizontal plane (ground).

        Parameters:
        - surface_point: Point on the surface where the ray starts (x, y, z).
        - unit_vector: Direction of the ray (vx, vy, vz).
        - ground_plane_z: The z-coordinate of the ground plane (typically 0).

        Returns:
        - intersection_point: The coordinates of the intersection on the ground (x, y, z).
        """
        if unit_vector[2] != 0:
            t = (ground_plane_z - surface_point[2]) / unit_vector[2]
            intersection_point = surface_point + t * unit_vector
            return intersection_point
        return None
    
    def get_ground_index(self, intersection_point, ground_grid):
        """
        Map the intersection point to the nearest grid index on the ground.

        Parameters:
        - intersection_point: Coordinates of the intersection on the ground (x, y, z).
        - ground_grid: 3D array of ground points.

        Returns:
        - index: Tuple of grid indices (i, j) for the ground grid.
        """
        distances = np.sqrt(
            (ground_grid[0] - intersection_point[0])**2 + 
            (ground_grid[1] - intersection_point[1])**2 + 
            (ground_grid[2] - intersection_point[2])**2
        )
        min_index = np.unravel_index(np.argmin(distances), distances.shape)
        return min_index
    
    def ray_surface_to_ground(self, surface_grid, ground_grid, sun_vecs, irradiance_frames, ground_plane_z):
        """
        Trace rays from each point on the surface to the ground and map irradiance values.

        Parameters:
        - surface_grid: 3D array of surface grid points [3, 20, 20].
        - ground_grid: 3D array of ground grid points [3, 20, 20].
        - sun_vecs: Array of unit vectors for the sun direction over time [N, 3].
        - irradiance_frames: 3D array of irradiance values on the surface over time [N, 20, 20].
        - ground_plane_z: Z-coordinate of the ground plane.

        Returns:
        - irradiance_on_ground: 3D array of irradiance values on the ground over time [N, 20, 20].
        """
        N, grid_x, grid_y = irradiance_frames.shape
        irradiance_on_ground = np.zeros((N, grid_x, grid_y))
        
        for I in range(N):  # Loop over time points
            for i in range(grid_x):
                for j in range(grid_y):
                    surface_point = np.array([
                        surface_grid[0][i][j], 
                        surface_grid[1][i][j], 
                        surface_grid[2][i][j]
                    ])
                    unit_vector = sun_vecs[I]
                    
                    intersection_point = self.find_intersection(surface_point, unit_vector, ground_plane_z)
                    
                    if intersection_point is not None:
                        ground_idx = self.get_ground_index(intersection_point, ground_grid)
                        
                        if ground_idx is not None:
                            irradiance_on_ground[I][ground_idx[0]][ground_idx[1]] += irradiance_frames[I][i][j]
        
        return irradiance_on_ground
    
    def global_irradiance_ground(self, direct_irradiance_ground, diffuse_irradiance_ground):

        global_irradiance_frames = []

        for i in range(len(direct_irradiance_ground)):

            calc = direct_irradiance_ground[i] + diffuse_irradiance_ground[i]

            global_irradiance_frames.append(calc)

        return global_irradiance_frames
    



   