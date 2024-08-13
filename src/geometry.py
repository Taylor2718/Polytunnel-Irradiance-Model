import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Polytunnel:

    def __init__(self, radius = 1, length = 10, n_points = 1000, xy_angle = 0, z_angle = 0):
        self.radius = radius
        self.length = length
        self.n = int(n_points)
        self.xy_angle = xy_angle
        self.z_angle = z_angle

    def generate_ground_grid(self):
        y_ground = np.linspace(-self.length, self.length, self.n)
        x_ground = np.linspace(-self.radius+0.01, self.radius-0.01, self.n)

        X, Y = np.meshgrid(x_ground, y_ground)
        Z = np.zeros_like(X)

        # Rotation in the XY plane
        cos_xy = np.cos(self.xy_angle)
        sin_xy = np.sin(self.xy_angle)
        
        X_rot_xy = X * cos_xy - Y * sin_xy
        Y = X * sin_xy + Y * cos_xy

        # Rotation around the Z-axis
        cos_z = np.cos(self.z_angle)
        sin_z = np.sin(self.z_angle)

        X = X_rot_xy * cos_z - Z * sin_z
        Z = X_rot_xy * sin_z + Z * cos_z  # Z_final will still be zero since Z is zero initially

        return (X, Y, Z)
    
    def generate_surface(self):
        # Create the grid in cylindrical coordinates
        y = np.linspace(-self.length, self.length, self.n)
        theta = np.linspace(0, np.pi, self.n)  # Semi-circular cross-section

        # Create a meshgrid for Y and Theta
        Y, Theta = np.meshgrid(y, theta)

        # Convert cylindrical coordinates to Cartesian coordinates
        X = self.radius * np.cos(Theta)  # Horizontal width
        Z = self.radius * np.sin(Theta)  # Height above the ground
        

        # Rotation in the XY plane
        cos_xy = np.cos(self.xy_angle)
        sin_xy = np.sin(self.xy_angle)
        
        X_rot_xy = X * cos_xy - Y * sin_xy
        Y = X * sin_xy + Y * cos_xy

        # Rotation around the Z-axis
        cos_z = np.cos(self.z_angle)
        sin_z = np.sin(self.z_angle)

        X = X_rot_xy * cos_z - Z * sin_z
        Z = X_rot_xy * sin_z + Z * cos_z

        return (X, Y, Z)
    
    def generate_distances_grid(self, ground_grid, surface_grid):
        
        X_ground, Y_ground, Z_ground = ground_grid
        X_surface, Y_surface, Z_surface = surface_grid
        
        distances_list = []
        unit_vectors_list = []

        # Iterate over each point on the ground grid
        for i in range(X_ground.shape[0]):
            row_distances = []
            row_unit_vectors = []

            for j in range(X_ground.shape[1]):
                # Extract the ground point
                ground_point = np.array([X_ground[i, j], Y_ground[i, j], Z_ground[i, j]])
                
                separation_vector_x = X_surface - ground_point[0]
                separation_vector_y = Y_surface - ground_point[1]
                separation_vector_z = Z_surface - ground_point[2]

                # Compute the distance from this ground point to all surface points
                distances = np.sqrt((X_surface - ground_point[0])**2 + 
                                    (Y_surface - ground_point[1])**2 + 
                                    (Z_surface - ground_point[2])**2)
                
                # Compute the unit vectors by dividing the separation vectors by the distances
                unit_vectors_x = separation_vector_x / distances
                unit_vectors_y = separation_vector_y / distances
                unit_vectors_z = separation_vector_z / distances
                
                # Stack the unit vectors into an array for this ground point
                unit_vectors = np.stack((unit_vectors_x, unit_vectors_y, unit_vectors_z), axis=-1)
                
                # Append the distances and unit vectors to their respective lists
                row_distances.append(distances)
                row_unit_vectors.append(unit_vectors)
                
            distances_list.append(row_distances)
            unit_vectors_list.append(row_unit_vectors)

        return distances_list, unit_vectors_list

    def ground_element_unit_vectors(self):
        
        X = self.generate_ground_grid()[0]
        Y = self.generate_ground_grid()[1]
        Z = self.generate_ground_grid()[2]

        dx = np.gradient(X, axis=0)
        dy = np.gradient(Y, axis=0)
        dz = np.gradient(Z, axis=0)

        # Compute normal vectors using the cross product of the gradients
        normals = np.cross(np.array([np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]), np.array([dx, dy, dz]), axis=0)

        # Normalize the normal vectors
        normals_magnitude = np.linalg.norm(normals, axis=0)
        normals_unit = normals / normals_magnitude

        return normals_unit, normals_magnitude
    
    def surface_element_unit_vectors(self):
        
        X = self.generate_surface()[0]
        Y = self.generate_surface()[1]
        Z = self.generate_surface()[2]

        dx = np.gradient(X, axis=0)
        dy = np.gradient(Y, axis=0)
        dz = np.gradient(Z, axis=0)

        # Compute normal vectors using the cross product of the gradients
        normals = np.cross(np.array([np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]), np.array([dx, dy, dz]), axis=0)

        # Normalize the normal vectors
        normals_magnitude = np.linalg.norm(normals, axis=0)
        normals_unit = normals / normals_magnitude

        return normals_unit, normals_magnitude
    
    
    def surface_tilt(self, normals):

        tilt = np.arccos(normals[:][2])

        return tilt

    def generate_angle_grid(self, ground_grid, surface_grid, ground_unit_vectors, surface_unit_vectors):
    
        X_ground, Y_ground, Z_ground = ground_grid
        X_surface, Y_surface, Z_surface = surface_grid
        
        # Initialize a 4D array to store the angles
        angles_grid = np.zeros((X_ground.shape[0], X_ground.shape[1], X_surface.shape[0], X_surface.shape[1]))

        # Iterate over each point on the ground grid
        for i in range(X_ground.shape[0]):
            for j in range(X_ground.shape[1]):
                # Extract the ground point and the corresponding unit vector
                ground_point = np.array([X_ground[i, j], Y_ground[i, j], Z_ground[i, j]])
                ground_unit_vector = ground_unit_vectors[:, i, j]
                
                # Iterate over each point on the surface grid
                for k in range(X_surface.shape[0]):
                    for l in range(X_surface.shape[1]):
                        # Extract the surface point and corresponding unit vector
                        surface_point = np.array([X_surface[k, l], Y_surface[k, l], Z_surface[k, l]])
                        surface_unit_vector = surface_unit_vectors[:, k, l]
                        
                        # Normalize the translated surface vector to ensure it's a unit vector
                        surface_unit_vector /= np.linalg.norm(surface_unit_vector)
                        
                        # Compute the dot product
                        dot_product = np.dot(ground_unit_vector, surface_unit_vector)
                        
                        # Store the result in the 4D grid
                        angles_grid[i][j][k][l] = dot_product
        
        return angles_grid