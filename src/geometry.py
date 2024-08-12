import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Polytunnel:

    def __init__(self, radius = 1, length = 10, n_points = 1000):
        self.radius = radius
        self.length = length
        self.n = int(n_points)

    def generate_ground_grid(self):
        y_ground = np.linspace(-self.length, self.length, self.n)
        x_ground = np.linspace(-self.radius+0.01, self.radius-0.01, self.n)

        X, Y = np.meshgrid(x_ground, y_ground)
        return (X, Y)
    
    def generate_surface(self):
        # Create the grid in cylindrical coordinates
        y = np.linspace(-self.length, self.length, self.n)
        theta = np.linspace(0, np.pi, self.n)  # Semi-circular cross-section

        # Create a meshgrid for Y and Theta
        Y, Theta = np.meshgrid(y, theta)

        # Convert cylindrical coordinates to Cartesian coordinates
        X = self.radius * np.cos(Theta)  # Horizontal width
        Z = self.radius * np.sin(Theta)  # Height above the ground

        return (X, Y, Z)
    
    def generate_distances_grid(self, ground_grid, surface_grid):
        
        X_ground, Y_ground = ground_grid
        X_surface, Y_surface, Z_surface = surface_grid
        
        distances_list = []

        # Iterate over each point on the ground grid
        for i in range(X_ground.shape[0]):
            row_distances = []
            for j in range(X_ground.shape[1]):
                # Extract the ground point
                ground_point = np.array([X_ground[i, j], Y_ground[i, j], 0])

                # Compute the distance from this ground point to all surface points
                distances = np.sqrt((X_surface - ground_point[0])**2 + 
                                    (Y_surface - ground_point[1])**2 + 
                                    Z_surface**2)
                
                row_distances.append(distances)
            distances_list.append(row_distances)
        
        return distances_list

    def vec_grid(self, surface):

        X, Y, Z = surface
        surface_grid = np.empty(X.shape + (3,))
    
        # Iterate over the rows and columns of the grid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                surface_grid[i, j] = [X[i, j], Y[i, j], Z[i, j]]
        
        return surface_grid
    
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

        return normals_unit
    
    def surface_tilt(self, normals):

        tilt = np.arccos(normals[:][2])

        return tilt

    
    