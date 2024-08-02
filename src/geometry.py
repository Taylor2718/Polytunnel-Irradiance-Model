import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Polytunnel:

    def __init__(self, radius = 1, length = 10, n_points = 1000):
        self.radius = radius
        self.length = length
        self.n = int(n_points)

    def generate_ground_grid(self):
        x = np.arange(-self.length / 2, self.length / 2, self.n)
        y = np.arange(-self.radius, self.radius, self.n)
        return np.meshgrid(x, y)
    
    def generate_surface(self):
        # Create the grid in cylindrical coordinates
        y = np.linspace(0, self.length, self.n)
        theta = np.linspace(0, np.pi, self.n)  # Semi-circular cross-section

        # Create a meshgrid for Y and Theta
        Y, Theta = np.meshgrid(y, theta)

        # Convert cylindrical coordinates to Cartesian coordinates
        X = self.radius * np.cos(Theta)  # Horizontal width
        Z = self.radius * np.sin(Theta)  # Height above the ground

        return (X, Y, Z)
    
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
