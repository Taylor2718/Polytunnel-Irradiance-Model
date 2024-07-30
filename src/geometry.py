import numpy as np

class Polytunnel:

    def __init__(self, radius = 1, length = 10, resolution = 0.1):
        self.radius = radius
        self.length = length
        self.resolution = resolution

    def generate_ground_grid(self):
        x = np.arange(-self.length / 2, self.length / 2, self.resolution)
        y = np.arange(-self.radius, self.radius, self.resolution)
        return np.meshgrid(x, y)

