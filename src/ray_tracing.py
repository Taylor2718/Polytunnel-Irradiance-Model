import numpy as np

class Sun:
    def __init__(self, altitude, azimuth):
        self.altitude = np.radians(altitude)
        self.azimuth = np.radians(azimuth)
        self.direction = self.calculate_direction()

    def calculate_direction(self):
            return np.array([
                np.cos(self.altitude) * np.cos(self.azimuth),
                np.cos(self.altitude) * np.sin(self.azimuth),
                np.sin(self.altitude)
            ])
    
class RayTracer:
    def __init__(self, tunnel, sun):
        self.tunnel = tunnel
        
    def trace_rays(self, ground_x, ground_y):
        radius = self.tunnel.radius
        length = self.tunnel.length
        sun_direction = self.sun.calculate_direction

        # Initialize the irradiance grid
        irradiance = np.zeros_like(ground_x)

        for i in range(ground_x.shape[0]):
            for j in range(ground_x.shape[1]):
                point = np.array([ground_x[i, j], ground_y[i, j], 0])
                ray_origin = point + sun_direction * 100  # Trace the ray backwards

                # Check if the ray intersects with the poly tunnel
                if (ray_origin[1] ** 2 + ray_origin[2] ** 2) <= radius ** 2:
                    irradiance[i, j] = 1  # Simplified irradiance value

        return irradiance

    