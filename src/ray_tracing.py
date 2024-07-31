import numpy as np

class RayTracer:
    def __init__(self, polytunnel):
        """
        Initializes the RayTracer class for calculating solar irradiance on the polytunnel ground.
        
        :param polytunnel: Instance of the Polytunnel class.
        """
        self.polytunnel = polytunnel

    def trace_rays(self, ground_grid_x, ground_grid_y, sun_positions):

        irradiance = np.zeros_like(ground_grid_x)

        for altitude, azimuth in sun_positions:
            if altitude > 0:  # Only consider the sun positions where the sun is above the horizon
                irradiance += self.calculate_irradiance(ground_grid_x, ground_grid_y, altitude, azimuth)
        
        return irradiance

    def calculate_irradiance(self, ground_grid_x, ground_grid_y, altitude, azimuth):

        # Convert altitude and azimuth to radians
        altitude_rad = np.radians(altitude)
        azimuth_rad = np.radians(azimuth)
        
        # Compute the direction cosines of the sun rays
        sun_dir_x = np.sin(azimuth_rad) * np.cos(altitude_rad)
        sun_dir_y = np.cos(azimuth_rad) * np.cos(altitude_rad)
        sun_dir_z = np.sin(altitude_rad)

        # Calculate the projection of the sun direction on the ground plane
        dot_product = sun_dir_x * ground_grid_x + sun_dir_y * ground_grid_y + sun_dir_z * self.polytunnel.radius
        irradiance_contribution = np.maximum(dot_product, 0)  # Only positive contributions

        return irradiance_contribution
