import numpy as np
from sun import Sun
class RayTracer:
    def __init__(self, polytunnel):
        """
        Initializes the RayTracer class for calculating solar irradiance on the polytunnel ground.
        
        :param polytunnel: Instance of the Polytunnel class.
        """
        self.polytunnel = polytunnel

    def irradiance_rays(self, normals_unit, sun_positions, sun_vecs):

        irradiance = []

        for i in range(len(sun_positions)):
            altitude = sun_positions[i][0]
            if altitude > 0:
                sun_vec = sun_vecs[i]
                irradiance.append(self.calculate_irradiance(normals_unit, sun_vec))

        return irradiance

    def calculate_irradiance(self, normals_unit, sun_vec):
        
        irradiance_surface = 1000 * np.tensordot(normals_unit, sun_vec, axes=(0, 0))
        
        return irradiance_surface
