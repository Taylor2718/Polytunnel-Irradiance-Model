import numpy as np
from sun import Sun
class RayTracer:
    def __init__(self, polytunnel):
        """
        Initializes the RayTracer class for calculating solar irradiance on the polytunnel ground.
        
        :param polytunnel: Instance of the Polytunnel class.
        """
        self.polytunnel = polytunnel
    """""
    def sun_surface_az_alt(self, sun_vecs, surface_vecs, distance_array):
        az_array = []
        alt_array = []
        for i in range(len(surface_vecs)):
            for j in range(len())
            c = (distance_array[i] * sun_vecs[i]) - surface_vecs[j]
            x, y, z = c
            az = np.degrees(np.arctan2(x, y))
            alt = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
            az_array.append(az)
            alt_array.append(alt)
        return az_array, alt_array
    """""
    def irradiance_rays(self, normals_unit, sun_positions, sun_vecs, irradiance_frames):

        irradiance = []

        for i in range(len(sun_positions)):
            altitude = sun_positions[i][0]
            if altitude > 0:
                sun_vec = sun_vecs[i]
                calc = self.calculate_irradiance(normals_unit, sun_vec, irradiance_frames[i])
                irradiance.append(calc)
        return irradiance
    
    def calculate_irradiance(self, normals_unit, sun_vec, spectral_irradiance):
        
        irradiance_surface = np.clip((spectral_irradiance * np.tensordot(normals_unit, sun_vec, axes=(0, 0))), a_min = 0, a_max = None)
        
        return irradiance_surface
    
   