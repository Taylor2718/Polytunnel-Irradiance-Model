import numpy as np

class Tracing:
    def __init__(self, main_tunnel, sun_vecs, left_tunnel=None, right_tunnel=None):
        """
        Initialize the Tracing class.

        Parameters:
            main_tunnel (Tunnel): The main polytunnel.
            sun (Sun): The sun object containing sun direction vectors.
            left_tunnel (Tunnel, optional): A neighboring polytunnel to the left.
            right_tunnel (Tunnel, optional): A neighboring polytunnel to the right.
        """
        self.main_tunnel = main_tunnel
        self.left_tunnel = left_tunnel
        self.right_tunnel = right_tunnel
        self.sun_vecs = sun_vecs

    def ray_intersects_triangle(self, ray_origin, ray_direction, p0, p1, p2):
        """
        Möller–Trumbore intersection algorithm to check if a ray intersects with a triangle.
        
        Parameters:
            ray_origin (np.array): The origin of the ray.
            ray_direction (np.array): The direction of the ray (unit vector).
            p0, p1, p2 (np.array): Vertices of the triangle.
            
        Returns:
            bool: True if the ray intersects the triangle, False otherwise.
        """
        EPSILON = 1e-8
        edge1 = p1 - p0
        edge2 = p2 - p0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)
        if -EPSILON < a < EPSILON:
            return False  # The ray is parallel to the triangle.

        f = 1.0 / a
        s = ray_origin - p0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)
        if v < 0.0 or u + v > 1.0:
            return False

        t = f * np.dot(edge2, q)
        if t > EPSILON:  # Ray intersection with the triangle
            return True

        return False  # No valid intersection

    def ray_intersects_surface(self, ray_origin, ray_direction, surface_grid):
        """
        Check if a ray from the given point in the direction of ray_direction intersects
        with any triangle on the surface grid.

        Parameters:
            ray_origin (np.array): The 3D coordinates of the point on the main polytunnel (x, y, z).
            ray_direction (np.array): The sun vector pointing towards the sun (3D unit vector).
            surface_grid (np.array): 3D mesh grid of the polytunnel surface to check against.

        Returns:
            bool: True if the ray intersects the surface, False otherwise.
        """
        # Iterate over all triangles formed by the surface grid
        for i in range(len(surface_grid[0]) - 1):
            for j in range(len(surface_grid[0][0]) - 1):
                # Extract vertices of the first triangle in the grid square
                p0 = np.array([surface_grid[0][i][j], surface_grid[1][i][j], surface_grid[2][i][j]])
                p1 = np.array([surface_grid[0][i+1][j], surface_grid[1][i+1][j], surface_grid[2][i+1][j]])
                p2 = np.array([surface_grid[0][i][j+1], surface_grid[1][i][j+1], surface_grid[2][i][j+1]])

                if self.ray_intersects_triangle(ray_origin, ray_direction, p0, p1, p2):
                    return True

                # Extract vertices of the second triangle in the grid square
                p0 = np.array([surface_grid[0][i+1][j], surface_grid[1][i+1][j], surface_grid[2][i+1][j]])
                p1 = np.array([surface_grid[0][i+1][j+1], surface_grid[1][i+1][j+1], surface_grid[2][i+1][j+1]])
                p2 = np.array([surface_grid[0][i][j+1], surface_grid[1][i][j+1], surface_grid[2][i][j+1]])

                if self.ray_intersects_triangle(ray_origin, ray_direction, p0, p1, p2):
                    return True

        return False

    def calculate_light_exposure(self):
        """
        Calculate the light exposure for each point on the main polytunnel.

        Returns:
            np.array: A 2D array representing the light exposure at each point on the main tunnel surface grid.
        """
        surface_grid = self.main_tunnel.generate_surface()
        sun_vecs = self.sun_vecs

        exposure_maps = []

        for i in range(len(sun_vecs)):
            exposure_map = np.ones((len(surface_grid[0]), len(surface_grid[0][0])))

            for j in range(len(surface_grid[0])):
                for k in range(len(surface_grid[0][0])):
                    point = np.array([surface_grid[0][j][k], surface_grid[1][j][k], surface_grid[2][j][k]])
                    sun_vec = sun_vecs[i]

                    # Check if the point is shaded by the left or right tunnel
                    if self.left_tunnel and self.ray_intersects_surface(point, sun_vec, self.left_tunnel.generate_surface()):
                        exposure_map[j, k] = 0  # Mark as shadowed (0 light exposure)
                    if self.right_tunnel and self.ray_intersects_surface(point, sun_vec, self.right_tunnel.generate_surface()):
                        exposure_map[j, k] = 0  # Mark as shadowed (0 light exposure)
            

            exposure_maps.append(exposure_map)

        return exposure_maps