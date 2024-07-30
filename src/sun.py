import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Sun:
    def __init__(self, start_time='2024-07-30T00:00:00Z', end_time='2024-07-30T23:59:59Z', latitude = 51.1950, longitude = 0.2757, altitude=0, resolution_minutes=1):
        self.start_time = self.parse_datetime(start_time)
        self.end_time = self.parse_datetime(end_time)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.resolution = resolution_minutes
    
    @staticmethod
    def parse_datetime(datetime_str):
        return datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%SZ')

    def get_times(self):
        times = []
        current_time = self.start_time
        while current_time <= self.end_time:
            times.append(current_time)
            current_time += timedelta(minutes=self.resolution)
        return times

    def generate_sun_positions(self):
        times = self.get_times()
        
        # Create a DataFrame for the times
        times_df = pd.DatetimeIndex(times)

        # Get solar position data
        solar_position = pvlib.solarposition.get_solarposition(times_df, self.latitude, self.longitude, altitude=self.altitude)

        # Extract altitude and azimuth
        altitude_array = solar_position['apparent_elevation'].values
        azimuth_array = solar_position['azimuth'].values
        
        return altitude_array, azimuth_array
