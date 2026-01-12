"""
Flight physics engine for analyzing historical flight tracks.
Uses OpenAP to calculate fuel consumption and other performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from openap import FuelFlow
from openap.extra import aero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightAnalyzer:
    """
    Analyzes flight tracks using physics-based models from OpenAP.
    Calculates fuel consumption, flight performance metrics, etc.
    """
    
    def __init__(self, aircraft_type: str = 'A320'):
        """
        Initialize the FlightAnalyzer.
        
        Args:
            aircraft_type: ICAO aircraft type code (default: 'A320')
        """
        self.aircraft_type = aircraft_type
        try:
            self.fuel_flow = FuelFlow(ac=aircraft_type)
            logger.info(f"Initialized FlightAnalyzer for {aircraft_type}")
        except Exception as e:
            logger.error(f"Error initializing FuelFlow for {aircraft_type}: {str(e)}")
            raise
        
        # Standard aircraft parameters (can be refined based on aircraft type)
        self.standard_takeoff_mass = 60000  # kg (typical for A320)
        self.standard_empty_mass = 42000    # kg (typical for A320)
        self.max_fuel_capacity = 24000      # kg (typical for A320)
    
    def calculate_track_fuel(self, track_df: pd.DataFrame) -> float:
        """
        Calculate total fuel consumption for a flight track.
        
        Args:
            track_df: DataFrame with columns: timestamp, latitude, longitude, altitude, speed
                     - altitude in feet
                     - speed in knots (assumed to be ground speed or TAS)
        
        Returns:
            Total fuel consumed in kg
        """
        if track_df.empty:
            logger.warning("Empty track DataFrame provided")
            return 0.0
        
        # Make a copy to avoid modifying the original
        df = track_df.copy().reset_index(drop=True)
        
        # Validate required columns
        required_cols = ['timestamp', 'latitude', 'longitude', 'altitude', 'speed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(df) < 2:
            logger.warning("Track has less than 2 points, cannot calculate fuel")
            return 0.0
        
        # Initialize tracking variables
        total_fuel = 0.0
        current_mass = self.standard_takeoff_mass
        
        logger.info(f"Calculating fuel for {len(df)} track points")
        
        # Iterate through each segment
        for i in range(len(df) - 1):
            try:
                # Current and next points
                current = df.iloc[i]
                next_point = df.iloc[i + 1]
                
                # Calculate segment parameters
                dt = next_point['timestamp'] - current['timestamp']  # seconds
                
                if dt <= 0:
                    logger.warning(f"Invalid time delta at index {i}: {dt}s")
                    continue
                
                # Altitude in feet, convert to meters for calculations
                alt_ft = current['altitude']
                alt_m = alt_ft * 0.3048
                
                # Calculate path angle (climb/descent angle)
                path_angle = self._calculate_path_angle(
                    current['latitude'], current['longitude'], current['altitude'],
                    next_point['latitude'], next_point['longitude'], next_point['altitude']
                )
                
                # True Airspeed (TAS) in m/s
                # Assuming speed is in knots, convert to m/s
                tas_kts = current['speed']
                tas_ms = tas_kts * 0.514444  # knots to m/s
                
                # Ensure reasonable values
                if tas_ms < 50:  # Less than ~100 knots, unrealistic cruise
                    logger.debug(f"Low speed at index {i}: {tas_kts} kts")
                    tas_ms = max(tas_ms, 50)  # Minimum 50 m/s (~100 kts)
                
                if alt_m < 0:
                    alt_m = 0
                
                # Calculate vertical speed from path angle
                # vs = TAS * sin(path_angle)
                vs = tas_ms * np.sin(path_angle)  # in m/s
                
                # Calculate fuel flow in kg/s using OpenAP
                try:
                    fuel_flow_rate = self.fuel_flow.enroute(
                        mass=current_mass,
                        tas=tas_ms,
                        alt=alt_m,
                        vs=vs
                    )
                    
                    # Calculate fuel consumed in this segment
                    segment_fuel = fuel_flow_rate * dt
                    
                    # Validate fuel consumption is reasonable
                    if segment_fuel < 0:
                        logger.warning(f"Negative fuel flow at index {i}, setting to 0")
                        segment_fuel = 0
                    
                    # Update total fuel and mass
                    total_fuel += segment_fuel
                    current_mass -= segment_fuel
                    
                    # Ensure mass doesn't go below empty mass
                    if current_mass < self.standard_empty_mass:
                        logger.warning(f"Mass below empty mass at index {i}, adjusting")
                        current_mass = self.standard_empty_mass
                        
                except Exception as e:
                    logger.warning(f"Error calculating fuel flow at index {i}: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing segment {i}: {str(e)}")
                continue
        
        logger.info(f"Total fuel consumed: {total_fuel:.2f} kg")
        logger.info(f"Final mass: {current_mass:.2f} kg")
        
        return total_fuel
    
    def _calculate_path_angle(
        self,
        lat1: float, lon1: float, alt1: float,
        lat2: float, lon2: float, alt2: float
    ) -> float:
        """
        Calculate the path angle (climb/descent angle) between two points.
        
        Args:
            lat1, lon1, alt1: First point coordinates (lat/lon in degrees, alt in feet)
            lat2, lon2, alt2: Second point coordinates (lat/lon in degrees, alt in feet)
        
        Returns:
            Path angle in radians (positive for climb, negative for descent)
        """
        # Calculate horizontal distance using Haversine formula
        horizontal_dist = self._haversine_distance(lat1, lon1, lat2, lon2)  # in meters
        
        if horizontal_dist < 1:  # Less than 1 meter, effectively no movement
            return 0.0
        
        # Vertical distance in meters
        vertical_dist = (alt2 - alt1) * 0.3048  # feet to meters
        
        # Path angle = arctan(vertical / horizontal)
        path_angle = np.arctan2(vertical_dist, horizontal_dist)
        
        # Limit path angle to reasonable values (-15 to +15 degrees)
        max_angle = np.radians(15)
        path_angle = np.clip(path_angle, -max_angle, max_angle)
        
        return path_angle
    
    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lat1, lon1: First point coordinates in degrees
            lat2, lon2: Second point coordinates in degrees
        
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        # Earth radius in meters
        R = 6371000
        distance = R * c
        
        return distance
    
    def analyze_flight_statistics(self, track_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a flight track.
        
        Args:
            track_df: DataFrame with flight track data
        
        Returns:
            Dictionary with various flight statistics
        """
        if track_df.empty:
            return {}
        
        df = track_df.copy()
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(len(df) - 1):
            dist = self._haversine_distance(
                df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                df.iloc[i + 1]['latitude'], df.iloc[i + 1]['longitude']
            )
            total_distance += dist
        
        # Calculate duration
        duration_seconds = df['timestamp'].max() - df['timestamp'].min()
        duration_hours = duration_seconds / 3600
        
        # Calculate fuel consumption
        fuel_consumed = self.calculate_track_fuel(df)
        
        stats = {
            'total_distance_km': total_distance / 1000,
            'total_distance_nm': total_distance / 1852,  # nautical miles
            'duration_hours': duration_hours,
            'duration_minutes': duration_seconds / 60,
            'fuel_consumed_kg': fuel_consumed,
            'avg_altitude_ft': df['altitude'].mean(),
            'max_altitude_ft': df['altitude'].max(),
            'avg_speed_kts': df['speed'].mean(),
            'max_speed_kts': df['speed'].max(),
            'num_data_points': len(df),
        }
        
        # Calculate fuel efficiency
        if total_distance > 0:
            stats['fuel_per_km'] = fuel_consumed / (total_distance / 1000)
            stats['fuel_per_nm'] = fuel_consumed / (total_distance / 1852)
        
        return stats


def main():
    """
    Example usage of FlightAnalyzer.
    """
    analyzer = FlightAnalyzer(aircraft_type='A320')
    from  data_loader import FlightLoader
    loader=FlightLoader()
    flight = loader.load_csv('data/AI2886_38a1bfd2.csv')
    track = flight['AIC2886']
    # 
    fuel = analyzer.calculate_track_fuel(track)
    print(f"Total fuel consumed: {fuel:.2f} kg")
    # 
    stats = analyzer.analyze_flight_statistics(track)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("FlightAnalyzer module loaded successfully")


if __name__ == "__main__":
    main()
