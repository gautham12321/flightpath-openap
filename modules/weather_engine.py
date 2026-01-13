"""
Weather data engine using FastMeteo for ERA5 data.
Provides fast and simple weather grid data for flight optimization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path

try:
    from fastmeteo.source import ArcoEra5
    FASTMETEO_AVAILABLE = True
except ImportError:
    FASTMETEO_AVAILABLE = False
    ArcoEra5 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherEngine:
    """
    Handles weather data fetching using FastMeteo ERA5 data.
    Provides optimized weather grids for flight trajectory optimization.
    """
    
    def __init__(self, cache_dir: str = './data/weather_cache'):
        """
        Initialize the WeatherEngine.
        
        Args:
            cache_dir: Directory to store cached weather data
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        if not FASTMETEO_AVAILABLE:
            logger.warning("FastMeteo library not available. Weather features will be limited.")
            self.grid = None
        else:
            # Initialize ArcoEra5 grid with local cache
            self.grid = ArcoEra5(local_store=cache_dir, model_levels=37)
            logger.info(f"Initialized WeatherEngine with cache: {cache_dir}")
    
    def get_weather_grid(
        self,
        origin: Tuple[float, float],
        dest: Tuple[float, float],
        flight_date: str,
        flight_level: int = 350
    ) -> Optional[pd.DataFrame]:
        """
        Get weather grid data for a flight route using FastMeteo.
        
        Args:
            origin: Tuple of (latitude, longitude) for origin
            dest: Tuple of (latitude, longitude) for destination
            flight_date: Date string in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            flight_level: Flight level in hundreds of feet (default: 350 = 35,000 ft)
        
        Returns:
            DataFrame with weather data formatted for OpenAP compatibility
            Columns: lat, lon, h (altitude in m), u, v (wind components in m/s), t (temperature in K), ts (timestamp)
        """
        if not FASTMETEO_AVAILABLE or self.grid is None:
            logger.warning("FastMeteo not available, returning None")
            return None
        
        try:
            origin_lat, origin_lon = origin
            dest_lat, dest_lon = dest
            
            # Parse flight date
            flight_datetime = pd.to_datetime(flight_date)
            
            logger.info(f"Fetching weather grid for route:")
            logger.info(f"  Origin: ({origin_lat:.4f}, {origin_lon:.4f})")
            logger.info(f"  Destination: ({dest_lat:.4f}, {dest_lon:.4f})")
            logger.info(f"  Date: {flight_datetime}")
            logger.info(f"  Flight Level: FL{flight_level}")
            
            # Convert flight level to altitude in feet
            altitude_ft = flight_level * 100
            
            # Create a simple flight trajectory DataFrame
            # FastMeteo's interpolate method expects: timestamp, latitude, longitude, altitude
            num_points = 20  # Number of waypoints along the route
            
            flight_trajectory = pd.DataFrame({
                'timestamp': [flight_datetime] * num_points,
                'latitude': np.linspace(origin_lat, dest_lat, num_points),
                'longitude': np.linspace(origin_lon, dest_lon, num_points),
                'altitude': [altitude_ft] * num_points  # in feet
            })
            
            logger.info(f"Created flight trajectory with {num_points} points")
            logger.info("Fetching weather data from FastMeteo (this may take a moment)...")
            
            # Use FastMeteo's interpolate method
            weather_data = self.grid.interpolate(flight_trajectory)
            
            if weather_data is None or weather_data.empty:
                logger.warning("No weather data returned from FastMeteo")
                return None
            
            logger.info(f"Received weather data: {len(weather_data)} points")
            logger.debug(f"Original columns: {weather_data.columns.tolist()}")
            
            # Format data for OpenAP compatibility
            formatted_data = self._format_for_openap(weather_data, altitude_ft, flight_datetime)
            
            if formatted_data is not None and not formatted_data.empty:
                logger.info(f"Formatted weather grid: {len(formatted_data)} points")
                logger.debug(f"Formatted columns: {formatted_data.columns.tolist()}")
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error fetching weather grid: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _format_for_openap(
        self,
        weather_data: pd.DataFrame,
        altitude_ft: float,
        flight_datetime: datetime
    ) -> pd.DataFrame:
        """
        Format FastMeteo weather data for OpenAP compatibility.
        
        Args:
            weather_data: Raw weather data from FastMeteo
            altitude_ft: Altitude in feet
            flight_datetime: Flight datetime for timestamp calculation
        
        Returns:
            Formatted DataFrame with OpenAP-compatible columns
        """
        try:
            df = weather_data.copy()
            
            # FastMeteo returns columns like: u_component_of_wind, v_component_of_wind, temperature, specific_humidity
            # We need to rename them for OpenAP: u, v, t, h
            
            column_mapping = {
                'u_component_of_wind': 'u',
                'v_component_of_wind': 'v',
                'temperature': 't',
                'specific_humidity': 'q',
                'latitude': 'lat',
                'longitude': 'lon',
                'altitude': 'h'  # altitude column if present
            }
            
            # Rename columns that exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
                    logger.debug(f"Renamed column: {old_name} → {new_name}")
            
            # Ensure required columns exist
            required_columns = ['lat', 'lon', 'u', 'v']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns after mapping: {missing_columns}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Add/convert altitude to meters if not present or in wrong units
            if 'h' not in df.columns:
                df['h'] = altitude_ft * 0.3048  # feet to meters
                logger.debug(f"Added altitude column: {altitude_ft:.0f} ft ({altitude_ft * 0.3048:.0f} m)")
            elif 'altitude' in weather_data.columns:
                # FastMeteo returns altitude in feet, convert to meters
                df['h'] = df['h'] * 0.3048
                logger.debug("Converted altitude from feet to meters")
            
            # Temperature should be in Kelvin for OpenAP
            # FastMeteo returns it in Kelvin already, but verify it's reasonable
            if 't' in df.columns:
                t_mean = df['t'].mean()
                if t_mean < 150:  # Likely Celsius, convert to Kelvin
                    df['t'] = df['t'] + 273.15
                    logger.debug("Converted temperature from Celsius to Kelvin")
                else:
                    logger.debug(f"Temperature appears to be in Kelvin (mean: {t_mean:.1f} K)")
            else:
                # Use ISA standard atmosphere approximation
                altitude_m = df['h'].mean() if 'h' in df.columns else altitude_ft * 0.3048
                df['t'] = 288.15 - 0.0065 * altitude_m
                logger.debug("Added temperature column using ISA approximation")
            
            # Create relative timestamps
            if 'timestamp' in weather_data.columns:
                # Convert to relative seconds
                timestamps = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else pd.to_datetime(weather_data['timestamp'])
                start_time = timestamps.min()
                df['ts'] = (timestamps - start_time).dt.total_seconds().fillna(0)
                logger.debug("Created relative timestamp column")
            else:
                # Create evenly spaced timestamps
                if len(df) > 1:
                    # Estimate based on distance
                    if 'lat' in df.columns and 'lon' in df.columns:
                        lat_diff = df['lat'].iloc[-1] - df['lat'].iloc[0]
                        lon_diff = df['lon'].iloc[-1] - df['lon'].iloc[0]
                        distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)
                        distance_km = distance_deg * 111
                        flight_duration_hours = distance_km / 833  # 450 kts cruise
                        flight_duration_seconds = flight_duration_hours * 3600
                        df['ts'] = np.linspace(0, flight_duration_seconds, len(df))
                    else:
                        df['ts'] = np.arange(0, len(df) * 60, 60)
                else:
                    df['ts'] = 0
                logger.debug(f"Created timestamp column: 0 to {df['ts'].max():.0f} seconds")
            
            # Select and order columns for OpenAP
            output_columns = ['lat', 'lon', 'h', 'u', 'v', 't', 'ts']
            available_columns = [col for col in output_columns if col in df.columns]
            
            df = df[available_columns]
            
            # Verify wind components are in m/s (FastMeteo returns m/s)
            if 'u' in df.columns and 'v' in df.columns:
                u_mean = df['u'].abs().mean()
                v_mean = df['v'].abs().mean()
                logger.debug(f"Wind speeds: u={u_mean:.1f} m/s, v={v_mean:.1f} m/s")
            
            # Remove any NaN values
            initial_len = len(df)
            df = df.dropna()
            if len(df) < initial_len:
                logger.warning(f"Removed {initial_len - len(df)} rows with NaN values")
            
            return df
            
        except Exception as e:
            logger.error(f"Error formatting weather data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_weather_summary(
        self,
        weather_data: pd.DataFrame
    ) -> dict:
        """
        Generate summary statistics for weather data.
        
        Args:
            weather_data: Formatted weather DataFrame
        
        Returns:
            Dictionary with weather statistics
        """
        if weather_data is None or weather_data.empty:
            return {}
        
        summary = {
            'num_points': len(weather_data),
            'avg_u_wind': weather_data['u'].mean() if 'u' in weather_data.columns else None,
            'avg_v_wind': weather_data['v'].mean() if 'v' in weather_data.columns else None,
            'avg_wind_speed': np.sqrt(weather_data['u']**2 + weather_data['v']**2).mean() if 'u' in weather_data.columns else None,
            'avg_temperature': weather_data['t'].mean() if 't' in weather_data.columns else None,
            'altitude_range': (weather_data['h'].min(), weather_data['h'].max()) if 'h' in weather_data.columns else None,
        }
        
        # Calculate wind direction
        if 'u' in weather_data.columns and 'v' in weather_data.columns:
            wind_dir_rad = np.arctan2(-weather_data['u'], -weather_data['v'])
            wind_dir_deg = np.degrees(wind_dir_rad) % 360
            summary['avg_wind_direction'] = wind_dir_deg.mean()
        
        return summary


def main():
    """
    Example usage of WeatherEngine.
    """
    engine = WeatherEngine(cache_dir='./data/weather_cache')
    
    if engine.grid is None:
        print("\n✗ FastMeteo not available. Install it with: pip install fastmeteo")
        return
    
    # Example: Get weather for Mumbai to Kochi flight
    weather_grid = engine.get_weather_grid(
        origin=(19.0896, 72.8656),  # Mumbai (BOM)
        dest=(9.9312, 76.2673),      # Kochi (COK)
        flight_date='2024-10-12 01:10:00',  # Use recent date
        flight_level=350
    )
    
    if weather_grid is not None:
        print(f"\n✓ Weather data fetched successfully!")
        print(f"  Data points: {len(weather_grid)}")
        print(f"\nColumns: {weather_grid.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(weather_grid.head())
        
        # Get summary
        summary = engine.get_weather_summary(weather_grid)
        print(f"\nWeather Summary:")
        for key, value in summary.items():
            if isinstance(value, tuple):
                print(f"  {key}: {value[0]:.2f} to {value[1]:.2f}")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("\n✗ Failed to fetch weather data")


if __name__ == "__main__":
    main()


