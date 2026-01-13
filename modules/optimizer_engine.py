"""
Flight trajectory optimization engine using OpenAP.
Includes weather data integration for realistic trajectory planning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime, timedelta

try:
    from openap.top import CompleteFlight
    OPENAP_TOP_AVAILABLE = True
except ImportError:
    OPENAP_TOP_AVAILABLE = False
    CompleteFlight = None

try:
    from .weather_engine import WeatherEngine
    WEATHER_ENGINE_AVAILABLE = True
except ImportError:
    WEATHER_ENGINE_AVAILABLE = False
    WeatherEngine = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherHandler:
    """
    Handles weather data fetching and processing for flight optimization.
    Provides wind grid data for trajectory optimization.
    """
    
    def __init__(self):
        """Initialize the WeatherHandler."""
        logger.info("Initialized WeatherHandler")
    
    def fetch_wind_grid(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        date_str: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch wind grid data for the given route and date.
        
        Args:
            origin_lat, origin_lon: Origin coordinates
            dest_lat, dest_lon: Destination coordinates
            date_str: Date string in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        
        Returns:
            DataFrame with wind data or None if unavailable
        """
        try:
            # Parse date string
            flight_date = pd.to_datetime(date_str)
            logger.info(f"Fetching wind data for {flight_date}")
            
            # Calculate bounding box with buffer
            lat_min = min(origin_lat, dest_lat) - 2
            lat_max = max(origin_lat, dest_lat) + 2
            lon_min = min(origin_lon, dest_lon) - 2
            lon_max = max(origin_lon, dest_lon) + 2
            
            logger.info(f"Bounding box: lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]")
            
            # Try to fetch wind data using openap's wind module
            try:
                from openap import wind
                
                # Fetch wind data for the route
                wind_data = wind.get(
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    date=flight_date
                )
                
                if wind_data is not None:
                    logger.info(f"Successfully fetched wind data: {len(wind_data)} points")
                    return wind_data
                else:
                    logger.warning("Wind data fetch returned None")
                    
            except ImportError:
                logger.warning("openap.wind module not available")
            except Exception as e:
                logger.warning(f"Error fetching wind data: {str(e)}")
            
            # Fallback: Generate synthetic wind data for testing
            logger.warning("Using synthetic wind data (no real weather service available)")
            wind_grid = self._generate_synthetic_wind(
                lat_min, lat_max, lon_min, lon_max
            )
            
            return wind_grid
            
        except Exception as e:
            logger.error(f"Error in fetch_wind_grid: {str(e)}")
            return None
    
    def _generate_synthetic_wind(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        grid_size: int = 5
    ) -> pd.DataFrame:
        """
        Generate synthetic wind data for testing purposes.
        
        Args:
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds
            grid_size: Number of grid points per dimension
        
        Returns:
            DataFrame with synthetic wind data
        """
        # Create grid
        lats = np.linspace(lat_min, lat_max, grid_size)
        lons = np.linspace(lon_min, lon_max, grid_size)
        alts = np.array([0, 5000, 10000, 20000, 30000, 40000])  # feet
        
        wind_data = []
        
        # Typical wind patterns: stronger at altitude, westerly component
        for alt in alts:
            # Wind increases with altitude
            wind_speed_base = 10 + (alt / 1000) * 2  # knots
            wind_dir_base = 270  # westerly
            
            for lat in lats:
                for lon in lons:
                    # Add some variation
                    wind_speed = wind_speed_base + np.random.normal(0, 5)
                    wind_dir = wind_dir_base + np.random.normal(0, 20)
                    
                    # Convert to u, v components
                    wind_speed_ms = wind_speed * 0.514444  # knots to m/s
                    wind_dir_rad = np.radians(wind_dir)
                    
                    u = -wind_speed_ms * np.sin(wind_dir_rad)  # East-West component
                    v = -wind_speed_ms * np.cos(wind_dir_rad)  # North-South component
                    
                    wind_data.append({
                        'lat': lat,
                        'lon': lon,
                        'alt': alt * 0.3048,  # feet to meters
                        'u': u,
                        'v': v,
                        'timestamp': 0  # Not time-dependent for synthetic data
                    })
        
        df = pd.DataFrame(wind_data)
        logger.info(f"Generated synthetic wind grid: {len(df)} points")
        
        return df


class FlightOptimizer:
    """
    Optimizes flight trajectories using OpenAP trajectory optimization.
    """

    def __init__(self, aircraft_type: str = 'A320'):
        """
        Initialize the FlightOptimizer.

        Args:
            aircraft_type: ICAO aircraft type code (default: 'A320')
        """
        if not OPENAP_TOP_AVAILABLE:
            raise ImportError("openap.top module not available. Cannot use FlightOptimizer.")

        self.aircraft_type = aircraft_type

        # Initialize WeatherEngine if available
        if WEATHER_ENGINE_AVAILABLE:
            self.weather_engine = WeatherEngine()
            logger.info(f"Initialized FlightOptimizer for {aircraft_type} with WeatherEngine")
        else:
            self.weather_engine = None
            logger.warning(f"Initialized FlightOptimizer for {aircraft_type} without WeatherEngine")

        logger.info(f"Initialized FlightOptimizer for {aircraft_type}")

    def run_optimization(
        self,
        origin: Tuple[float, float],
        dest: Tuple[float, float],
        date_str: str,
        mass_fraction: float = 0.85
    ) -> Dict:
        """
        Run trajectory optimization for a flight.

        Args:
            origin: Tuple of (latitude, longitude) for origin
            dest: Tuple of (latitude, longitude) for destination
            date_str: Date string in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            mass_fraction: Fraction of maximum takeoff mass (default: 0.85)

        Returns:
            Dictionary containing:
                - trajectory: Full result DataFrame with optimized path
                - total_fuel_kg: Total fuel consumption in kg
                - total_time_min: Total flight time in minutes
                - success: Boolean indicating if optimization succeeded
        """
        try:
            origin_lat, origin_lon = origin
            dest_lat, dest_lon = dest

            logger.info(f"\n{'='*60}")
            logger.info(f"Running trajectory optimization")
            logger.info(f"{'='*60}")
            logger.info(f"Origin: ({origin_lat:.4f}, {origin_lon:.4f})")
            logger.info(f"Destination: ({dest_lat:.4f}, {dest_lon:.4f})")
            logger.info(f"Date: {date_str}")
            logger.info(f"Mass fraction: {mass_fraction}")

            # Parse and validate date
            try:
                flight_date = pd.to_datetime(date_str)
                logger.info(f"Parsed date: {flight_date}")
            except Exception as e:
                logger.error(f"Error parsing date '{date_str}': {str(e)}")
                return self._error_result("Invalid date format")

            # Fetch weather data
            logger.info("Fetching weather data...")
            wind_grid = None

            if self.weather_engine is not None:
                try:
                    wind_grid = self.weather_engine.get_weather_grid(
                        origin=(origin_lat, origin_lon),
                        dest=(dest_lat, dest_lon),
                        flight_date=date_str,
                        flight_level=350
                    )
                    rename_map = {
                        'lat': 'latitude',
                        'lon': 'longitude',
                        'alt': 'h',  # Synthetic uses 'alt'
                        'altitude': 'h',  # FastMeteo uses 'altitude'
                        'timestamp': 'ts',
                        'u_component_of_wind': 'u',
                        'v_component_of_wind': 'v'
                    }
                    actual_rename = {k: v for k, v in rename_map.items() if k in wind_grid.columns}
                    wind_grid = wind_grid.rename(columns=actual_rename)
                    wind_grid = wind_grid.dropna()
                    wind_grid = wind_grid[wind_grid['h'] > 100]


                except Exception as e:
                    logger.warning(f"Error fetching weather data: {str(e)}")
                    wind_grid = None
            else:
                logger.warning("WeatherEngine not available, proceeding without weather data")

            # Initialize OpenAP trajectory optimizer
            logger.info(f"Initializing CompleteFlight for {self.aircraft_type}")


            try:

                flight = CompleteFlight(actype=self.aircraft_type, m0=mass_fraction,origin='VOCI',destination='VIDP',)
                logger.info("CompleteFlight initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing CompleteFlight: {str(e)}")
                return self._error_result(f"Failed to initialize optimizer: {str(e)}")

            # Enable wind if available
            if wind_grid is not None and not wind_grid.empty:
                try:
                    logger.info("Enabling wind data for optimization")
                    flight.enable_wind(wind_grid)
                    logger.info("Wind data enabled successfully")
                except Exception as e:
                    logger.warning(f"Could not enable wind data: {str(e)}")
                    logger.info("Proceeding without wind data")
            else:
                logger.warning("No wind data available, optimizing without wind")

            # Run trajectory optimization
            logger.info("Running trajectory optimization (objective: fuel)...")

            try:
                # Format coordinates for OpenAP
                origin_str = f"{origin_lat:.6f},{origin_lon:.6f}"
                dest_str = f"{dest_lat:.6f},{dest_lon:.6f}"

                logger.info(f"Optimizing from {origin_str} to {dest_str}")

                # Run optimization
                trajectory_result = flight.trajectory(
                    origin=origin_str,
                    destination=dest_str,
                    objective='fuel'
                )

                if trajectory_result is None or trajectory_result.empty:
                    logger.error("Trajectory optimization returned empty result")
                    return self._error_result("Optimization failed to produce trajectory")

                logger.info(f"Optimization complete: {len(trajectory_result)} waypoints")

                # Extract results
                result = self._process_trajectory_result(trajectory_result)

                logger.info(f"Total fuel: {result['total_fuel_kg']:.2f} kg")
                logger.info(f"Total time: {result['total_time_min']:.2f} minutes")
                logger.info(f"{'='*60}\n")

                return result

            except Exception as e:
                logger.error(f"Error during trajectory optimization: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return self._error_result(f"Optimization failed: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error in run_optimization: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._error_result(f"Unexpected error: {str(e)}")

    def _process_trajectory_result(self, trajectory: pd.DataFrame) -> Dict:
        """
        Process the trajectory result from OpenAP.

        Args:
            trajectory: DataFrame from CompleteFlight.trajectory()

        Returns:
            Dictionary with processed results
        """
        # Calculate total fuel
        if 'fuelflow' in trajectory.columns and 'ts' in trajectory.columns:
            # Integrate fuel flow over time
            dt = trajectory['ts'].diff().fillna(0)
            fuel_consumed = (trajectory['fuelflow'] * dt).sum()
            total_fuel_kg = fuel_consumed
        elif 'fuel' in trajectory.columns:
            # Use fuel column directly
            total_fuel_kg = trajectory['fuel'].iloc[-1] - trajectory['fuel'].iloc[0]
        else:
            logger.warning("No fuel data found in trajectory, returning 0")
            total_fuel_kg = 0

        # Calculate total time
        if 'ts' in trajectory.columns:
            total_time_sec = trajectory['ts'].iloc[-1] - trajectory['ts'].iloc[0]
            total_time_min = total_time_sec / 60
        elif 't' in trajectory.columns:
            total_time_sec = trajectory['t'].iloc[-1] - trajectory['t'].iloc[0]
            total_time_min = total_time_sec / 60
        else:
            logger.warning("No time data found in trajectory")
            total_time_min = 0

        return {
            'trajectory': trajectory,
            'total_fuel_kg': abs(total_fuel_kg),  # Ensure positive
            'total_time_min': total_time_min,
            'success': True,
            'num_waypoints': len(trajectory)
        }

    def _error_result(self, error_message: str) -> Dict:
        """
        Create an error result dictionary.

        Args:
            error_message: Description of the error

        Returns:
            Dictionary with error information
        """
        return {
            'trajectory': pd.DataFrame(),
            'total_fuel_kg': 0,
            'total_time_min': 0,
            'success': False,
            'error': error_message
        }

    def optimize_multiple_flights(
        self,
        flight_specs: List[Dict]
    ) -> List[Dict]:
        """
        Optimize multiple flights in batch.

        Args:
            flight_specs: List of dictionaries with 'origin', 'dest', 'date_str', 'mass_fraction'

        Returns:
            List of optimization results
        """
        results = []

        for i, spec in enumerate(flight_specs):
            logger.info(f"\nOptimizing flight {i+1}/{len(flight_specs)}")

            result = self.run_optimization(
                origin=spec.get('origin'),
                dest=spec.get('dest'),
                date_str=spec.get('date_str'),
                mass_fraction=spec.get('mass_fraction', 0.85)
            )

            results.append(result)

        return results


def main():
    """
    Example usage of FlightOptimizer.
    """
    optimizer = FlightOptimizer(aircraft_type='A320')

    # Example: Optimize a flight from Mumbai to Kochi
    result = optimizer.run_optimization(
        origin=(19.0896, 72.8656),  # Mumbai (BOM)
        dest=(9.9312, 76.2673),      # Kochi (COK)
        date_str='2025-01-08 14:45:00',
        mass_fraction=0.85
    )

    if result['success']:
        print(f"\n✓ Optimization successful!")
        print(f"  Total fuel: {result['total_fuel_kg']:.2f} kg")
        print(f"  Total time: {result['total_time_min']:.2f} minutes")
        print(f"  Waypoints: {result['num_waypoints']}")

        if not result['trajectory'].empty:
            print(f"\nTrajectory columns: {list(result['trajectory'].columns)}")
            print(f"\nFirst few waypoints:")
            print(result['trajectory'].head())
    else:
        print(f"\n✗ Optimization failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
