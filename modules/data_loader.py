"""
Flight tracking data loader and processor.
Handles CSV data with timestamp, position, altitude, and speed information.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightLoader:
    """
    Loads and processes flight tracking data from CSV files.
    """
    
    def __init__(self):
        """Initialize the FlightLoader."""
        pass
    
    def load_csv(self, filepath: str) -> Dict[str, pd.DataFrame]:
        """
        Load and process flight tracking data from a CSV file.
        
        Args:
            filepath: Path to the CSV file containing flight data
            
        Returns:
            Dictionary where keys are callsigns and values are processed DataFrames
            with columns: timestamp, latitude, longitude, altitude, speed
            
        Expected CSV format:
            Timestamp, UTC, Callsign, Position, Altitude, Speed
            Example: 1736347508, 2025-01-08T14:45:08Z, AIC2886, "10.1543,76.3897", 400, 150
        """
        try:
            # Read the CSV file
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_columns = ['Timestamp', 'UTC', 'Callsign', 'Position', 'Altitude', 'Speed']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and process the data
            df = self._clean_data(df)
            
            # Filter out ground data (altitude < 1000 ft)
            logger.info(f"Filtering ground data (altitude < 1000 ft)")
            initial_count = len(df)
            df = df[df['altitude'] >= 1000]
            filtered_count = initial_count - len(df)
            logger.info(f"Removed {filtered_count} ground data points")
            
            # Group by callsign and process each flight
            result = {}
            callsigns = df['Callsign'].unique()
            logger.info(f"Processing {len(callsigns)} unique callsigns")
            
            for callsign in callsigns:
                flight_data = df[df['Callsign'] == callsign].copy()
                
                # Sort by timestamp
                flight_data = flight_data.sort_values('timestamp')
                
                # Resample to 1-minute intervals
                resampled_data = self._resample_flight_data(flight_data)
                
                if not resampled_data.empty:
                    # Select and rename columns for output
                    resampled_data = resampled_data[['timestamp', 'latitude', 'longitude', 'altitude', 'speed']]
                    result[callsign] = resampled_data
                    logger.info(f"Processed {callsign}: {len(resampled_data)} data points after resampling")
                else:
                    logger.warning(f"No data remaining for {callsign} after processing")
            
            logger.info(f"Successfully processed {len(result)} flights")
            return result
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform the raw data.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            Cleaned DataFrame with split position and proper data types
        """
        df = df.copy()
        
        # Parse Position column into latitude and longitude
        logger.info("Parsing Position column into latitude and longitude")
        position_data = df['Position'].apply(self._parse_position)
        df['latitude'] = position_data.apply(lambda x: x[0] if x is not None else np.nan)
        df['longitude'] = position_data.apply(lambda x: x[1] if x is not None else np.nan)
        
        # Drop rows with invalid position data
        invalid_positions = df[['latitude', 'longitude']].isna().any(axis=1).sum()
        if invalid_positions > 0:
            logger.warning(f"Dropping {invalid_positions} rows with invalid position data")
            df = df.dropna(subset=['latitude', 'longitude'])
        
        # Rename columns to lowercase for consistency
        df = df.rename(columns={
            'Timestamp': 'timestamp',
            'Callsign': 'Callsign',  # Keep Callsign for grouping
            'Altitude': 'altitude',
            'Speed': 'speed',
            'UTC': 'utc'
        })
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # Handle missing or invalid values
        df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        
        # Drop rows with critical missing values
        df = df.dropna(subset=['datetime', 'altitude', 'speed'])
        
        # Drop the original Position column
        df = df.drop(columns=['Position'])
        
        return df
    
    def _parse_position(self, position_str):
        """
        Parse the Position string into latitude and longitude.
        
        Args:
            position_str: String in format "lat,lon" (e.g., "10.1543,76.3897")
            
        Returns:
            Tuple of (latitude, longitude) or None if parsing fails
        """
        try:
            if pd.isna(position_str):
                return None
            
            # Remove quotes and whitespace
            position_str = str(position_str).strip().strip('"').strip("'")
            
            # Split by comma
            parts = position_str.split(',')
            if len(parts) != 2:
                logger.warning(f"Invalid position format: {position_str}")
                return None
            
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            
            # Validate latitude and longitude ranges
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                logger.warning(f"Position out of valid range: lat={lat}, lon={lon}")
                return None
            
            return (lat, lon)
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing position '{position_str}': {str(e)}")
            return None
    
    def _resample_flight_data(self, flight_data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample flight data to 1-minute intervals.
        
        Args:
            flight_data: DataFrame for a single flight
            
        Returns:
            Resampled DataFrame with 1-minute intervals
        """
        if flight_data.empty:
            return flight_data
        
        # Set datetime as index for resampling
        flight_data = flight_data.set_index('datetime')
        
        # Resample to 1-minute intervals
        # Use mean for numeric values, first for categorical
        resampled = flight_data.resample('1min').agg({
            'timestamp': 'first',  # Keep the first timestamp in the interval
            'latitude': 'mean',
            'longitude': 'mean',
            'altitude': 'mean',
            'speed': 'mean'
        })
        
        # Drop rows where all values are NaN (gaps in data)
        resampled = resampled.dropna(how='all')
        
        # Forward fill small gaps (up to 2 minutes)
        resampled = resampled.ffill(limit=2)
        
        # Drop any remaining NaN values
        resampled = resampled.dropna()
        
        # Reset index to have datetime as a column
        resampled = resampled.reset_index()
        
        # Update timestamp to match resampled datetime
        resampled['timestamp'] = resampled['datetime'].astype(np.int64) // 10**9
        
        # Drop the datetime column as we have timestamp
        resampled = resampled.drop(columns=['datetime'])
        
        return resampled


def main():
    """
    Example usage of FlightLoader.
    """
    loader = FlightLoader()
    
    
    flights = loader.load_csv('data/AI2886_38a1bfd2.csv')
    
    for callsign, data in flights.items():
        print(f"\n{callsign}:")
        print(f"  Number of points: {len(data)}")
        print(f"  Duration: {data['timestamp'].max() - data['timestamp'].min()} seconds")
        print(f"  Altitude range: {data['altitude'].min():.0f} - {data['altitude'].max():.0f} ft")
        print(data.head())
    
    print("FlightLoader module loaded successfully")


if __name__ == "__main__":
    main()
