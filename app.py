"""
Streamlit app for visualizing flight path optimization results.
Compares historical flight tracks with AI-optimized trajectories.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pydeck as pdk
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Flight Path Optimizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_results(results_dir: str = 'results'):
    """
    Load benchmark results and flight paths.
    
    Args:
        results_dir: Directory containing results files
    
    Returns:
        Tuple of (results_df, flight_paths_dict)
    """
    results_path = Path(results_dir)
    
    try:
        # Load results CSV
        csv_file = results_path / 'results.csv'
        if not csv_file.exists():
            return None, None
        
        results_df = pd.read_csv(csv_file)
        
        # Load flight paths JSON
        json_file = results_path / 'flight_paths.json'
        if not json_file.exists():
            return results_df, None
        
        with open(json_file, 'r') as f:
            flight_paths = json.load(f)
        
        return results_df, flight_paths
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return None, None


def create_path_layer(path_data, color, width=3, name="Path"):
    """
    Create a PyDeck PathLayer.
    
    Args:
        path_data: List of [lon, lat] coordinates
        color: RGB color tuple (e.g., [255, 0, 0])
        width: Line width in pixels
        name: Layer name
    
    Returns:
        pdk.Layer
    """
    return pdk.Layer(
        "PathLayer",
        data=[{"path": path_data}],
        get_path="path",
        get_color=color,
        width_min_pixels=width,
        pickable=True,
        auto_highlight=True,
    )


def create_scatterplot_layer(points, color, radius=5000, name="Points"):
    """
    Create a PyDeck ScatterplotLayer.
    
    Args:
        points: List of dictionaries with 'lat', 'lon' keys
        color: RGB color tuple
        radius: Point radius in meters
        name: Layer name
    
    Returns:
        pdk.Layer
    """
    return pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position=["lon", "lat"],
        get_color=color,
        get_radius=radius,
        pickable=True,
        auto_highlight=True,
    )


def create_wind_layer(wind_data, altitude_filter=30000):
    """
    Create a wind vector layer using LineLayer.
    
    Args:
        wind_data: DataFrame with wind information
        altitude_filter: Altitude level to display (in feet)
    
    Returns:
        pdk.Layer or None
    """
    if wind_data is None or wind_data.empty:
        return None
    
    # Filter wind data by altitude
    # Convert altitude from meters to feet if needed
    if 'alt' in wind_data.columns:
        wind_filtered = wind_data[
            abs(wind_data['alt'] * 3.28084 - altitude_filter) < 5000
        ].copy()
    else:
        wind_filtered = wind_data.copy()
    
    if wind_filtered.empty:
        return None
    
    # Create line segments for wind vectors
    wind_lines = []
    scale = 0.05  # Scale factor for wind vector visualization
    
    for _, row in wind_filtered.iterrows():
        if 'u' in row and 'v' in row:
            start_lon = row['lon']
            start_lat = row['lat']
            
            # Calculate end point based on wind components
            end_lon = start_lon + row['u'] * scale
            end_lat = start_lat + row['v'] * scale
            
            wind_lines.append({
                "start": [start_lon, start_lat],
                "end": [end_lon, end_lat]
            })
    
    if not wind_lines:
        return None
    
    return pdk.Layer(
        "LineLayer",
        data=wind_lines,
        get_source_position="start",
        get_target_position="end",
        get_color=[100, 150, 255, 180],  # Light blue
        get_width=2,
        pickable=False,
    )


def main():
    """Main Streamlit app."""
    
    # Title and description
    st.title("✈️ Flight Path Optimization Dashboard")
    st.markdown("""
    Compare historical flight tracks (The Pilot 🧑‍✈️) with AI-optimized trajectories (The AI 🤖).
    Visualize fuel savings potential and route differences.
    """)
    
    # Load results
    results_df, flight_paths = load_results('results')
    
    if results_df is None or flight_paths is None:
        st.error("❌ No results found. Please run the benchmark first:")
        st.code("python -m modules.benchmark data/your_flight_data.csv results")
        st.info("Expected files:\n- results/results.csv\n- results/flight_paths.json")
        return
    
    # Sidebar
    st.sidebar.header("📊 Flight Selection")
    
    # Dropdown to select callsign
    callsigns = results_df['Callsign'].tolist()
    selected_callsign = st.sidebar.selectbox(
        "Select Flight Callsign:",
        callsigns,
        help="Choose a flight to visualize"
    )
    
    # Get selected flight data
    flight_info = results_df[results_df['Callsign'] == selected_callsign].iloc[0]
    flight_path_data = flight_paths.get(selected_callsign, {})
    
    # Sidebar metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Flight Information")
    st.sidebar.metric("Date", flight_info['Date'])
    st.sidebar.metric("Distance", f"{flight_info['Distance_km']:.1f} km")
    st.sidebar.metric("Duration", f"{flight_info['Duration_hours']:.2f} hrs")
    st.sidebar.metric("Avg Altitude", f"{flight_info['Avg_Altitude_ft']:.0f} ft")
    st.sidebar.metric("Avg Speed", f"{flight_info['Avg_Speed_kts']:.0f} kts")
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🧑‍✈️ Pilot's Fuel Burn",
            f"{flight_info['Historical_Fuel_kg']:.0f} kg",
            help="Actual fuel consumption from historical data"
        )
    
    with col2:
        st.metric(
            "🤖 AI's Estimated Burn",
            f"{flight_info['Optimal_Fuel_kg']:.0f} kg",
            delta=f"-{flight_info['Savings_kg']:.0f} kg",
            delta_color="inverse",
            help="Optimized fuel consumption"
        )
    
    with col3:
        st.metric(
            "💰 Fuel Savings",
            f"{flight_info['Savings_percent']:.1f}%",
            f"{flight_info['Savings_kg']:.0f} kg saved",
            help="Potential fuel savings with optimization"
        )
    
    st.markdown("---")
    
    # Map visualization
    st.subheader("🗺️ Flight Path Comparison")
    
    # Extract path data
    actual_path = flight_path_data.get('actual_path', [])
    optimal_path = flight_path_data.get('optimal_path', [])
    origin = flight_path_data.get('origin', {})
    destination = flight_path_data.get('destination', {})
    
    if not actual_path:
        st.warning("⚠️ No path data available for visualization")
        return
    
    # Prepare path data for PyDeck (needs [lon, lat] format)
    actual_path_coords = [[p['lon'], p['lat']] for p in actual_path]
    optimal_path_coords = [[p['lon'], p['lat']] for p in optimal_path] if optimal_path else []
    
    # Calculate map center
    all_lats = [p['lat'] for p in actual_path]
    all_lons = [p['lon'] for p in actual_path]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    # Calculate zoom level based on path extent
    lat_range = max(all_lats) - min(all_lats)
    lon_range = max(all_lons) - min(all_lons)
    max_range = max(lat_range, lon_range)
    
    if max_range > 10:
        zoom = 5
    elif max_range > 5:
        zoom = 6
    elif max_range > 2:
        zoom = 7
    else:
        zoom = 8
    
    # Create layers
    layers = []
    
    # Layer 1: Historical track (RED) - The Pilot
    historical_layer = create_path_layer(
        actual_path_coords,
        color=[255, 0, 0, 200],  # Red
        width=4,
        name="Historical Track"
    )
    layers.append(historical_layer)
    
    # Layer 2: Optimized trajectory (GREEN) - The AI
    if optimal_path_coords:
        optimized_layer = create_path_layer(
            optimal_path_coords,
            color=[0, 255, 0, 200],  # Green
            width=4,
            name="Optimized Track"
        )
        layers.append(optimized_layer)
    
    # Add origin and destination markers
    if origin and destination:
        waypoints = [
            {"lat": origin['lat'], "lon": origin['lon'], "type": "Origin"},
            {"lat": destination['lat'], "lon": destination['lon'], "type": "Destination"}
        ]
        
        origin_layer = create_scatterplot_layer(
            [waypoints[0]],
            color=[0, 200, 0, 255],  # Green
            radius=15000,
            name="Origin"
        )
        
        dest_layer = create_scatterplot_layer(
            [waypoints[1]],
            color=[200, 0, 0, 255],  # Red
            radius=15000,
            name="Destination"
        )
        
        layers.append(origin_layer)
        layers.append(dest_layer)
    
    # Layer 3: Wind vectors (optional - placeholder)
    # Wind data integration would go here
    # wind_layer = create_wind_layer(wind_data)
    # if wind_layer:
    #     layers.append(wind_layer)
    
    # Create PyDeck map
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        bearing=0
    )
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={
            "text": "Flight Path"
        }
    )
    
    # Display map
    st.pydeck_chart(deck)
    
    # Legend
    st.markdown("""
    **Legend:**
    - 🔴 **Red Path**: Historical flight track (The Pilot)
    - 🟢 **Green Path**: AI-optimized trajectory (The AI)
    - 🟢 **Green Marker**: Origin airport
    - 🔴 **Red Marker**: Destination airport
    """)
    
    # Additional details
    with st.expander("📊 Detailed Flight Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Historical Flight")
            st.write(f"**Fuel Consumed:** {flight_info['Historical_Fuel_kg']:.2f} kg")
            st.write(f"**Data Points:** {flight_info['Num_Points']}")
            st.write(f"**Origin:** ({flight_info['Origin_Lat']:.4f}, {flight_info['Origin_Lon']:.4f})")
            st.write(f"**Destination:** ({flight_info['Dest_Lat']:.4f}, {flight_info['Dest_Lon']:.4f})")
        
        with col2:
            st.markdown("### Optimized Flight")
            st.write(f"**Fuel Consumed:** {flight_info['Optimal_Fuel_kg']:.2f} kg")
            st.write(f"**Fuel Saved:** {flight_info['Savings_kg']:.2f} kg ({flight_info['Savings_percent']:.2f}%)")
            st.write(f"**Waypoints:** {len(optimal_path) if optimal_path else 'N/A'}")
            
            if flight_info['Savings_kg'] > 0:
                # Calculate CO2 savings (approximate: 1 kg jet fuel ≈ 3.15 kg CO2)
                co2_saved = flight_info['Savings_kg'] * 3.15
                st.write(f"**CO₂ Reduced:** ~{co2_saved:.0f} kg")
    
    # Summary table
    with st.expander("📋 All Flights Summary"):
        st.dataframe(
            results_df[[
                'Callsign', 'Date', 'Distance_km', 'Duration_hours',
                'Historical_Fuel_kg', 'Optimal_Fuel_kg', 'Savings_kg', 'Savings_percent'
            ]].style.format({
                'Distance_km': '{:.1f}',
                'Duration_hours': '{:.2f}',
                'Historical_Fuel_kg': '{:.0f}',
                'Optimal_Fuel_kg': '{:.0f}',
                'Savings_kg': '{:.0f}',
                'Savings_percent': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Overall statistics
        st.markdown("### 📈 Overall Statistics")
        total_historical = results_df['Historical_Fuel_kg'].sum()
        total_optimal = results_df['Optimal_Fuel_kg'].sum()
        total_savings = results_df['Savings_kg'].sum()
        avg_savings_pct = results_df['Savings_percent'].mean()
        
        cols = st.columns(4)
        cols[0].metric("Total Flights", len(results_df))
        cols[1].metric("Total Historical Fuel", f"{total_historical:,.0f} kg")
        cols[2].metric("Total Optimal Fuel", f"{total_optimal:,.0f} kg")
        cols[3].metric("Total Savings", f"{total_savings:,.0f} kg ({avg_savings_pct:.1f}%)")


if __name__ == "__main__":
    main()
