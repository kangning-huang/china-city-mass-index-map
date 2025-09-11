#!/usr/bin/env python3
"""
Create interactive dual-layer map for China City Mass Index.

This script:
1. Loads city CMI statistics from step 07
2. Loads neighborhood CMI multipliers from step 07
3. Loads spatial boundaries (city boundaries and H3 grids)
4. Creates dual-layer interactive map with Folium
5. Saves final interactive HTML map

Input:
- City CMI statistics from step 07
- Neighborhood CMI multipliers from step 07
- City boundary geometries
- H3 grid geometries
- China boundary for overlay

Output:
- Interactive HTML map with dual layers

Author: Generated for China City Mass Index Map
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *
from utils.visualization_utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_interactive_map(
    city_stats_file: Optional[str] = None,
    neighborhood_file: Optional[str] = None,
    city_boundaries_file: Optional[str] = None,
    h3_boundaries_file: Optional[str] = None,
    china_boundary_file: Optional[str] = None,
    debug: bool = False,
    country_code: Optional[str] = None,
    output_file: Optional[str] = None
) -> bool:
    """
    Create interactive dual-layer map.
    
    Args:
        city_stats_file: Path to city CMI statistics
        neighborhood_file: Path to neighborhood multipliers
        city_boundaries_file: Path to city boundary geometries
        h3_boundaries_file: Path to H3 grid geometries
        china_boundary_file: Path to China boundary
        debug: Use debug mode
        country_code: Country code filter
        output_file: Custom output filename
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("CREATING INTERACTIVE DUAL-LAYER MAP")
        logger.info("=" * 70)
        
        # Find input files if not specified
        if city_stats_file is None:
            stats_dir = RESULTS_DIR / "statistics"
            if not stats_dir.exists():
                stats_dir = OUTPUT_DIRS["statistics"]
            
            city_files = list(stats_dir.glob("**/china_city_statistics.csv"))
            if city_files:
                city_stats_file = city_files[0]
                logger.info(f"Using city stats file: {city_stats_file}")
            else:
                logger.error("No city statistics files found. Run step 07 first.")
                return False
        
        if neighborhood_file is None:
            neighborhood_files = list(stats_dir.glob("**/china_neighborhoods_cmi.csv"))
            if neighborhood_files:
                neighborhood_file = neighborhood_files[0]
                logger.info(f"Using neighborhood file: {neighborhood_file}")
            else:
                logger.error("No neighborhood files found. Run step 07 first.")
                return False
        
        # Load statistical data
        logger.info("Loading statistical analysis results...")
        city_stats = pd.read_csv(city_stats_file)
        neighborhood_stats = pd.read_csv(neighborhood_file)
        
        logger.info(f"Loaded {len(city_stats)} city statistics")
        logger.info(f"Loaded {len(neighborhood_stats)} neighborhood statistics")
        
        # Filter for China if specified
        if country_code:
            city_stats = city_stats[city_stats['CTR_MN_ISO'] == country_code.upper()]
            neighborhood_stats = neighborhood_stats[neighborhood_stats['CTR_MN_ISO'] == country_code.upper()]
            logger.info(f"Filtered to {len(city_stats)} cities and {len(neighborhood_stats)} neighborhoods in {country_code}")
        
        if debug:
            # Use subset for faster testing
            city_stats = city_stats.head(20)
            city_ids = city_stats['ID_HDC_G0'].unique()
            neighborhood_stats = neighborhood_stats[neighborhood_stats['ID_HDC_G0'].isin(city_ids)]
            logger.info(f"Debug mode: Using {len(city_stats)} cities and {len(neighborhood_stats)} neighborhoods")
        
        # Load spatial boundaries
        logger.info("Loading spatial boundary data...")
        
        # Try to find boundary files
        boundary_sources = [
            OUTPUT_DIRS.get("boundaries", Path(".")),
            RESULTS_DIR / "shapefiles",
            DATA_DIR / "processed" / "boundaries"
        ]
        
        city_boundaries_gdf = None
        h3_boundaries_gdf = None
        
        # Look for city boundaries
        for source_dir in boundary_sources:
            if source_dir.exists():
                boundary_files = list(source_dir.glob("*cities_boundaries*"))
                if boundary_files:
                    city_boundaries_file = boundary_files[0]
                    break
        
        if city_boundaries_file and Path(city_boundaries_file).exists():
            logger.info(f"Loading city boundaries from: {city_boundaries_file}")
            city_boundaries_gdf = gpd.read_file(city_boundaries_file)
            # Merge with city statistics
            city_boundaries_gdf = city_boundaries_gdf.merge(
                city_stats, on='ID_HDC_G0', how='inner'
            )
            logger.info(f"Loaded {len(city_boundaries_gdf)} city boundaries with statistics")
        else:
            logger.warning("City boundary file not found, creating synthetic boundaries from city points")
            # Create point geometries from city statistics if coordinates available
            if all(col in city_stats.columns for col in ['longitude', 'latitude']):
                from shapely.geometry import Point
                city_boundaries_gdf = city_stats.copy()
                city_boundaries_gdf['geometry'] = city_boundaries_gdf.apply(
                    lambda row: Point(row['longitude'], row['latitude']), axis=1
                )
                city_boundaries_gdf = gpd.GeoDataFrame(city_boundaries_gdf, crs='EPSG:4326')
            else:
                logger.error("No city boundary data available and no coordinates in city statistics")
                return False
        
        # Look for H3 boundaries
        for source_dir in boundary_sources:
            if source_dir.exists():
                h3_files = list(source_dir.glob("*h3_grids*"))
                if h3_files:
                    h3_boundaries_file = h3_files[0]
                    break
        
        if h3_boundaries_file and Path(h3_boundaries_file).exists():
            logger.info(f"Loading H3 boundaries from: {h3_boundaries_file}")
            h3_boundaries_gdf = gpd.read_file(h3_boundaries_file)
            
            # Merge with neighborhood statistics
            h3_boundaries_gdf = h3_boundaries_gdf.merge(
                neighborhood_stats, on='h3index', how='inner'
            )
            logger.info(f"Loaded {len(h3_boundaries_gdf)} H3 boundaries with statistics")
        else:
            logger.warning("H3 boundary file not found, skipping neighborhood layer")
            h3_boundaries_gdf = None
        
        # Load China boundary
        china_boundary_gdf = None
        if china_boundary_file and Path(china_boundary_file).exists():
            logger.info(f"Loading China boundary from: {china_boundary_file}")
            china_boundary_gdf = gpd.read_file(china_boundary_file)
        else:
            # Look for China boundary in data directory
            china_files = list(DATA_DIR.glob("**/China*.gpkg")) + list(DATA_DIR.glob("**/China*.geojson"))
            if china_files:
                china_boundary_gdf = gpd.read_file(china_files[0])
                logger.info(f"Found China boundary: {china_files[0]}")
        
        # Create the interactive map
        logger.info("Creating interactive map...")
        
        # Initialize map centered on China
        center_lat, center_lon = 36.0, 104.0  # Center of China
        m = create_folium_map(center_lat, center_lon, zoom_start=5, tiles='CartoDB positron')
        
        # Add China boundary if available
        if china_boundary_gdf is not None:
            folium.GeoJson(
                china_boundary_gdf,
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0
                },
                name='China Boundary'
            ).add_to(m)
            logger.info("Added China boundary to map")
        
        # Add city layer
        if city_boundaries_gdf is not None and not city_boundaries_gdf.empty:
            logger.info("Adding city CMI layer...")
            
            # Calculate point sizes and colors
            radii, areas, area_stats = calculate_point_sizes(
                city_boundaries_gdf, 
                min_radius=5, 
                max_radius=25
            )
            
            # Create city layer
            city_layer = folium.FeatureGroup(name="City CMI", show=True)
            
            # Get CMI values for coloring
            cmi_values = city_boundaries_gdf['true_city_cmi'].fillna(1.0)
            cmi_min, cmi_max = cmi_values.min(), cmi_values.max()
            
            for idx, row in city_boundaries_gdf.iterrows():
                if pd.isna(row.geometry):
                    continue
                
                # Get coordinates (centroid if polygon, direct if point)
                if row.geometry.geom_type == 'Point':
                    lat, lon = row.geometry.y, row.geometry.x
                elif hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    lat, lon = centroid.y, centroid.x
                else:
                    continue
                
                cmi_value = row.get('true_city_cmi', 1.0)
                if pd.isna(cmi_value):
                    cmi_value = 1.0
                
                # Get color based on CMI value
                color = get_continuous_color(cmi_value, cmi_min, cmi_max, 'seismic')
                
                # Calculate marker size
                radius = radii.iloc[idx] if idx < len(radii) else 8
                
                # Create popup data
                popup_data = {
                    'City': row.get('UC_NM_MN', 'Unknown'),
                    'Country': row.get('CTR_MN_NM', 'Unknown'),
                    'City CMI': f"{cmi_value:.3f}",
                    'Population': f"{row.get('population', 0):,.0f}",
                    'Material Mass': f"{row.get('total_material_mass_tonnes', 0):,.0f} tonnes"
                }
                
                popup = create_popup_html(popup_data)
                
                # Add marker
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=popup,
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.8,
                    tooltip=f"{row.get('UC_NM_MN', 'Unknown')}: CMI = {cmi_value:.3f}"
                ).add_to(city_layer)
            
            city_layer.add_to(m)
            logger.info(f"Added {len(city_boundaries_gdf)} city markers")
        
        # Add neighborhood layer
        if h3_boundaries_gdf is not None and not h3_boundaries_gdf.empty:
            logger.info("Adding neighborhood multiplier layer...")
            
            # Create neighborhood layer
            neighborhood_layer = folium.FeatureGroup(name="Neighborhood Multipliers", show=False)
            
            # Get multiplier values for coloring
            multiplier_values = h3_boundaries_gdf['neighborhood_multiplier'].fillna(1.0)
            
            # Filter extreme outliers for better visualization
            p5, p95 = np.percentile(multiplier_values, [5, 95])
            
            for idx, row in h3_boundaries_gdf.iterrows():
                if pd.isna(row.geometry):
                    continue
                
                multiplier = row.get('neighborhood_multiplier', 1.0)
                if pd.isna(multiplier):
                    multiplier = 1.0
                
                # Determine if outlier
                is_low_outlier = multiplier < p5
                is_high_outlier = multiplier > p95
                
                # Get color based on multiplier value
                color = get_continuous_multiplier_color(multiplier, is_low_outlier, is_high_outlier)
                
                # Create popup data
                popup_data = {
                    'City': row.get('UC_NM_MN', 'Unknown'),
                    'H3 Index': row.get('h3index', '')[:8] + '...',
                    'Multiplier': f"{multiplier:.3f}",
                    'Population': f"{row.get('population', 0):,.0f}",
                    'Material Mass': f"{row.get('total_material_mass_tonnes', 0):,.0f} tonnes"
                }
                
                popup = create_popup_html(popup_data)
                
                # Add hexagon
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.7
                    },
                    popup=popup,
                    tooltip=f"Multiplier: {multiplier:.3f}"
                ).add_to(neighborhood_layer)
            
            neighborhood_layer.add_to(m)
            logger.info(f"Added {len(h3_boundaries_gdf)} neighborhood hexagons")
        
        # Add layer control and other features
        add_layer_control(m)
        add_fullscreen_button(m)
        
        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            if debug:
                output_file = f"china_dual_layer_cmi_multiplier_map_debug_{timestamp}.html"
            else:
                output_file = get_timestamped_filename("final_map", timestamp=timestamp)
        
        # Save the map
        output_path = OUTPUT_DIRS["maps"] / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving interactive map to: {output_path}")
        
        # Save with error handling
        success = save_map_safely(m, output_path, use_temp_location=True)
        
        if success:
            # Also create index.html in project root
            index_path = BASE_DIR / "index.html"
            logger.info(f"Creating index.html at: {index_path}")
            save_map_safely(m, index_path, use_temp_location=False)
        
        # Summary
        logger.info("=" * 70)
        logger.info("INTERACTIVE MAP CREATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Map saved to: {output_path}")
        logger.info(f"Index file: {index_path}")
        
        if city_boundaries_gdf is not None:
            logger.info(f"City layer: {len(city_boundaries_gdf)} cities")
            china_cities = len(city_boundaries_gdf[city_boundaries_gdf['CTR_MN_ISO'] == 'CHN'])
            logger.info(f"  China cities: {china_cities}")
        
        if h3_boundaries_gdf is not None:
            logger.info(f"Neighborhood layer: {len(h3_boundaries_gdf)} hexagons")
            outliers = len(h3_boundaries_gdf[
                (h3_boundaries_gdf['neighborhood_multiplier'] < p5) | 
                (h3_boundaries_gdf['neighborhood_multiplier'] > p95)
            ])
            logger.info(f"  Outliers (5th-95th percentile): {outliers}")
        
        if success:
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"Final map size: {file_size_mb:.1f} MB")
            logger.info("Interactive map creation completed successfully!")
            return True
        else:
            logger.error("Failed to save interactive map")
            return False
        
    except Exception as e:
        logger.error(f"Error creating interactive map: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main(debug: bool = False, country_code: Optional[str] = None, **kwargs) -> bool:
    """Main function for command line execution."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = create_interactive_map(
            debug=debug,
            country_code=country_code,
            **kwargs
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create interactive dual-layer map')
    parser.add_argument('--city-stats', type=str, help='Path to city statistics file')
    parser.add_argument('--neighborhood-file', type=str, help='Path to neighborhood data file')
    parser.add_argument('--city-boundaries', type=str, help='Path to city boundary geometries')
    parser.add_argument('--h3-boundaries', type=str, help='Path to H3 grid geometries')
    parser.add_argument('--china-boundary', type=str, help='Path to China boundary')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process subset of data')
    parser.add_argument('--output', type=str, help='Custom output filename')
    
    args = parser.parse_args()
    
    success = main(
        debug=args.debug,
        country_code=args.country,
        city_stats_file=args.city_stats,
        neighborhood_file=args.neighborhood_file,
        city_boundaries_file=args.city_boundaries,
        h3_boundaries_file=args.h3_boundaries,
        china_boundary_file=args.china_boundary,
        output_file=args.output
    )
    
    exit(0 if success else 1)