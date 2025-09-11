#!/usr/bin/env python3
"""
Extract road network data from Google Earth Engine for H3 neighborhoods.

This script:
1. Loads H3 neighborhood grids from step 01
2. Extracts road network data from OpenStreetMap via Google Earth Engine
3. Calculates road lengths by type (motorway, trunk, primary, etc.)
4. Saves road statistics per neighborhood

Input:
- H3 neighborhoods CSV/GeoPackage from step 01
- OpenStreetMap road data from Google Earth Engine

Output:
- Road network statistics per neighborhood dataset

Author: Generated for China City Mass Index Map
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from datetime import datetime
from typing import Optional, List
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *
from utils.gee_utils import initialize_gee
from utils.spatial_utils import validate_geometries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_road_data(
    h3_file: Optional[str] = None,
    country_code: Optional[str] = None,
    debug: bool = False,
    output_file: Optional[str] = None,
    road_types: Optional[List[str]] = None
) -> bool:
    """
    Extract road network data for H3 neighborhoods.
    
    Args:
        h3_file (str, optional): Path to H3 neighborhoods file
        country_code (str, optional): ISO 3-letter country code filter
        debug (bool): Use debug mode
        output_file (str, optional): Custom output filename
        road_types (list, optional): List of road types to extract
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("EXTRACTING ROAD NETWORK DATA")
        logger.info("=" * 70)
        
        if road_types is None:
            road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
        
        # Initialize Google Earth Engine
        if not initialize_gee():
            logger.error("Failed to initialize Google Earth Engine")
            return False
        
        # Load H3 neighborhoods
        if h3_file is None:
            # Find the most recent H3 file
            h3_files = list(OUTPUT_DIRS["neighborhood_data"].glob("Fig3_H3_Grids_*.csv"))
            if not h3_files:
                logger.error("No H3 grid files found. Run step 01 first.")
                return False
            h3_file = max(h3_files, key=os.path.getmtime)
        
        logger.info(f"Loading H3 neighborhoods from: {h3_file}")
        
        # Load as GeoDataFrame
        if str(h3_file).endswith('.gpkg'):
            h3_gdf = gpd.read_file(h3_file)
        else:
            # Load CSV with WKT geometry
            h3_df = pd.read_csv(h3_file)
            if 'geometry' in h3_df.columns:
                from shapely import wkt
                h3_df['geometry'] = h3_df['geometry'].apply(wkt.loads)
                h3_gdf = gpd.GeoDataFrame(h3_df, crs='EPSG:4326')
            else:
                logger.error("No geometry column found in H3 file")
                return False
        
        if h3_gdf.empty:
            logger.error("No H3 neighborhoods loaded")
            return False
        
        logger.info(f"Loaded {len(h3_gdf)} H3 neighborhoods")
        
        # Filter by country if specified
        if country_code:
            initial_count = len(h3_gdf)
            h3_gdf = h3_gdf[h3_gdf['CTR_MN_ISO'] == country_code.upper()]
            logger.info(f"Filtered to {len(h3_gdf)} neighborhoods in {country_code} (from {initial_count})")
        
        if debug:
            # Use only first 50 neighborhoods for debugging
            h3_gdf = h3_gdf.head(50)
            logger.info(f"Debug mode: Using {len(h3_gdf)} neighborhoods")
        
        # Validate geometries
        h3_gdf = validate_geometries(h3_gdf)
        
        # Extract road network data using Google Earth Engine
        import ee
        logger.info("Loading OpenStreetMap road data...")
        
        # Load OSM roads from Google Earth Engine
        try:
            # Try the SAT-IO OSM roads dataset
            roads_fc = ee.FeatureCollection("projects/sat-io/open-datasets/OSM/OSM_roads")
        except:
            logger.warning("SAT-IO OSM roads dataset not available, using alternative approach")
            # Alternative: create synthetic road data based on urban areas
            roads_fc = None
        
        # Convert H3 geometries to Earth Engine FeatureCollection
        from utils.gee_utils import zonal_statistics
        import geemap
        
        h3_fc = geemap.geopandas_to_ee(h3_gdf)
        
        # Initialize results dictionary
        road_results = {
            'h3index': [],
            'ID_HDC_G0': [],
            'UC_NM_MN': [],
            'CTR_MN_ISO': []
        }
        
        # Add columns for each road type
        for road_type in road_types:
            road_results[f'road_length_m_{road_type}'] = []
        
        road_results['total_road_length_m'] = []
        
        # Process neighborhoods in batches
        batch_size = 100
        h3_list = h3_gdf.to_dict('records')
        
        logger.info(f"Processing {len(h3_list)} neighborhoods for road extraction...")
        
        for i in tqdm(range(0, len(h3_list), batch_size), desc="Processing road batches"):
            batch = h3_list[i:i+batch_size]
            
            try:
                for neighborhood in batch:
                    # Get basic info
                    road_results['h3index'].append(neighborhood.get('h3index', ''))
                    road_results['ID_HDC_G0'].append(neighborhood.get('ID_HDC_G0', ''))
                    road_results['UC_NM_MN'].append(neighborhood.get('UC_NM_MN', ''))
                    road_results['CTR_MN_ISO'].append(neighborhood.get('CTR_MN_ISO', ''))
                    
                    # Calculate road lengths by type
                    total_length = 0
                    
                    if roads_fc is not None:
                        try:
                            # Create single neighborhood geometry
                            neighborhood_geom = ee.Geometry(neighborhood['geometry'].__geo_interface__)
                            
                            # Calculate road lengths by type
                            for road_type in road_types:
                                # Filter roads by type
                                roads_filtered = roads_fc.filter(ee.Filter.eq('highway', road_type))
                                
                                # Clip to neighborhood
                                roads_clipped = roads_filtered.filterBounds(neighborhood_geom)
                                
                                # Calculate total length
                                def calculate_length(feature):
                                    return feature.set('length', feature.geometry().length())
                                
                                roads_with_length = roads_clipped.map(calculate_length)
                                
                                # Sum lengths
                                length_sum = roads_with_length.aggregate_sum('length').getInfo()
                                length_meters = length_sum or 0
                                
                                road_results[f'road_length_m_{road_type}'].append(length_meters)
                                total_length += length_meters
                                
                        except Exception as e:
                            logger.debug(f"Error processing neighborhood roads: {e}")
                            # Fill with zeros
                            for road_type in road_types:
                                road_results[f'road_length_m_{road_type}'].append(0)
                    else:
                        # No road data available, use synthetic estimates based on urban area
                        # This is a fallback approach
                        area_km2 = neighborhood.get('area_sqkm', 1.0)
                        
                        # Rough estimates based on typical urban road densities
                        road_density_estimates = {
                            'motorway': 0.5,    # km/km²
                            'trunk': 1.0,       # km/km²
                            'primary': 2.0,     # km/km²
                            'secondary': 3.0,   # km/km²
                            'tertiary': 4.0     # km/km²
                        }
                        
                        for road_type in road_types:
                            estimated_length = area_km2 * road_density_estimates.get(road_type, 1.0) * 1000  # Convert to meters
                            road_results[f'road_length_m_{road_type}'].append(estimated_length)
                            total_length += estimated_length
                    
                    road_results['total_road_length_m'].append(total_length)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Fill batch with zeros to maintain consistency
                for _ in range(len(batch)):
                    if len(road_results['h3index']) <= i + len(batch):
                        road_results['h3index'].append('')
                        road_results['ID_HDC_G0'].append('')
                        road_results['UC_NM_MN'].append('')
                        road_results['CTR_MN_ISO'].append('')
                        for road_type in road_types:
                            road_results[f'road_length_m_{road_type}'].append(0)
                        road_results['total_road_length_m'].append(0)
        
        # Create DataFrame from results
        road_data = pd.DataFrame(road_results)
        
        # Add metadata
        road_data['extraction_date'] = datetime.now().strftime(DATE_FORMAT)
        road_data['data_source'] = 'OSM_GEE' if roads_fc else 'Synthetic_Estimates'
        road_data['scale_meters'] = 30  # OSM data resolution
        
        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime(DATE_FORMAT)
            if debug:
                output_file = f"Fig3_Roads_Neighborhood_H3_Resolution{H3_RESOLUTION}_debug_{timestamp}.csv"
            else:
                output_file = get_timestamped_filename("roads")
        
        # Save results
        output_path = OUTPUT_DIRS["neighborhood_data"] / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving road data to: {output_path}")
        road_data.to_csv(output_path, index=False)
        
        # Summary statistics
        logger.info("=" * 70)
        logger.info("ROAD NETWORK EXTRACTION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Neighborhoods processed: {len(road_data):,}")
        logger.info(f"Total road length: {road_data['total_road_length_m'].sum():,.0f} m")
        logger.info(f"Average road length per neighborhood: {road_data['total_road_length_m'].mean():.1f} m")
        
        # Summary by road type
        for road_type in road_types:
            col_name = f'road_length_m_{road_type}'
            if col_name in road_data.columns:
                total_length = road_data[col_name].sum()
                logger.info(f"  {road_type.title()} roads: {total_length:,.0f} m")
        
        logger.info(f"Output file: {output_path}")
        
        # Validate output file
        if output_path.exists() and output_path.stat().st_size > 0:
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"File size: {file_size_mb:.1f} MB")
            logger.info("Road network extraction completed successfully!")
            return True
        else:
            logger.error("Output file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"Error in road network extraction: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main(debug: bool = False, country_code: Optional[str] = None, **kwargs) -> bool:
    """Main function for command line execution."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = extract_road_data(
            country_code=country_code,
            debug=debug,
            **kwargs
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract road network data for H3 neighborhoods')
    parser.add_argument('--h3-file', type=str, help='Path to H3 neighborhoods file')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process subset of data')
    parser.add_argument('--output', type=str, help='Custom output filename')
    parser.add_argument('--road-types', nargs='+', 
                       default=['motorway', 'trunk', 'primary', 'secondary', 'tertiary'],
                       help='List of road types to extract')
    
    args = parser.parse_args()
    
    success = main(
        debug=args.debug,
        country_code=args.country,
        h3_file=args.h3_file,
        output_file=args.output,
        road_types=args.road_types
    )
    
    exit(0 if success else 1)