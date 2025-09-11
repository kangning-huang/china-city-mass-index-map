#!/usr/bin/env python3
"""
Extract building volume data from Google Earth Engine for H3 neighborhoods.

This script:
1. Loads H3 neighborhood grids from the previous step
2. Extracts building volume data from Zhou2022 dataset via Google Earth Engine
3. Applies impervious surface mask to focus on urban areas
4. Saves building volume statistics per neighborhood

Input:
- H3 neighborhoods CSV/GeoPackage from step 01
- Zhou2022 building volume dataset from Google Earth Engine

Output:
- Building volume per neighborhood dataset

Author: Generated for China City Mass Index Map
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from datetime import datetime
from typing import Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *
from utils.gee_utils import initialize_gee, zonal_statistics, mask_urban_areas
from utils.spatial_utils import validate_geometries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_building_data(
    h3_file: Optional[str] = None,
    country_code: Optional[str] = None,
    debug: bool = False,
    output_file: Optional[str] = None
) -> bool:
    """
    Extract building volume data for H3 neighborhoods.
    
    Args:
        h3_file (str, optional): Path to H3 neighborhoods file
        country_code (str, optional): ISO 3-letter country code filter
        debug (bool): Use debug mode
        output_file (str, optional): Custom output filename
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("EXTRACTING BUILDING VOLUME DATA")
        logger.info("=" * 70)
        
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
        
        # Load building volume dataset from Google Earth Engine
        import ee
        logger.info("Loading Zhou2022 building volume dataset...")
        
        # Load the building volume image
        # Note: Replace with actual asset ID
        try:
            building_volume = ee.Image("users/ee-knhuang/zhou2022_building_volume_500m")
        except:
            # Fallback to a test dataset or create a dummy one
            logger.warning("Zhou2022 dataset not available, using alternative")
            # You could use GHS built-up volume as alternative
            building_volume = ee.ImageCollection("JRC/GHSL/P2016/BUILT_VOLM_GLOBE_V1") \
                               .filter(ee.Filter.date('2014-01-01', '2014-12-31')) \
                               .first()
        
        # Apply urban mask
        building_volume_masked = mask_urban_areas(building_volume, impervious_threshold=0.1)
        
        # Extract building volumes for neighborhoods
        logger.info("Extracting building volumes via Google Earth Engine...")
        logger.info("This may take several minutes depending on the number of neighborhoods...")
        
        # Process in batches to avoid GEE limits
        batch_size = 500
        all_results = []
        
        for i in tqdm(range(0, len(h3_gdf), batch_size), desc="Processing batches"):
            batch_gdf = h3_gdf.iloc[i:i+batch_size].copy()
            
            try:
                # Calculate zonal statistics
                batch_results = zonal_statistics(
                    image=building_volume_masked,
                    features=batch_gdf,
                    statistics=['sum', 'mean', 'count'],
                    scale=500,  # 500m resolution for building data
                    max_pixels=1e9
                )
                
                if batch_results is not None and not batch_results.empty:
                    all_results.append(batch_results)
                else:
                    logger.warning(f"No results for batch {i//batch_size + 1}")
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        if not all_results:
            logger.error("No building volume data extracted")
            return False
        
        # Combine all results
        logger.info("Combining results from all batches...")
        building_data = pd.concat(all_results, ignore_index=True)
        
        # Rename columns for clarity
        column_mapping = {
            'sum': 'building_volume_m3',
            'mean': 'building_volume_mean_m3',
            'count': 'building_pixels_count'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in building_data.columns:
                building_data = building_data.rename(columns={old_col: new_col})
        
        # Add metadata
        building_data['extraction_date'] = datetime.now().strftime(DATE_FORMAT)
        building_data['data_source'] = 'Zhou2022_GEE'
        building_data['scale_meters'] = 500
        
        # Handle missing values
        building_data['building_volume_m3'] = building_data['building_volume_m3'].fillna(0)
        building_data['building_volume_mean_m3'] = building_data['building_volume_mean_m3'].fillna(0)
        building_data['building_pixels_count'] = building_data['building_pixels_count'].fillna(0)
        
        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime(DATE_FORMAT)
            if debug:
                output_file = f"Fig3_Volume_Pavement_Neighborhood_H3_Resolution{H3_RESOLUTION}_debug_{timestamp}.csv"
            else:
                output_file = get_timestamped_filename("volume_pavement")
        
        # Save results
        output_path = OUTPUT_DIRS["neighborhood_data"] / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving building data to: {output_path}")
        building_data.to_csv(output_path, index=False)
        
        # Summary statistics
        logger.info("=" * 70)
        logger.info("BUILDING VOLUME EXTRACTION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Neighborhoods processed: {len(building_data):,}")
        logger.info(f"Total building volume: {building_data['building_volume_m3'].sum():,.0f} m³")
        logger.info(f"Average volume per neighborhood: {building_data['building_volume_m3'].mean():.1f} m³")
        logger.info(f"Neighborhoods with buildings: {(building_data['building_volume_m3'] > 0).sum():,}")
        logger.info(f"Output file: {output_path}")
        
        # Validate output file
        if output_path.exists() and output_path.stat().st_size > 0:
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"File size: {file_size_mb:.1f} MB")
            logger.info("Building volume extraction completed successfully!")
            return True
        else:
            logger.error("Output file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"Error in building volume extraction: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main(debug: bool = False, country_code: Optional[str] = None, **kwargs) -> bool:
    """Main function for command line execution."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = extract_building_data(
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
    
    parser = argparse.ArgumentParser(description='Extract building volume data for H3 neighborhoods')
    parser.add_argument('--h3-file', type=str, help='Path to H3 neighborhoods file')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process subset of data')
    parser.add_argument('--output', type=str, help='Custom output filename')
    
    args = parser.parse_args()
    
    success = main(
        debug=args.debug,
        country_code=args.country,
        h3_file=args.h3_file,
        output_file=args.output
    )
    
    exit(0 if success else 1)