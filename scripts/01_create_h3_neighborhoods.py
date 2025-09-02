#!/usr/bin/env python3
"""
Create H3 hexagonal neighborhoods for all urban centres.

This script:
1. Loads urban centres from Google Earth Engine
2. Generates H3 hexagons at specified resolution for each city
3. Saves the H3 grids with city attributes

Input:
- Google Earth Engine FeatureCollection: "users/kh3657/GHS_STAT_UCDB2015"

Output:
- H3 neighborhoods CSV file with geometry and city attributes

Author: Generated for China City Mass Index Map
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import logging
from datetime import datetime
from typing import Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *
from utils.gee_utils import initialize_gee, load_urban_centres
from utils.spatial_utils import generate_h3_grids

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_h3_neighborhoods(
    country_code: Optional[str] = None,
    debug: bool = False,
    output_file: Optional[str] = None
) -> bool:
    """
    Create H3 neighborhood grids for all urban centres.
    
    Args:
        country_code (str, optional): ISO 3-letter country code filter
        debug (bool): Use debug mode (Bangladesh only) 
        output_file (str, optional): Custom output filename
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("CREATING H3 NEIGHBORHOOD GRIDS")
        logger.info("=" * 70)
        
        # Initialize Google Earth Engine
        if not initialize_gee():
            logger.error("Failed to initialize Google Earth Engine")
            return False
        
        # Load urban centres
        logger.info("Loading urban centres from Google Earth Engine...")
        cities_gdf = load_urban_centres(country_code=country_code, debug=debug)
        
        if cities_gdf is None or cities_gdf.empty:
            logger.error("No urban centres loaded")
            return False
        
        logger.info(f"Processing {len(cities_gdf)} urban centres")
        
        # Generate H3 grids for all cities
        all_h3_grids = []
        successful_cities = 0
        failed_cities = 0
        
        logger.info(f"Generating H3 grids at resolution {H3_RESOLUTION}...")
        
        for idx, city_row in tqdm(cities_gdf.iterrows(), total=len(cities_gdf), desc="Processing cities"):
            try:
                # Create single-city GeoDataFrame
                city_gdf = gpd.GeoDataFrame([city_row], crs=cities_gdf.crs)
                
                # Generate H3 grids for this city
                city_h3_grids = generate_h3_grids(
                    city_gdf, 
                    resolution=H3_RESOLUTION,
                    clip_to_boundary=True
                )
                
                if city_h3_grids is not None and not city_h3_grids.empty:
                    all_h3_grids.append(city_h3_grids)
                    successful_cities += 1
                    
                    if successful_cities % 100 == 0:
                        logger.info(f"Processed {successful_cities} cities successfully")
                else:
                    failed_cities += 1
                    logger.debug(f"Failed to generate H3 grids for city {city_row.get('UC_NM_MN', 'Unknown')}")
                    
            except Exception as e:
                failed_cities += 1
                logger.error(f"Error processing city {idx}: {e}")
                continue
        
        if not all_h3_grids:
            logger.error("No H3 grids generated successfully")
            return False
        
        # Combine all H3 grids
        logger.info(f"Combining H3 grids from {len(all_h3_grids)} cities...")
        combined_h3_grids = gpd.GeoDataFrame(pd.concat(all_h3_grids, ignore_index=True))
        
        # Ensure consistent CRS
        if combined_h3_grids.crs is None:
            combined_h3_grids = combined_h3_grids.set_crs('EPSG:4326')
        
        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime(DATE_FORMAT)
            if debug:
                output_file = f"Fig3_H3_Grids_Neighborhood_Resolution{H3_RESOLUTION}_debug_{timestamp}.csv"
            else:
                output_file = get_timestamped_filename("h3_grids")
        
        # Ensure output directory exists
        output_path = OUTPUT_DIRS["neighborhood_data"] / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV (geometry will be saved as WKT)
        logger.info(f"Saving H3 grids to: {output_path}")
        combined_h3_grids.to_csv(output_path, index=False)
        
        # Summary statistics
        logger.info("=" * 70)
        logger.info("H3 NEIGHBORHOOD CREATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total urban centres processed: {len(cities_gdf)}")
        logger.info(f"Successful cities: {successful_cities}")
        logger.info(f"Failed cities: {failed_cities}")
        logger.info(f"Total H3 hexagons created: {len(combined_h3_grids):,}")
        logger.info(f"H3 resolution: {H3_RESOLUTION}")
        logger.info(f"Output file: {output_path}")
        
        # Validate output
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            logger.info("H3 neighborhood creation completed successfully!")
            return True
        else:
            logger.error("Output file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"Error in H3 neighborhood creation: {e}")
        return False

def main():
    """Main function for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create H3 neighborhood grids for urban centres')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process Bangladesh only')
    parser.add_argument('--output', type=str, help='Custom output filename')
    parser.add_argument('--resolution', type=int, default=H3_RESOLUTION, help=f'H3 resolution (default: {H3_RESOLUTION})')
    
    args = parser.parse_args()
    
    # Update global H3 resolution if specified
    if args.resolution != H3_RESOLUTION:
        globals()['H3_RESOLUTION'] = args.resolution
        logger.info(f"Using H3 resolution: {args.resolution}")
    
    # Validate country code
    if args.country and len(args.country) != 3:
        logger.error("Country code must be 3 letters (e.g., CHN for China)")
        return False
    
    # Run the main process
    success = create_h3_neighborhoods(
        country_code=args.country,
        debug=args.debug,
        output_file=args.output
    )
    
    if success:
        logger.info("Script completed successfully!")
        return True
    else:
        logger.error("Script failed!")
        return False

if __name__ == "__main__":
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = main()
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)