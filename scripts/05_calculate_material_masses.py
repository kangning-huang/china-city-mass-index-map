#!/usr/bin/env python3
"""
Calculate material masses using climate-adjusted intensities.

This script:
1. Loads merged infrastructure data from step 04
2. Applies climate-adjusted material intensity coefficients
3. Calculates building and road material masses by type
4. Saves material mass estimates per neighborhood

Input:
- Merged infrastructure data from step 04
- Material intensity parameters from configuration

Output:
- Neighborhood-level material mass estimates by type

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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *
from utils.material_intensity_utils import MaterialIntensityCalculator, validate_material_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_material_masses(
    merged_file: Optional[str] = None,
    country_code: Optional[str] = None,
    debug: bool = False,
    output_file: Optional[str] = None
) -> bool:
    """
    Calculate material masses from infrastructure data.
    
    Args:
        merged_file (str, optional): Path to merged infrastructure data
        country_code (str, optional): ISO 3-letter country code filter
        debug (bool): Use debug mode
        output_file (str, optional): Custom output filename
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("CALCULATING MATERIAL MASSES")
        logger.info("=" * 70)
        
        # Find merged data file if not specified
        if merged_file is None:
            merged_files = list(OUTPUT_DIRS["merged_data"].glob("Fig3_Merged_*.csv"))
            if merged_files:
                merged_file = max(merged_files, key=os.path.getmtime)
                logger.info(f"Using merged file: {merged_file}")
            else:
                logger.error("No merged data files found. Run step 04 first.")
                return False
        
        # Load merged infrastructure data
        logger.info(f"Loading merged infrastructure data from: {merged_file}")
        
        if str(merged_file).endswith('.gpkg'):
            data = gpd.read_file(merged_file)
        else:
            data = pd.read_csv(merged_file)
            # Convert to GeoDataFrame if geometry column exists
            if 'geometry' in data.columns:
                from shapely import wkt
                data['geometry'] = data['geometry'].apply(wkt.loads)
                data = gpd.GeoDataFrame(data, crs='EPSG:4326')
        
        if data.empty:
            logger.error("No data loaded from merged file")
            return False
        
        logger.info(f"Loaded {len(data)} neighborhood records")
        
        # Filter by country if specified
        if country_code:
            initial_count = len(data)
            data = data[data['CTR_MN_ISO'] == country_code.upper()]
            logger.info(f"Filtered to {len(data)} neighborhoods in {country_code} (from {initial_count})")
        
        if debug:
            # Use subset for debugging
            data = data.head(50)
            logger.info(f"Debug mode: Using {len(data)} neighborhoods")
        
        # Validate required columns
        required_columns = ['building_volume_m3', 'population', 'area_sqkm']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Add latitude for climate zone determination
        if 'latitude' not in data.columns:
            if isinstance(data, gpd.GeoDataFrame) and data.geometry is not None:
                # Calculate centroid latitude
                centroids = data.geometry.centroid
                data['latitude'] = centroids.y
                data['longitude'] = centroids.x
            else:
                logger.warning("No latitude data available, using default temperate climate")
                data['latitude'] = 40.0  # Default to temperate zone
        
        # Validate data
        if not validate_material_data(data, required_columns):
            logger.warning("Data validation issues detected, continuing with available data")
        
        # Initialize material intensity calculator
        logger.info("Initializing material intensity calculator...")
        calculator = MaterialIntensityCalculator(
            material_intensities=MATERIAL_INTENSITIES,
            road_intensities=ROAD_INTENSITIES
        )
        
        # Calculate building material masses
        logger.info("Calculating building material masses...")
        data = calculator.calculate_building_materials(
            data,
            volume_col='building_volume_m3',
            pop_col='population',
            area_col='area_sqkm',
            lat_col='latitude'
        )
        
        # Calculate road material masses
        logger.info("Calculating road material masses...")
        road_length_cols = [col for col in data.columns if col.startswith('road_length_m_')]
        
        if road_length_cols:
            data = calculator.calculate_road_materials(data, road_length_cols)
        else:
            logger.warning("No road length columns found, skipping road material calculations")
            # Add zero road material columns
            for material in ['concrete', 'steel', 'asphalt']:
                data[f'road_{material}_mass_tonnes'] = 0.0
            data['total_road_mass_tonnes'] = 0.0
        
        # Calculate total material masses
        logger.info("Calculating total material masses...")
        data = calculator.calculate_total_mass(data)
        
        # Add per capita and per area metrics
        logger.info("Calculating normalized metrics...")
        
        # Per capita metrics (avoid division by zero)
        population_safe = data['population'].replace(0, np.nan)
        
        if 'total_building_mass_tonnes' in data.columns:
            data['building_mass_per_capita'] = data['total_building_mass_tonnes'] / population_safe
        
        if 'total_road_mass_tonnes' in data.columns:
            data['road_mass_per_capita'] = data['total_road_mass_tonnes'] / population_safe
        
        if 'total_material_mass_tonnes' in data.columns:
            data['total_mass_per_capita'] = data['total_material_mass_tonnes'] / population_safe
        
        # Per area metrics
        area_safe = data['area_sqkm'].replace(0, np.nan)
        
        if 'total_building_mass_tonnes' in data.columns:
            data['building_mass_density_tonnes_per_km2'] = data['total_building_mass_tonnes'] / area_safe
        
        if 'total_road_mass_tonnes' in data.columns:
            data['road_mass_density_tonnes_per_km2'] = data['total_road_mass_tonnes'] / area_safe
        
        if 'total_material_mass_tonnes' in data.columns:
            data['total_mass_density_tonnes_per_km2'] = data['total_material_mass_tonnes'] / area_safe
        
        # Fill NaN values from division by zero
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'per_capita' in col or 'density' in col:
                data[col] = data[col].fillna(0)
        
        # Add metadata
        data['material_calc_date'] = datetime.now().strftime(DATE_FORMAT)
        data['fixed_slope'] = FIXED_SLOPE
        data['h3_resolution'] = H3_RESOLUTION
        
        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime(DATE_FORMAT)
            if debug:
                output_file = f"Fig3_Mass_Neighborhood_H3_Resolution{H3_RESOLUTION}_debug_{timestamp}.csv"
            else:
                output_file = get_timestamped_filename("mass")
        
        # Save material mass data
        output_path = OUTPUT_DIRS["merged_data"] / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving material mass data to: {output_path}")
        
        # Save as CSV
        data.to_csv(output_path, index=False)
        
        # Also save as GeoPackage for spatial analysis
        if isinstance(data, gpd.GeoDataFrame):
            gpkg_path = output_path.with_suffix('.gpkg')
            data.to_file(gpkg_path, driver='GPKG')
        
        # Generate summary statistics
        logger.info("=" * 70)
        logger.info("MATERIAL MASS CALCULATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Neighborhoods processed: {len(data):,}")
        logger.info(f"Countries: {data['CTR_MN_NM'].nunique()}")
        logger.info(f"Cities: {data['ID_HDC_G0'].nunique()}")
        
        # Material mass summaries
        logger.info("\nMaterial Mass Summary:")
        
        if 'total_building_mass_tonnes' in data.columns:
            total_building = data['total_building_mass_tonnes'].sum()
            logger.info(f"  Total building mass: {total_building:,.0f} tonnes ({total_building/1e6:.1f} Mt)")
            
            # By material type
            building_materials = ['steel', 'concrete', 'brick', 'aluminum', 'glass', 'timber']
            for material in building_materials:
                col_name = f'{material}_mass_tonnes'
                if col_name in data.columns:
                    material_total = data[col_name].sum()
                    percentage = (material_total / total_building * 100) if total_building > 0 else 0
                    logger.info(f"    {material.title()}: {material_total:,.0f} tonnes ({percentage:.1f}%)")
        
        if 'total_road_mass_tonnes' in data.columns:
            total_roads = data['total_road_mass_tonnes'].sum()
            logger.info(f"  Total road mass: {total_roads:,.0f} tonnes ({total_roads/1e6:.1f} Mt)")
            
            # By material type
            road_materials = ['concrete', 'steel', 'asphalt']
            for material in road_materials:
                col_name = f'road_{material}_mass_tonnes'
                if col_name in data.columns:
                    material_total = data[col_name].sum()
                    percentage = (material_total / total_roads * 100) if total_roads > 0 else 0
                    logger.info(f"    Road {material}: {material_total:,.0f} tonnes ({percentage:.1f}%)")
        
        if 'total_material_mass_tonnes' in data.columns:
            total_mass = data['total_material_mass_tonnes'].sum()
            logger.info(f"  TOTAL MATERIAL MASS: {total_mass:,.0f} tonnes ({total_mass/1e6:.1f} Mt)")
        
        # Intensity metrics
        logger.info("\nIntensity Metrics:")
        
        if 'total_mass_per_capita' in data.columns:
            avg_per_capita = data['total_mass_per_capita'].mean()
            logger.info(f"  Average mass per capita: {avg_per_capita:.1f} tonnes/person")
        
        if 'total_mass_density_tonnes_per_km2' in data.columns:
            avg_density = data['total_mass_density_tonnes_per_km2'].mean()
            logger.info(f"  Average mass density: {avg_density:,.0f} tonnes/km²")
        
        # Building class distribution
        if 'building_class' in data.columns:
            logger.info("\nBuilding Class Distribution:")
            class_counts = data['building_class'].value_counts()
            for building_class, count in class_counts.items():
                percentage = count / len(data) * 100
                logger.info(f"  {building_class}: {count:,} neighborhoods ({percentage:.1f}%)")
        
        # Climate zone distribution
        if 'climate_zone' in data.columns:
            logger.info("\nClimate Zone Distribution:")
            climate_counts = data['climate_zone'].value_counts()
            for climate, count in climate_counts.items():
                percentage = count / len(data) * 100
                logger.info(f"  {climate.title()}: {count:,} neighborhoods ({percentage:.1f}%)")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  CSV: {output_path}")
        if isinstance(data, gpd.GeoDataFrame):
            logger.info(f"  GeoPackage: {gpkg_path}")
        
        # Validate output file
        if output_path.exists() and output_path.stat().st_size > 0:
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"CSV file size: {file_size_mb:.1f} MB")
            logger.info("Material mass calculation completed successfully!")
            return True
        else:
            logger.error("Output file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"Error in material mass calculation: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main(debug: bool = False, country_code: Optional[str] = None, **kwargs) -> bool:
    """Main function for command line execution."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = calculate_material_masses(
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
    
    parser = argparse.ArgumentParser(description='Calculate material masses using climate-adjusted intensities')
    parser.add_argument('--merged-file', type=str, help='Path to merged infrastructure data file')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process subset of data')
    parser.add_argument('--output', type=str, help='Custom output filename')
    
    args = parser.parse_args()
    
    success = main(
        debug=args.debug,
        country_code=args.country,
        merged_file=args.merged_file,
        output_file=args.output
    )
    
    exit(0 if success else 1)