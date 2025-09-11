#!/usr/bin/env python3
"""
Generate city boundary geometries from H3 grids and material mass data.

This script:
1. Loads material mass data with H3 geometries from step 05
2. Dissolves H3 hexagons by city to create city boundaries
3. Calculates city-level summary statistics
4. Saves city boundary geometries for visualization

Input:
- Material mass data with H3 geometries from step 05

Output:
- City boundary geometries (GeoPackage)
- H3 grid geometries (GeoPackage)
- City-level summary statistics

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
from utils.spatial_utils import dissolve_by_attribute, validate_geometries, calculate_polygon_areas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_city_boundaries(
    material_file: Optional[str] = None,
    country_code: Optional[str] = None,
    debug: bool = False
) -> bool:
    """
    Generate city boundary geometries from H3 grids.
    
    Args:
        material_file: Path to material mass data with geometries
        country_code: ISO 3-letter country code filter
        debug: Use debug mode
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("GENERATING CITY BOUNDARIES")
        logger.info("=" * 70)
        
        # Find material mass file if not specified
        if material_file is None:
            material_files = list(OUTPUT_DIRS["merged_data"].glob("Fig3_Mass_*.gpkg"))
            if not material_files:
                # Try CSV files
                csv_files = list(OUTPUT_DIRS["merged_data"].glob("Fig3_Mass_*.csv"))
                if csv_files:
                    material_file = max(csv_files, key=os.path.getmtime)
                else:
                    logger.error("No material mass files found. Run step 05 first.")
                    return False
            else:
                material_file = max(material_files, key=os.path.getmtime)
            
            logger.info(f"Using material file: {material_file}")
        
        # Load material mass data with geometries
        logger.info(f"Loading material mass data from: {material_file}")
        
        if str(material_file).endswith('.gpkg'):
            h3_data = gpd.read_file(material_file)
        else:
            # Load CSV and convert to GeoDataFrame
            data_df = pd.read_csv(material_file)
            if 'geometry' in data_df.columns:
                from shapely import wkt
                data_df['geometry'] = data_df['geometry'].apply(wkt.loads)
                h3_data = gpd.GeoDataFrame(data_df, crs='EPSG:4326')
            else:
                logger.error("No geometry column found in material file")
                return False
        
        if h3_data.empty:
            logger.error("No data loaded from material file")
            return False
        
        logger.info(f"Loaded {len(h3_data)} H3 neighborhoods with material data")
        
        # Filter by country if specified
        if country_code:
            initial_count = len(h3_data)
            h3_data = h3_data[h3_data['CTR_MN_ISO'] == country_code.upper()]
            logger.info(f"Filtered to {len(h3_data)} neighborhoods in {country_code} (from {initial_count})")
        
        if debug:
            # Use subset for debugging
            h3_data = h3_data.head(200)
            logger.info(f"Debug mode: Using {len(h3_data)} neighborhoods")
        
        # Validate geometries
        h3_data = validate_geometries(h3_data)
        
        # Ensure required columns exist
        if 'ID_HDC_G0' not in h3_data.columns:
            logger.error("No city ID column (ID_HDC_G0) found")
            return False
        
        # Create city boundaries by dissolving H3 hexagons
        logger.info("Creating city boundaries by dissolving H3 hexagons...")
        
        # Define aggregation functions for numeric columns
        numeric_columns = h3_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Aggregation rules
        agg_dict = {}
        
        # Sum these columns
        sum_columns = [
            'population', 'total_material_mass_tonnes', 'building_volume_m3',
            'total_road_length_m', 'total_building_mass_tonnes', 'total_road_mass_tonnes'
        ]
        
        # Mean these columns  
        mean_columns = [
            'building_density_m3_per_km2', 'road_density_m_per_km2', 'population_density_per_km2'
        ]
        
        for col in numeric_columns:
            if col in sum_columns:
                agg_dict[col] = 'sum'
            elif col in mean_columns:
                agg_dict[col] = 'mean'
            elif col.endswith('_tonnes') or col.endswith('_m3') or col.endswith('_m'):
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'mean'
        
        # Preserve categorical columns (use first value)
        categorical_columns = ['UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2']
        for col in categorical_columns:
            if col in h3_data.columns:
                agg_dict[col] = 'first'
        
        # Dissolve by city
        city_boundaries = dissolve_by_attribute(h3_data, by='ID_HDC_G0', agg_dict=agg_dict)
        
        logger.info(f"Created {len(city_boundaries)} city boundaries")
        
        # Calculate city areas
        city_boundaries['city_area_sqkm'] = calculate_polygon_areas(city_boundaries, 'km2')
        
        # Calculate derived metrics
        logger.info("Calculating city-level derived metrics...")
        
        # Recalculate density metrics based on city totals
        if 'population' in city_boundaries.columns and 'city_area_sqkm' in city_boundaries.columns:
            city_boundaries['city_population_density_per_km2'] = (
                city_boundaries['population'] / city_boundaries['city_area_sqkm']
            )
        
        if 'total_material_mass_tonnes' in city_boundaries.columns and 'city_area_sqkm' in city_boundaries.columns:
            city_boundaries['city_mass_density_tonnes_per_km2'] = (
                city_boundaries['total_material_mass_tonnes'] / city_boundaries['city_area_sqkm']
            )
        
        if 'total_material_mass_tonnes' in city_boundaries.columns and 'population' in city_boundaries.columns:
            city_boundaries['city_mass_per_capita'] = (
                city_boundaries['total_material_mass_tonnes'] / city_boundaries['population']
            )
        
        # Count neighborhoods per city
        neighborhood_counts = h3_data.groupby('ID_HDC_G0').size().reset_index(name='neighborhood_count')
        city_boundaries = city_boundaries.merge(neighborhood_counts, on='ID_HDC_G0', how='left')
        
        # Add metadata
        city_boundaries['boundary_created_date'] = datetime.now().strftime(DATE_FORMAT)
        city_boundaries['h3_resolution'] = H3_RESOLUTION
        
        # Prepare H3 grids data for saving
        h3_grids_clean = h3_data.copy()
        
        # Ensure output directory exists
        boundaries_dir = OUTPUT_DIRS.get("boundaries", RESULTS_DIR / "shapefiles")
        boundaries_dir.mkdir(parents=True, exist_ok=True)
        
        # Save city boundaries
        city_boundaries_path = boundaries_dir / "all_cities_boundaries.gpkg"
        logger.info(f"Saving city boundaries to: {city_boundaries_path}")
        city_boundaries.to_file(city_boundaries_path, driver='GPKG')
        
        # Save H3 grids
        h3_grids_path = boundaries_dir / "all_cities_h3_grids.gpkg"
        logger.info(f"Saving H3 grids to: {h3_grids_path}")
        h3_grids_clean.to_file(h3_grids_path, driver='GPKG')
        
        # Create city statistics CSV (without geometry for easier analysis)
        city_stats_df = city_boundaries.drop(columns=['geometry']).copy()
        city_stats_path = boundaries_dir / "city_summary_statistics.csv"
        city_stats_df.to_csv(city_stats_path, index=False)
        logger.info(f"Saved city statistics to: {city_stats_path}")
        
        # Summary statistics
        logger.info("=" * 70)
        logger.info("CITY BOUNDARY GENERATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Cities processed: {len(city_boundaries):,}")
        logger.info(f"Countries: {city_boundaries['CTR_MN_NM'].nunique()}")
        logger.info(f"Total neighborhoods: {city_boundaries['neighborhood_count'].sum():,}")
        
        # Size statistics
        logger.info(f"\nCity Size Statistics:")
        if 'city_area_sqkm' in city_boundaries.columns:
            logger.info(f"  Average city area: {city_boundaries['city_area_sqkm'].mean():.1f} km²")
            logger.info(f"  Largest city: {city_boundaries['city_area_sqkm'].max():.1f} km²")
            logger.info(f"  Smallest city: {city_boundaries['city_area_sqkm'].min():.1f} km²")
        
        if 'population' in city_boundaries.columns:
            logger.info(f"  Average city population: {city_boundaries['population'].mean():,.0f}")
            logger.info(f"  Largest city population: {city_boundaries['population'].max():,.0f}")
            logger.info(f"  Total population: {city_boundaries['population'].sum():,.0f}")
        
        if 'total_material_mass_tonnes' in city_boundaries.columns:
            total_mass = city_boundaries['total_material_mass_tonnes'].sum()
            logger.info(f"  Total material mass: {total_mass:,.0f} tonnes ({total_mass/1e6:.1f} Mt)")
            logger.info(f"  Average city mass: {city_boundaries['total_material_mass_tonnes'].mean():,.0f} tonnes")
        
        # Top cities by population and mass
        if 'population' in city_boundaries.columns and 'UC_NM_MN' in city_boundaries.columns:
            logger.info(f"\nTop 5 Cities by Population:")
            top_pop_cities = city_boundaries.nlargest(5, 'population')[['UC_NM_MN', 'population']]
            for idx, row in top_pop_cities.iterrows():
                logger.info(f"  {row['UC_NM_MN']}: {row['population']:,.0f}")
        
        if 'total_material_mass_tonnes' in city_boundaries.columns and 'UC_NM_MN' in city_boundaries.columns:
            logger.info(f"\nTop 5 Cities by Material Mass:")
            top_mass_cities = city_boundaries.nlargest(5, 'total_material_mass_tonnes')[['UC_NM_MN', 'total_material_mass_tonnes']]
            for idx, row in top_mass_cities.iterrows():
                logger.info(f"  {row['UC_NM_MN']}: {row['total_material_mass_tonnes']:,.0f} tonnes")
        
        # File information
        logger.info(f"\nOutput Files:")
        logger.info(f"  City boundaries: {city_boundaries_path}")
        logger.info(f"  H3 grids: {h3_grids_path}")
        logger.info(f"  City statistics: {city_stats_path}")
        
        # File sizes
        for path in [city_boundaries_path, h3_grids_path, city_stats_path]:
            if path.exists():
                file_size_mb = path.stat().st_size / 1024 / 1024
                logger.info(f"    {path.name}: {file_size_mb:.1f} MB")
        
        logger.info("City boundary generation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error generating city boundaries: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main(debug: bool = False, country_code: Optional[str] = None, **kwargs) -> bool:
    """Main function for command line execution."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = generate_city_boundaries(
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
    
    parser = argparse.ArgumentParser(description='Generate city boundary geometries from H3 grids')
    parser.add_argument('--material-file', type=str, help='Path to material mass data file')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process subset of data')
    
    args = parser.parse_args()
    
    success = main(
        debug=args.debug,
        country_code=args.country,
        material_file=args.material_file
    )
    
    exit(0 if success else 1)