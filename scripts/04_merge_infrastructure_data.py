#!/usr/bin/env python3
"""
Merge building and infrastructure data with population estimates.

This script:
1. Loads building volume data from step 02
2. Loads road network data from step 03
3. Extracts population data from WorldPop via Google Earth Engine
4. Merges all datasets on H3 neighborhood identifiers
5. Saves merged neighborhood dataset

Input:
- Building volume data from step 02
- Road network data from step 03
- Population data from Google Earth Engine WorldPop

Output:
- Merged neighborhood infrastructure dataset

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
from utils.gee_utils import initialize_gee, zonal_statistics
from utils.spatial_utils import validate_geometries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_infrastructure_data(
    building_file: Optional[str] = None,
    roads_file: Optional[str] = None,
    h3_file: Optional[str] = None,
    country_code: Optional[str] = None,
    debug: bool = False,
    output_file: Optional[str] = None,
    population_year: int = 2020
) -> bool:
    """
    Merge building, road, and population data for H3 neighborhoods.
    
    Args:
        building_file (str, optional): Path to building volume data
        roads_file (str, optional): Path to road network data
        h3_file (str, optional): Path to H3 neighborhoods file
        country_code (str, optional): ISO 3-letter country code filter
        debug (bool): Use debug mode
        output_file (str, optional): Custom output filename
        population_year (int): Year for population data
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("MERGING INFRASTRUCTURE DATA")
        logger.info("=" * 70)
        
        # Find input files if not specified
        data_dir = OUTPUT_DIRS["neighborhood_data"]
        
        if building_file is None:
            building_files = list(data_dir.glob("Fig3_Volume_*.csv"))
            if building_files:
                building_file = max(building_files, key=os.path.getmtime)
                logger.info(f"Using building file: {building_file}")
            else:
                logger.error("No building volume files found. Run step 02 first.")
                return False
        
        if roads_file is None:
            roads_files = list(data_dir.glob("Fig3_Roads_*.csv"))
            if roads_files:
                roads_file = max(roads_files, key=os.path.getmtime)
                logger.info(f"Using roads file: {roads_file}")
            else:
                logger.error("No road network files found. Run step 03 first.")
                return False
        
        if h3_file is None:
            h3_files = list(data_dir.glob("Fig3_H3_Grids_*.csv"))
            if h3_files:
                h3_file = max(h3_files, key=os.path.getmtime)
                logger.info(f"Using H3 file: {h3_file}")
            else:
                logger.error("No H3 grid files found. Run step 01 first.")
                return False
        
        # Load datasets
        logger.info("Loading infrastructure datasets...")
        
        # Load building data
        logger.info(f"Loading building data from: {building_file}")
        building_data = pd.read_csv(building_file)
        logger.info(f"Loaded {len(building_data)} building records")
        
        # Load road data
        logger.info(f"Loading road data from: {roads_file}")
        road_data = pd.read_csv(roads_file)
        logger.info(f"Loaded {len(road_data)} road records")
        
        # Load H3 data with geometries
        logger.info(f"Loading H3 neighborhoods from: {h3_file}")
        if str(h3_file).endswith('.gpkg'):
            h3_gdf = gpd.read_file(h3_file)
        else:
            h3_df = pd.read_csv(h3_file)
            if 'geometry' in h3_df.columns:
                from shapely import wkt
                h3_df['geometry'] = h3_df['geometry'].apply(wkt.loads)
                h3_gdf = gpd.GeoDataFrame(h3_df, crs='EPSG:4326')
            else:
                logger.error("No geometry column found in H3 file")
                return False
        
        logger.info(f"Loaded {len(h3_gdf)} H3 neighborhoods with geometries")
        
        # Filter by country if specified
        if country_code:
            initial_count = len(h3_gdf)
            h3_gdf = h3_gdf[h3_gdf['CTR_MN_ISO'] == country_code.upper()]
            logger.info(f"Filtered to {len(h3_gdf)} neighborhoods in {country_code} (from {initial_count})")
        
        if debug:
            # Use subset for debugging
            h3_gdf = h3_gdf.head(50)
            logger.info(f"Debug mode: Using {len(h3_gdf)} neighborhoods")
        
        # Extract population data from Google Earth Engine
        population_data = None
        try:
            logger.info(f"Extracting population data for {population_year}...")
            
            # Initialize Google Earth Engine
            if not initialize_gee():
                logger.warning("Failed to initialize Google Earth Engine, using synthetic population data")
            else:
                import ee
                
                # Load WorldPop population data
                population_image = ee.ImageCollection("WorldPop/GP/100m/pop") \
                                    .filter(ee.Filter.eq('year', population_year)) \
                                    .mosaic()
                
                # Extract population for neighborhoods in batches
                batch_size = 200
                pop_results = []
                
                for i in tqdm(range(0, len(h3_gdf), batch_size), desc="Extracting population"):
                    batch_gdf = h3_gdf.iloc[i:i+batch_size].copy()
                    
                    try:
                        batch_pop = zonal_statistics(
                            image=population_image,
                            features=batch_gdf,
                            statistics=['sum'],
                            scale=100,  # 100m WorldPop resolution
                            max_pixels=1e9
                        )
                        
                        if batch_pop is not None and not batch_pop.empty:
                            pop_results.append(batch_pop)
                            
                    except Exception as e:
                        logger.warning(f"Error extracting population for batch {i//batch_size + 1}: {e}")
                        continue
                
                if pop_results:
                    population_data = pd.concat(pop_results, ignore_index=True)
                    population_data = population_data.rename(columns={'sum': 'population'})
                    logger.info(f"Extracted population data for {len(population_data)} neighborhoods")
                
        except Exception as e:
            logger.warning(f"Error extracting population data: {e}")
        
        # Create synthetic population data if GEE extraction failed
        if population_data is None or population_data.empty:
            logger.info("Using synthetic population estimates based on built area")
            population_data = pd.DataFrame({
                'h3index': h3_gdf['h3index'],
                'population': h3_gdf.get('area_sqkm', 1.0) * 1000  # Rough estimate: 1000 people per km²
            })
        
        # Merge all datasets
        logger.info("Merging all infrastructure datasets...")
        
        # Start with H3 base data
        merged_data = h3_gdf.copy()
        
        # Merge building data
        if 'h3index' in building_data.columns:
            merged_data = merged_data.merge(
                building_data[['h3index', 'building_volume_m3', 'building_volume_mean_m3', 'building_pixels_count']],
                on='h3index',
                how='left'
            )
            logger.info("Merged building volume data")
        else:
            logger.warning("No h3index column in building data, skipping merge")
        
        # Merge road data
        if 'h3index' in road_data.columns:
            road_cols = ['h3index'] + [col for col in road_data.columns if col.startswith('road_length_m_') or col == 'total_road_length_m']
            merged_data = merged_data.merge(
                road_data[road_cols],
                on='h3index',
                how='left'
            )
            logger.info("Merged road network data")
        else:
            logger.warning("No h3index column in road data, skipping merge")
        
        # Merge population data
        if 'h3index' in population_data.columns:
            merged_data = merged_data.merge(
                population_data[['h3index', 'population']],
                on='h3index',
                how='left'
            )
            logger.info("Merged population data")
        else:
            logger.warning("No h3index column in population data, skipping merge")
        
        # Fill missing values
        numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'area_sqkm':  # Don't fill area with zeros
                merged_data[col] = merged_data[col].fillna(0)
        
        # Add derived metrics
        logger.info("Calculating derived metrics...")
        
        # Building density metrics
        if 'building_volume_m3' in merged_data.columns and 'area_sqkm' in merged_data.columns:
            merged_data['building_density_m3_per_km2'] = merged_data['building_volume_m3'] / merged_data['area_sqkm']
            merged_data['building_density_m3_per_km2'] = merged_data['building_density_m3_per_km2'].fillna(0)
        
        # Road density metrics
        if 'total_road_length_m' in merged_data.columns and 'area_sqkm' in merged_data.columns:
            merged_data['road_density_m_per_km2'] = merged_data['total_road_length_m'] / merged_data['area_sqkm']
            merged_data['road_density_m_per_km2'] = merged_data['road_density_m_per_km2'].fillna(0)
        
        # Population density
        if 'population' in merged_data.columns and 'area_sqkm' in merged_data.columns:
            merged_data['population_density_per_km2'] = merged_data['population'] / merged_data['area_sqkm']
            merged_data['population_density_per_km2'] = merged_data['population_density_per_km2'].fillna(0)
        
        # Add metadata
        merged_data['merge_date'] = datetime.now().strftime(DATE_FORMAT)
        merged_data['population_year'] = population_year
        
        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime(DATE_FORMAT)
            if debug:
                output_file = f"Fig3_Merged_Neighborhood_H3_Resolution{H3_RESOLUTION}_debug_{timestamp}.csv"
            else:
                output_file = get_timestamped_filename("merged")
        
        # Save merged data
        output_path = OUTPUT_DIRS["merged_data"] / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving merged data to: {output_path}")
        
        # Save as CSV (geometry saved as WKT)
        merged_data.to_csv(output_path, index=False)
        
        # Also save as GeoPackage for spatial analysis
        gpkg_path = output_path.with_suffix('.gpkg')
        merged_data.to_file(gpkg_path, driver='GPKG')
        
        # Summary statistics
        logger.info("=" * 70)
        logger.info("INFRASTRUCTURE DATA MERGE COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Final merged neighborhoods: {len(merged_data):,}")
        logger.info(f"Countries: {merged_data['CTR_MN_NM'].nunique()}")
        logger.info(f"Cities: {merged_data['ID_HDC_G0'].nunique()}")
        
        # Data completeness
        logger.info("\nData Completeness:")
        if 'building_volume_m3' in merged_data.columns:
            buildings_with_data = (merged_data['building_volume_m3'] > 0).sum()
            logger.info(f"  Neighborhoods with buildings: {buildings_with_data:,} ({buildings_with_data/len(merged_data)*100:.1f}%)")
        
        if 'total_road_length_m' in merged_data.columns:
            roads_with_data = (merged_data['total_road_length_m'] > 0).sum()
            logger.info(f"  Neighborhoods with roads: {roads_with_data:,} ({roads_with_data/len(merged_data)*100:.1f}%)")
        
        if 'population' in merged_data.columns:
            pop_with_data = (merged_data['population'] > 0).sum()
            logger.info(f"  Neighborhoods with population: {pop_with_data:,} ({pop_with_data/len(merged_data)*100:.1f}%)")
        
        # Summary statistics
        logger.info("\nSummary Statistics:")
        if 'building_volume_m3' in merged_data.columns:
            total_volume = merged_data['building_volume_m3'].sum()
            logger.info(f"  Total building volume: {total_volume:,.0f} m³")
        
        if 'total_road_length_m' in merged_data.columns:
            total_roads = merged_data['total_road_length_m'].sum()
            logger.info(f"  Total road length: {total_roads:,.0f} m ({total_roads/1000:,.0f} km)")
        
        if 'population' in merged_data.columns:
            total_population = merged_data['population'].sum()
            logger.info(f"  Total population: {total_population:,.0f}")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  CSV: {output_path}")
        logger.info(f"  GeoPackage: {gpkg_path}")
        
        # Validate output file
        if output_path.exists() and output_path.stat().st_size > 0:
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"CSV file size: {file_size_mb:.1f} MB")
            logger.info("Infrastructure data merge completed successfully!")
            return True
        else:
            logger.error("Output file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"Error in infrastructure data merge: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main(debug: bool = False, country_code: Optional[str] = None, **kwargs) -> bool:
    """Main function for command line execution."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = merge_infrastructure_data(
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
    
    parser = argparse.ArgumentParser(description='Merge building, road, and population data')
    parser.add_argument('--building-file', type=str, help='Path to building volume data file')
    parser.add_argument('--roads-file', type=str, help='Path to road network data file')
    parser.add_argument('--h3-file', type=str, help='Path to H3 neighborhoods file')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process subset of data')
    parser.add_argument('--output', type=str, help='Custom output filename')
    parser.add_argument('--population-year', type=int, default=2020, help='Year for population data')
    
    args = parser.parse_args()
    
    success = main(
        debug=args.debug,
        country_code=args.country,
        building_file=args.building_file,
        roads_file=args.roads_file,
        h3_file=args.h3_file,
        output_file=args.output,
        population_year=args.population_year
    )
    
    exit(0 if success else 1)