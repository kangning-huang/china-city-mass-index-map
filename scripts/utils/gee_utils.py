"""
Google Earth Engine utilities for data extraction and processing.
"""

import ee
import geemap
import pandas as pd
import geopandas as gpd
import logging
from typing import List, Dict, Optional, Union
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import GEE_PROJECT_ID

logger = logging.getLogger(__name__)

def initialize_gee(project_id: str = GEE_PROJECT_ID):
    """
    Initialize Google Earth Engine with authentication.
    
    Args:
        project_id (str): GEE project ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ee.Initialize(project=project_id)
        logger.info(f"Google Earth Engine initialized successfully with project: {project_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Google Earth Engine: {e}")
        try:
            logger.info("Attempting to authenticate...")
            ee.Authenticate()
            ee.Initialize(project=project_id)
            logger.info("Authentication successful, GEE initialized")
            return True
        except Exception as auth_e:
            logger.error(f"Authentication failed: {auth_e}")
            return False

def load_urban_centres(country_code: Optional[str] = None, debug: bool = False) -> Optional[gpd.GeoDataFrame]:
    """
    Load urban centres from Google Earth Engine GHSL database.
    
    Args:
        country_code (str, optional): ISO 3-letter country code (e.g., 'CHN')
        debug (bool): If True, filter to Bangladesh for testing
        
    Returns:
        gpd.GeoDataFrame: Urban centres data
    """
    try:
        # Properties to select from the urban centres dataset
        properties = ['ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 
                     'GRGN_L1', 'GRGN_L2', 'E_KG_NM_LS']
        
        # Load urban centres collection
        fc_ucs = ee.FeatureCollection("users/kh3657/GHS_STAT_UCDB2015").select(
            propertySelectors=properties, 
            retainGeometry=True
        )
        
        # Apply filters
        if debug:
            logger.info("Debug mode: Filtering to Bangladesh cities only")
            fc_ucs = fc_ucs.filter(ee.Filter.eq('CTR_MN_ISO', 'BGD'))
        elif country_code:
            logger.info(f"Filtering to {country_code} cities only")
            fc_ucs = fc_ucs.filter(ee.Filter.eq('CTR_MN_ISO', country_code.upper()))
        
        # Convert to GeoDataFrame
        gdf_cities = geemap.ee_to_gdf(fc_ucs)
        
        filter_desc = "Bangladesh" if debug else (country_code if country_code else "all countries")
        logger.info(f"Loaded {len(gdf_cities)} urban centres from {filter_desc}")
        
        return gdf_cities
        
    except Exception as e:
        logger.error(f"Error loading urban centres: {e}")
        return None

def ee_featurecollection_to_dataframe(fc: ee.FeatureCollection, chunk_size: int = 5000) -> Optional[pd.DataFrame]:
    """
    Convert large Earth Engine FeatureCollection to pandas DataFrame with chunking.
    
    Args:
        fc (ee.FeatureCollection): Earth Engine FeatureCollection
        chunk_size (int): Number of features per chunk
        
    Returns:
        pd.DataFrame: Converted dataframe
    """
    try:
        # Get total size
        total_size = fc.size().getInfo()
        logger.info(f"Converting FeatureCollection with {total_size} features to DataFrame")
        
        if total_size <= chunk_size:
            # Small enough to convert directly
            return geemap.ee_to_df(fc)
        
        # Process in chunks
        chunks = []
        for i in range(0, total_size, chunk_size):
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(total_size//chunk_size) + 1}")
            chunk_fc = fc.limit(chunk_size, i)
            chunk_df = geemap.ee_to_df(chunk_fc)
            chunks.append(chunk_df)
        
        # Combine chunks
        result_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Successfully converted {len(result_df)} features to DataFrame")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error converting FeatureCollection to DataFrame: {e}")
        return None

def zonal_statistics(
    image: Union[ee.Image, str], 
    features: Union[ee.FeatureCollection, gpd.GeoDataFrame],
    statistics: List[str] = ['sum', 'mean', 'count'],
    scale: int = 100,
    crs: str = 'EPSG:4326',
    max_pixels: int = 1e9
) -> Optional[pd.DataFrame]:
    """
    Calculate zonal statistics for image over features.
    
    Args:
        image (ee.Image or str): Image or image asset ID
        features (ee.FeatureCollection or gpd.GeoDataFrame): Zones for calculation
        statistics (list): Statistics to calculate
        scale (int): Pixel scale in meters
        crs (str): Coordinate reference system
        max_pixels (int): Maximum pixels per computation
        
    Returns:
        pd.DataFrame: Zonal statistics results
    """
    try:
        # Convert inputs to EE objects if needed
        if isinstance(image, str):
            image = ee.Image(image)
        
        if isinstance(features, gpd.GeoDataFrame):
            features = geemap.geopandas_to_ee(features)
        
        # Calculate zonal statistics
        logger.info(f"Calculating zonal statistics with scale={scale}m")
        
        result_fc = geemap.zonal_stats(
            in_value_raster=image,
            in_zone_vector=features,
            out_stats=statistics,
            scale=scale,
            crs=crs,
            max_pixels=max_pixels,
            return_fc=True
        )
        
        # Convert to DataFrame
        result_df = ee_featurecollection_to_dataframe(result_fc)
        
        if result_df is not None:
            logger.info(f"Zonal statistics completed for {len(result_df)} features")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating zonal statistics: {e}")
        return None

def get_image_info(image: Union[ee.Image, str]) -> Dict:
    """
    Get information about an Earth Engine image.
    
    Args:
        image (ee.Image or str): Image or asset ID
        
    Returns:
        dict: Image information
    """
    try:
        if isinstance(image, str):
            image = ee.Image(image)
        
        info = {
            'bands': image.bandNames().getInfo(),
            'projection': image.projection().getInfo(),
            'scale': image.projection().nominalScale().getInfo()
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return {}

def mask_urban_areas(image: ee.Image, impervious_threshold: float = 0.1) -> ee.Image:
    """
    Mask image to urban areas using impervious surface data.
    
    Args:
        image (ee.Image): Image to mask
        impervious_threshold (float): Minimum impervious surface fraction
        
    Returns:
        ee.Image: Masked image
    """
    try:
        # Load GAIA FROM-GLC impervious surface data (adjust year as needed)
        impervious = ee.Image("users/wang_dongyu/FROM-GLC_2015/FROM_GLC_2015_100m_impervious")
        
        # Create urban mask
        urban_mask = impervious.gt(impervious_threshold * 100)  # Convert to percentage
        
        # Apply mask
        masked_image = image.updateMask(urban_mask)
        
        return masked_image
        
    except Exception as e:
        logger.error(f"Error masking urban areas: {e}")
        return image

def export_to_drive(
    image: ee.Image,
    description: str,
    folder: str = 'EarthEngine',
    file_name: Optional[str] = None,
    region: Optional[ee.Geometry] = None,
    scale: int = 100,
    crs: str = 'EPSG:4326'
) -> ee.batch.Task:
    """
    Export Earth Engine image to Google Drive.
    
    Args:
        image (ee.Image): Image to export
        description (str): Export description
        folder (str): Drive folder name
        file_name (str, optional): Output filename
        region (ee.Geometry, optional): Export region
        scale (int): Export scale in meters
        crs (str): Coordinate reference system
        
    Returns:
        ee.batch.Task: Export task
    """
    try:
        if file_name is None:
            file_name = description
        
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=file_name,
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=1e13
        )
        
        task.start()
        logger.info(f"Started export task: {description}")
        
        return task
        
    except Exception as e:
        logger.error(f"Error starting export task: {e}")
        return None