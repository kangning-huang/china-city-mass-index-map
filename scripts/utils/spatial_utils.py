"""
Spatial processing utilities including H3 hexagon generation and geometric operations.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
from tobler.util import h3fy
import numpy as np
import logging
from typing import List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def generate_h3_grids(
    city_gdf: gpd.GeoDataFrame, 
    resolution: int = 6,
    buffer_distance: float = 0.001,
    clip_to_boundary: bool = True
) -> Optional[gpd.GeoDataFrame]:
    """
    Generate H3 hexagonal grids for a city boundary.
    
    Args:
        city_gdf (gpd.GeoDataFrame): City boundary geometry
        resolution (int): H3 resolution level (0-15)
        buffer_distance (float): Buffer distance for grid generation
        clip_to_boundary (bool): Whether to clip grids to city boundary
        
    Returns:
        gpd.GeoDataFrame: H3 grid polygons with city information
    """
    try:
        logger.info(f"Generating H3 grids at resolution {resolution} for {len(city_gdf)} cities")
        
        # Ensure we have a valid CRS
        if city_gdf.crs is None:
            city_gdf = city_gdf.set_crs('EPSG:4326')
        
        # Generate H3 grids using tobler
        h3_grids = h3fy(city_gdf, resolution=resolution, clip=clip_to_boundary, buffer=True)
        
        if h3_grids is None or h3_grids.empty:
            logger.error("Failed to generate H3 grids")
            return None
        
        # Handle h3index column naming issues from tobler
        if h3_grids.index.name == 'hex_id':
            h3_grids.index.name = 'h3index'
        
        if 'h3index' not in h3_grids.columns:
            if h3_grids.index.name == 'h3index':
                h3_grids = h3_grids.reset_index()
            else:
                logger.error("Could not determine h3index column")
                return None
        
        # Add city information if not already present
        city_cols = ['ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2']
        
        # Get available city columns
        available_city_cols = [col for col in city_cols if col in city_gdf.columns and col not in h3_grids.columns]
        
        if available_city_cols:
            # Clean up any existing index columns that might conflict
            if 'index_right' in h3_grids.columns:
                h3_grids = h3_grids.drop(columns=['index_right'])
            
            # Prepare city data for join
            join_cols = ['geometry'] + available_city_cols
            city_join_data = city_gdf[join_cols].copy().reset_index(drop=True)
            
            # Spatial join to add city information
            h3_grids = gpd.sjoin(h3_grids, city_join_data, how='left', predicate='intersects')
            
            # Clean up join artifacts
            if 'index_right' in h3_grids.columns:
                h3_grids = h3_grids.drop(columns=['index_right'])
        
        # Create unique neighborhood identifier
        if 'ID_HDC_G0' in h3_grids.columns:
            h3_grids['neighborhood_id'] = h3_grids['h3index'] + '_' + h3_grids['ID_HDC_G0'].astype(str)
        
        logger.info(f"Generated {len(h3_grids)} H3 hexagons")
        
        return h3_grids
        
    except Exception as e:
        logger.error(f"Error generating H3 grids: {e}")
        return None

def h3_to_polygon(h3_address: str) -> Polygon:
    """
    Convert H3 address to Shapely Polygon.
    
    Args:
        h3_address (str): H3 hexagon address
        
    Returns:
        Polygon: Hexagon geometry
    """
    try:
        # Get boundary coordinates
        boundary = h3.h3_to_geo_boundary(h3_address, geo_json=True)
        
        # Create polygon (note: h3 returns [lng, lat] but Shapely expects [lng, lat])
        return Polygon(boundary)
        
    except Exception as e:
        logger.error(f"Error converting H3 {h3_address} to polygon: {e}")
        return None

def get_h3_neighbors(h3_address: str, k: int = 1) -> List[str]:
    """
    Get neighboring H3 hexagons.
    
    Args:
        h3_address (str): Center H3 address
        k (int): Number of rings of neighbors
        
    Returns:
        list: List of neighbor H3 addresses
    """
    try:
        neighbors = h3.k_ring(h3_address, k)
        return list(neighbors)
        
    except Exception as e:
        logger.error(f"Error getting H3 neighbors for {h3_address}: {e}")
        return []

def calculate_h3_area_km2(h3_address: str) -> float:
    """
    Calculate area of H3 hexagon in km².
    
    Args:
        h3_address (str): H3 address
        
    Returns:
        float: Area in km²
    """
    try:
        # Get area in m² and convert to km²
        area_m2 = h3.hex_area(h3.h3_get_resolution(h3_address), unit='m^2')
        return area_m2 / 1_000_000
        
    except Exception as e:
        logger.error(f"Error calculating H3 area for {h3_address}: {e}")
        return 0.0

def point_to_h3(lat: float, lng: float, resolution: int = 6) -> str:
    """
    Convert lat/lng point to H3 address.
    
    Args:
        lat (float): Latitude
        lng (float): Longitude  
        resolution (int): H3 resolution
        
    Returns:
        str: H3 address
    """
    try:
        return h3.geo_to_h3(lat, lng, resolution)
        
    except Exception as e:
        logger.error(f"Error converting point ({lat}, {lng}) to H3: {e}")
        return None

def calculate_polygon_areas(gdf: gpd.GeoDataFrame, unit: str = 'km2') -> pd.Series:
    """
    Calculate areas of polygons in a GeoDataFrame.
    
    Args:
        gdf (gpd.GeoDataFrame): Input geodataframe
        unit (str): Area unit ('km2', 'm2', or 'ha')
        
    Returns:
        pd.Series: Areas in specified unit
    """
    try:
        # Project to appropriate CRS for area calculation
        if gdf.crs.to_string() != 'EPSG:3857':  # Web Mercator
            gdf_projected = gdf.to_crs('EPSG:3857')
        else:
            gdf_projected = gdf
        
        # Calculate areas in m²
        areas_m2 = gdf_projected.geometry.area
        
        # Convert to requested unit
        if unit.lower() == 'km2':
            return areas_m2 / 1_000_000
        elif unit.lower() == 'ha':
            return areas_m2 / 10_000
        else:  # m2
            return areas_m2
            
    except Exception as e:
        logger.error(f"Error calculating polygon areas: {e}")
        return pd.Series(dtype=float)

def create_buffer(gdf: gpd.GeoDataFrame, distance: float, unit: str = 'meters') -> gpd.GeoDataFrame:
    """
    Create buffer around geometries.
    
    Args:
        gdf (gpd.GeoDataFrame): Input geodataframe
        distance (float): Buffer distance
        unit (str): Distance unit ('meters', 'km', or 'degrees')
        
    Returns:
        gpd.GeoDataFrame: Buffered geometries
    """
    try:
        buffered_gdf = gdf.copy()
        
        if unit.lower() == 'degrees':
            # Use geographic degrees directly
            buffered_gdf['geometry'] = gdf.geometry.buffer(distance)
        else:
            # Project to appropriate CRS for metric distances
            if unit.lower() == 'km':
                distance = distance * 1000  # Convert to meters
            
            # Use an appropriate projected CRS
            original_crs = gdf.crs
            projected_crs = 'EPSG:3857'  # Web Mercator
            
            # Buffer in projected coordinates
            gdf_proj = gdf.to_crs(projected_crs)
            buffered_proj = gdf_proj.copy()
            buffered_proj['geometry'] = gdf_proj.geometry.buffer(distance)
            
            # Transform back to original CRS
            buffered_gdf = buffered_proj.to_crs(original_crs)
        
        logger.info(f"Created {distance} {unit} buffer for {len(gdf)} geometries")
        
        return buffered_gdf
        
    except Exception as e:
        logger.error(f"Error creating buffer: {e}")
        return gdf

def spatial_join_largest_overlap(
    left_gdf: gpd.GeoDataFrame, 
    right_gdf: gpd.GeoDataFrame,
    left_id_col: str,
    right_cols: List[str]
) -> gpd.GeoDataFrame:
    """
    Spatial join based on largest overlap between polygons.
    
    Args:
        left_gdf (gpd.GeoDataFrame): Left dataset
        right_gdf (gpd.GeoDataFrame): Right dataset
        left_id_col (str): ID column in left dataset
        right_cols (List[str]): Columns to join from right dataset
        
    Returns:
        gpd.GeoDataFrame: Joined dataset
    """
    try:
        # Ensure same CRS
        if left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        
        # Find overlaps
        overlaps = gpd.overlay(left_gdf, right_gdf, how='intersection')
        
        if overlaps.empty:
            logger.warning("No spatial overlaps found")
            return left_gdf
        
        # Calculate overlap areas
        overlaps['overlap_area'] = calculate_polygon_areas(overlaps, unit='m2')
        
        # Find largest overlap for each left polygon
        largest_overlaps = overlaps.loc[overlaps.groupby(left_id_col)['overlap_area'].idxmax()]
        
        # Join back to left dataset
        join_cols = [left_id_col] + right_cols
        result_gdf = left_gdf.merge(
            largest_overlaps[join_cols], 
            on=left_id_col, 
            how='left'
        )
        
        logger.info(f"Spatial join completed with {len(largest_overlaps)} matches")
        
        return result_gdf
        
    except Exception as e:
        logger.error(f"Error in spatial join: {e}")
        return left_gdf

def dissolve_by_attribute(gdf: gpd.GeoDataFrame, by: str, agg_dict: Optional[dict] = None) -> gpd.GeoDataFrame:
    """
    Dissolve geometries by attribute with aggregation.
    
    Args:
        gdf (gpd.GeoDataFrame): Input geodataframe
        by (str): Column to dissolve by
        agg_dict (dict, optional): Aggregation dictionary for other columns
        
    Returns:
        gpd.GeoDataFrame: Dissolved geometries
    """
    try:
        if agg_dict is None:
            dissolved = gdf.dissolve(by=by).reset_index()
        else:
            dissolved = gdf.dissolve(by=by, aggfunc=agg_dict).reset_index()
        
        logger.info(f"Dissolved {len(gdf)} geometries to {len(dissolved)} by {by}")
        
        return dissolved
        
    except Exception as e:
        logger.error(f"Error dissolving geometries: {e}")
        return gdf

def validate_geometries(gdf: gpd.GeoDataFrame, fix_invalid: bool = True) -> gpd.GeoDataFrame:
    """
    Validate and optionally fix invalid geometries.
    
    Args:
        gdf (gpd.GeoDataFrame): Input geodataframe
        fix_invalid (bool): Whether to attempt fixing invalid geometries
        
    Returns:
        gpd.GeoDataFrame: Validated geodataframe
    """
    try:
        # Check for invalid geometries
        invalid_mask = ~gdf.geometry.is_valid
        num_invalid = invalid_mask.sum()
        
        if num_invalid > 0:
            logger.warning(f"Found {num_invalid} invalid geometries")
            
            if fix_invalid:
                # Attempt to fix using buffer(0)
                gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
                
                # Check if fixed
                still_invalid = ~gdf.geometry.is_valid
                num_still_invalid = still_invalid.sum()
                
                if num_still_invalid > 0:
                    logger.warning(f"Could not fix {num_still_invalid} geometries")
                else:
                    logger.info("All invalid geometries fixed")
        
        return gdf
        
    except Exception as e:
        logger.error(f"Error validating geometries: {e}")
        return gdf