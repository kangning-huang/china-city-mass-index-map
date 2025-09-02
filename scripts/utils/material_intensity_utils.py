"""
Material intensity calculation utilities for building and infrastructure materials.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BUILDING_CLASSES, MATERIAL_INTENSITIES, ROAD_MATERIALS, CLIMATE_MAPPING

logger = logging.getLogger(__name__)

def get_climate_category(koppen_code: str) -> str:
    """
    Convert Köppen climate code to broad climate category.
    
    Args:
        koppen_code (str): Köppen climate classification code
        
    Returns:
        str: Broad climate category
    """
    if pd.isna(koppen_code) or koppen_code == '':
        return 'default'
    
    # Use first letter of Köppen code
    first_letter = koppen_code[0].upper()
    return CLIMATE_MAPPING.get(first_letter, 'default')

def calculate_building_material_mass(
    volume_m3: float,
    building_class: str,
    climate_code: str = 'default',
    material_type: str = 'steel'
) -> float:
    """
    Calculate material mass for building volume.
    
    Args:
        volume_m3 (float): Building volume in m³
        building_class (str): Building class code (LW, RS, RM, NR, HR)
        climate_code (str): Köppen climate code
        material_type (str): Material type ('steel' or 'concrete')
        
    Returns:
        float: Material mass in tonnes
    """
    try:
        if pd.isna(volume_m3) or volume_m3 <= 0:
            return 0.0
        
        # Get climate category
        climate_category = get_climate_category(climate_code)
        
        # Get material intensity
        if material_type not in MATERIAL_INTENSITIES:
            logger.warning(f"Unknown material type: {material_type}")
            return 0.0
        
        if building_class not in MATERIAL_INTENSITIES[material_type]:
            logger.warning(f"Unknown building class: {building_class}")
            return 0.0
        
        climate_intensities = MATERIAL_INTENSITIES[material_type][building_class]
        intensity = climate_intensities.get(climate_category, climate_intensities['default'])
        
        # Calculate mass
        mass_tonnes = volume_m3 * intensity
        
        return max(0.0, mass_tonnes)
        
    except Exception as e:
        logger.error(f"Error calculating building material mass: {e}")
        return 0.0

def calculate_road_material_mass(
    road_area_m2: float,
    material_type: str = 'asphalt'
) -> float:
    """
    Calculate material mass for road area.
    
    Args:
        road_area_m2 (float): Road area in m²
        material_type (str): Material type ('asphalt' or 'concrete')
        
    Returns:
        float: Material mass in tonnes
    """
    try:
        if pd.isna(road_area_m2) or road_area_m2 <= 0:
            return 0.0
        
        if material_type == 'asphalt':
            thickness = ROAD_MATERIALS['asphalt_thickness_m']
            density = ROAD_MATERIALS['asphalt_density_tonnes_m3']
        elif material_type == 'concrete':
            thickness = ROAD_MATERIALS['concrete_thickness_m'] 
            density = ROAD_MATERIALS['concrete_density_tonnes_m3']
        else:
            logger.warning(f"Unknown road material type: {material_type}")
            return 0.0
        
        # Calculate volume and mass
        volume_m3 = road_area_m2 * thickness
        mass_tonnes = volume_m3 * density
        
        return max(0.0, mass_tonnes)
        
    except Exception as e:
        logger.error(f"Error calculating road material mass: {e}")
        return 0.0

def process_building_volumes_to_mass(
    df: pd.DataFrame,
    volume_columns: List[str],
    climate_column: str = 'climate_class',
    id_columns: List[str] = ['h3index']
) -> pd.DataFrame:
    """
    Convert building volume columns to material mass.
    
    Args:
        df (pd.DataFrame): Input dataframe with volume columns
        volume_columns (List[str]): Column names containing volume data
        climate_column (str): Column name with climate data
        id_columns (List[str]): ID columns to preserve
        
    Returns:
        pd.DataFrame: Dataframe with material mass columns
    """
    try:
        logger.info(f"Processing {len(volume_columns)} volume columns to material mass")
        
        # Start with ID columns
        result_df = df[id_columns + [climate_column]].copy()
        
        # Process each volume column
        mass_columns = []
        
        for vol_col in volume_columns:
            if vol_col not in df.columns:
                logger.warning(f"Volume column {vol_col} not found in dataframe")
                continue
            
            # Parse column name to extract data source and building class
            # Expected format: vol_DataSource_BuildingClass
            parts = vol_col.split('_')
            if len(parts) < 3:
                logger.warning(f"Cannot parse volume column name: {vol_col}")
                continue
            
            data_source = parts[1]
            building_class = parts[2]
            
            if building_class not in BUILDING_CLASSES:
                logger.warning(f"Unknown building class in column {vol_col}: {building_class}")
                continue
            
            # Calculate steel and concrete masses
            for material_type in ['steel', 'concrete']:
                mass_col = f'mass_{material_type}_{data_source}_{building_class}_tons'
                
                result_df[mass_col] = df.apply(
                    lambda row: calculate_building_material_mass(
                        row[vol_col],
                        building_class,
                        row[climate_column],
                        material_type
                    ), 
                    axis=1
                )
                
                mass_columns.append(mass_col)
        
        logger.info(f"Created {len(mass_columns)} material mass columns")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing building volumes to mass: {e}")
        return df

def aggregate_material_masses(
    df: pd.DataFrame,
    mass_columns: List[str],
    group_by: Optional[List[str]] = None,
    aggregation_method: str = 'sum'
) -> pd.DataFrame:
    """
    Aggregate material masses by specified grouping.
    
    Args:
        df (pd.DataFrame): Input dataframe with mass columns
        mass_columns (List[str]): Columns containing mass data
        group_by (List[str], optional): Columns to group by
        aggregation_method (str): Aggregation method ('sum', 'mean', etc.)
        
    Returns:
        pd.DataFrame: Aggregated dataframe
    """
    try:
        if group_by is None:
            # Calculate total masses across all mass columns
            steel_cols = [col for col in mass_columns if 'steel' in col]
            concrete_cols = [col for col in mass_columns if 'concrete' in col]
            
            result_df = df.copy()
            result_df['total_steel_mass_tons'] = df[steel_cols].sum(axis=1)
            result_df['total_concrete_mass_tons'] = df[concrete_cols].sum(axis=1)
            result_df['total_built_mass_tons'] = result_df['total_steel_mass_tons'] + result_df['total_concrete_mass_tons']
            
        else:
            # Group by specified columns
            agg_dict = {col: aggregation_method for col in mass_columns}
            result_df = df.groupby(group_by).agg(agg_dict).reset_index()
            
            # Calculate totals
            steel_cols = [col for col in mass_columns if 'steel' in col]
            concrete_cols = [col for col in mass_columns if 'concrete' in col]
            
            if steel_cols:
                result_df['total_steel_mass_tons'] = result_df[steel_cols].sum(axis=1)
            if concrete_cols:
                result_df['total_concrete_mass_tons'] = result_df[concrete_cols].sum(axis=1)
            
            if steel_cols and concrete_cols:
                result_df['total_built_mass_tons'] = result_df['total_steel_mass_tons'] + result_df['total_concrete_mass_tons']
        
        logger.info(f"Aggregated material masses for {len(result_df)} records")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error aggregating material masses: {e}")
        return df

def calculate_material_intensity_per_capita(
    df: pd.DataFrame,
    population_column: str = 'population_2015',
    mass_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate material intensity per capita.
    
    Args:
        df (pd.DataFrame): Input dataframe
        population_column (str): Column with population data
        mass_columns (List[str], optional): Mass columns to calculate intensity for
        
    Returns:
        pd.DataFrame: Dataframe with per capita intensity columns
    """
    try:
        result_df = df.copy()
        
        if mass_columns is None:
            mass_columns = [col for col in df.columns if col.endswith('_tons')]
        
        # Calculate per capita intensities
        for mass_col in mass_columns:
            if mass_col not in df.columns:
                continue
                
            intensity_col = mass_col.replace('_tons', '_tons_per_capita')
            
            # Avoid division by zero
            result_df[intensity_col] = np.where(
                df[population_column] > 0,
                df[mass_col] / df[population_column],
                0.0
            )
        
        logger.info(f"Calculated per capita intensities for {len(mass_columns)} mass columns")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating per capita intensities: {e}")
        return df

def validate_material_calculations(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Validate material mass calculations and return summary statistics.
    
    Args:
        df (pd.DataFrame): Dataframe with calculated masses
        
    Returns:
        dict: Validation statistics
    """
    try:
        stats = {}
        
        # Find mass columns
        mass_columns = [col for col in df.columns if col.endswith('_tons')]
        
        for col in mass_columns:
            stats[f'{col}_count'] = df[col].count()
            stats[f'{col}_sum'] = df[col].sum()
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_median'] = df[col].median()
            stats[f'{col}_min'] = df[col].min()
            stats[f'{col}_max'] = df[col].max()
            stats[f'{col}_zeros'] = (df[col] == 0).sum()
            stats[f'{col}_negatives'] = (df[col] < 0).sum()
            stats[f'{col}_nulls'] = df[col].isnull().sum()
        
        # Overall statistics
        stats['total_records'] = len(df)
        stats['total_mass_columns'] = len(mass_columns)
        
        logger.info(f"Material calculation validation completed for {len(df)} records")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error validating material calculations: {e}")
        return {}

def apply_material_intensity_corrections(
    df: pd.DataFrame,
    correction_factors: Dict[str, float],
    mass_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply correction factors to material intensities.
    
    Args:
        df (pd.DataFrame): Input dataframe
        correction_factors (Dict[str, float]): Correction factors by material type
        mass_columns (List[str], optional): Columns to apply corrections to
        
    Returns:
        pd.DataFrame: Dataframe with corrected masses
    """
    try:
        result_df = df.copy()
        
        if mass_columns is None:
            mass_columns = [col for col in df.columns if col.endswith('_tons')]
        
        corrections_applied = 0
        
        for col in mass_columns:
            # Check if any correction factor applies to this column
            for material_type, factor in correction_factors.items():
                if material_type.lower() in col.lower():
                    result_df[col] = df[col] * factor
                    corrections_applied += 1
                    logger.debug(f"Applied {factor}x correction to {col}")
                    break
        
        logger.info(f"Applied corrections to {corrections_applied} columns")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error applying material intensity corrections: {e}")
        return df