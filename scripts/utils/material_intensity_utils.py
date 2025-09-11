"""
Material intensity utilities for calculating building material masses.

This module contains functions for applying material intensities to building
volumes and calculating total material stocks based on climate zones.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Climate-adjusted material intensities (kg/m³) from literature
# Based on Haberl et al. (2021), Heeren (2019), and other studies
MATERIAL_INTENSITIES = {
    'steel': {
        'HR': {'temperate': 0.120, 'tropical': 0.105, 'default': 0.120},
        'MR': {'temperate': 0.085, 'tropical': 0.075, 'default': 0.085},
        'LR': {'temperate': 0.055, 'tropical': 0.048, 'default': 0.055},
        'default': {'temperate': 0.085, 'tropical': 0.075, 'default': 0.085}
    },
    'concrete': {
        'HR': {'temperate': 2.450, 'tropical': 2.200, 'default': 2.450},
        'MR': {'temperate': 1.850, 'tropical': 1.650, 'default': 1.850},
        'LR': {'temperate': 1.250, 'tropical': 1.100, 'default': 1.250},
        'default': {'temperate': 1.850, 'tropical': 1.650, 'default': 1.850}
    },
    'brick': {
        'HR': {'temperate': 1.950, 'tropical': 2.100, 'default': 1.950},
        'MR': {'temperate': 2.150, 'tropical': 2.300, 'default': 2.150},
        'LR': {'temperate': 2.450, 'tropical': 2.600, 'default': 2.450},
        'default': {'temperate': 2.150, 'tropical': 2.300, 'default': 2.150}
    },
    'aluminum': {
        'HR': {'temperate': 0.015, 'tropical': 0.013, 'default': 0.015},
        'MR': {'temperate': 0.012, 'tropical': 0.010, 'default': 0.012},
        'LR': {'temperate': 0.008, 'tropical': 0.007, 'default': 0.008},
        'default': {'temperate': 0.012, 'tropical': 0.010, 'default': 0.012}
    },
    'glass': {
        'HR': {'temperate': 0.085, 'tropical': 0.075, 'default': 0.085},
        'MR': {'temperate': 0.065, 'tropical': 0.058, 'default': 0.065},
        'LR': {'temperate': 0.035, 'tropical': 0.030, 'default': 0.035},
        'default': {'temperate': 0.065, 'tropical': 0.058, 'default': 0.065}
    },
    'timber': {
        'HR': {'temperate': 0.250, 'tropical': 0.180, 'default': 0.250},
        'MR': {'temperate': 0.350, 'tropical': 0.280, 'default': 0.350},
        'LR': {'temperate': 0.450, 'tropical': 0.380, 'default': 0.450},
        'default': {'temperate': 0.350, 'tropical': 0.280, 'default': 0.350}
    }
}

# Road material intensities (tonnes per km)
ROAD_MATERIAL_INTENSITIES = {
    'motorway': {'concrete': 2500, 'steel': 150, 'asphalt': 1800},
    'trunk': {'concrete': 1800, 'steel': 100, 'asphalt': 1400},
    'primary': {'concrete': 1200, 'steel': 75, 'asphalt': 1000},
    'secondary': {'concrete': 800, 'steel': 50, 'asphalt': 700},
    'tertiary': {'concrete': 400, 'steel': 25, 'asphalt': 500},
    'default': {'concrete': 800, 'steel': 50, 'asphalt': 700}
}

class MaterialIntensityCalculator:
    """Calculates material masses from building volumes and infrastructure."""
    
    def __init__(self, 
                 material_intensities: Dict = None,
                 road_intensities: Dict = None):
        """
        Initialize calculator with material intensities.
        
        Args:
            material_intensities: Building material intensities (kg/m³)
            road_intensities: Road material intensities (tonnes/km)
        """
        self.material_intensities = material_intensities or MATERIAL_INTENSITIES
        self.road_intensities = road_intensities or ROAD_MATERIAL_INTENSITIES
        
        logger.info("MaterialIntensityCalculator initialized")
    
    def classify_building_class(self, 
                               population_density: float,
                               built_volume_density: float) -> str:
        """
        Classify buildings into HR/MR/LR categories.
        
        Args:
            population_density: Population per sq km
            built_volume_density: Built volume per sq km (m³/km²)
            
        Returns:
            Building class ('HR', 'MR', or 'LR')
        """
        try:
            # Calculate volume per person
            if population_density > 0:
                volume_per_person = built_volume_density / population_density
            else:
                volume_per_person = 0
            
            # Classification thresholds based on literature
            if volume_per_person > 100:  # High-rise, low density
                return 'HR'
            elif volume_per_person > 50:   # Mid-rise
                return 'MR'
            else:                        # Low-rise, high density
                return 'LR'
                
        except Exception as e:
            logger.warning(f"Error classifying building class: {e}")
            return 'MR'  # Default to mid-rise
    
    def determine_climate_zone(self, latitude: float) -> str:
        """
        Determine climate zone based on latitude.
        
        Args:
            latitude: Latitude in decimal degrees
            
        Returns:
            Climate zone ('temperate' or 'tropical')
        """
        # Simplified climate classification
        if abs(latitude) < 23.5:  # Within tropics
            return 'tropical'
        else:
            return 'temperate'
    
    def calculate_building_materials(self, 
                                   df: pd.DataFrame,
                                   volume_col: str = 'building_volume_m3',
                                   pop_col: str = 'population',
                                   area_col: str = 'area_sqkm',
                                   lat_col: str = 'latitude') -> pd.DataFrame:
        """
        Calculate building material masses from volume data.
        
        Args:
            df: Input dataframe with building volumes
            volume_col: Building volume column name
            pop_col: Population column name
            area_col: Area column name
            lat_col: Latitude column name
            
        Returns:
            Dataframe with material mass columns added
        """
        try:
            logger.info(f"Calculating building materials for {len(df)} neighborhoods")
            
            df = df.copy()
            
            # Calculate densities
            df['pop_density_sqkm'] = df[pop_col] / df[area_col]
            df['volume_density_m3_sqkm'] = df[volume_col] / df[area_col]
            
            # Classify building types
            df['building_class'] = df.apply(
                lambda row: self.classify_building_class(
                    row['pop_density_sqkm'], 
                    row['volume_density_m3_sqkm']
                ), axis=1
            )
            
            # Determine climate zones
            df['climate_zone'] = df[lat_col].apply(self.determine_climate_zone)
            
            # Calculate material masses for each material type
            for material in self.material_intensities.keys():
                mass_col = f'{material}_mass_tonnes'
                
                df[mass_col] = df.apply(
                    lambda row: self._calculate_single_material_mass(
                        row[volume_col], 
                        material, 
                        row['building_class'], 
                        row['climate_zone']
                    ), axis=1
                )
            
            # Calculate total building mass
            material_cols = [f'{mat}_mass_tonnes' for mat in self.material_intensities.keys()]
            df['total_building_mass_tonnes'] = df[material_cols].sum(axis=1)
            
            logger.info("Building material calculation completed")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating building materials: {e}")
            return df
    
    def _calculate_single_material_mass(self, 
                                       volume: float, 
                                       material: str, 
                                       building_class: str, 
                                       climate: str) -> float:
        """
        Calculate mass for a single material type.
        
        Args:
            volume: Building volume in m³
            material: Material type
            building_class: Building class (HR/MR/LR)
            climate: Climate zone (temperate/tropical)
            
        Returns:
            Material mass in tonnes
        """
        try:
            if pd.isna(volume) or volume <= 0:
                return 0.0
            
            # Get material intensities
            material_dict = self.material_intensities.get(material, {})
            class_dict = material_dict.get(building_class, material_dict.get('default', {}))
            intensity = class_dict.get(climate, class_dict.get('default', 0))
            
            # Convert from kg/m³ to tonnes/m³ and calculate mass
            mass_tonnes = volume * intensity / 1000
            
            return max(0, mass_tonnes)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Error calculating {material} mass: {e}")
            return 0.0
    
    def calculate_road_materials(self, 
                                df: pd.DataFrame,
                                road_length_cols: list) -> pd.DataFrame:
        """
        Calculate road material masses from road length data.
        
        Args:
            df: Input dataframe with road lengths
            road_length_cols: List of road length column names
            
        Returns:
            Dataframe with road material columns added
        """
        try:
            logger.info(f"Calculating road materials for {len(df)} neighborhoods")
            
            df = df.copy()
            
            # Initialize material columns
            for material in ['concrete', 'steel', 'asphalt']:
                df[f'road_{material}_mass_tonnes'] = 0.0
            
            # Process each road type
            for road_col in road_length_cols:
                if road_col not in df.columns:
                    continue
                
                # Extract road type from column name
                road_type = self._extract_road_type(road_col)
                intensities = self.road_intensities.get(road_type, 
                                                       self.road_intensities['default'])
                
                # Convert length from meters to kilometers
                road_length_km = df[road_col] / 1000
                
                # Calculate material masses
                for material, intensity in intensities.items():
                    df[f'road_{material}_mass_tonnes'] += road_length_km * intensity / 1000
            
            # Calculate total road mass
            road_material_cols = [f'road_{mat}_mass_tonnes' 
                                for mat in ['concrete', 'steel', 'asphalt']]
            df['total_road_mass_tonnes'] = df[road_material_cols].sum(axis=1)
            
            logger.info("Road material calculation completed")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating road materials: {e}")
            return df
    
    def _extract_road_type(self, column_name: str) -> str:
        """
        Extract road type from column name.
        
        Args:
            column_name: Column name containing road type
            
        Returns:
            Road type string
        """
        # Common patterns in road length column names
        road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
        
        column_lower = column_name.lower()
        for road_type in road_types:
            if road_type in column_lower:
                return road_type
        
        return 'default'
    
    def calculate_total_mass(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total material mass (buildings + roads).
        
        Args:
            df: Dataframe with building and road masses
            
        Returns:
            Dataframe with total mass column
        """
        try:
            df = df.copy()
            
            building_mass = df.get('total_building_mass_tonnes', 0)
            road_mass = df.get('total_road_mass_tonnes', 0)
            
            df['total_material_mass_tonnes'] = building_mass + road_mass
            
            logger.info(f"Calculated total material mass for {len(df)} neighborhoods")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating total mass: {e}")
            return df

def get_default_material_intensities() -> Dict[str, Any]:
    """
    Get default material intensities.
    
    Returns:
        Dictionary of material intensities
    """
    return MATERIAL_INTENSITIES.copy()

def get_default_road_intensities() -> Dict[str, Any]:
    """
    Get default road material intensities.
    
    Returns:
        Dictionary of road material intensities
    """
    return ROAD_MATERIAL_INTENSITIES.copy()

def validate_material_data(df: pd.DataFrame, 
                          required_cols: list = None) -> bool:
    """
    Validate material calculation input data.
    
    Args:
        df: Input dataframe
        required_cols: Required column names
        
    Returns:
        True if data is valid
    """
    if required_cols is None:
        required_cols = ['building_volume_m3', 'population', 'area_sqkm']
    
    try:
        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for reasonable data ranges
        if 'building_volume_m3' in df.columns:
            if df['building_volume_m3'].max() > 1e12:  # Unreasonably large
                logger.warning("Building volumes seem unreasonably large")
        
        if 'population' in df.columns:
            if df['population'].min() < 0:
                logger.warning("Negative population values found")
        
        logger.info("Material data validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating material data: {e}")
        return False