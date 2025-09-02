"""
Configuration settings for China City Mass Index Map generation.

This module contains all configurable parameters for the data pipeline.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Google Earth Engine configuration
GEE_PROJECT_ID = "ee-knhuang"  # Update this with your GEE project ID

# H3 configuration
H3_RESOLUTION = 6  # H3 resolution level for neighborhood analysis

# Data file naming patterns
DATE_FORMAT = "%Y-%m-%d"

# File naming templates
FILENAME_TEMPLATES = {
    "h3_grids": "Fig3_H3_Grids_Neighborhood_Resolution{resolution}_{date}.csv",
    "volume_pavement": "Fig3_Volume_Pavement_Neighborhood_H3_Resolution{resolution}_{date}.csv", 
    "roads": "Fig3_Roads_Neighborhood_H3_Resolution{resolution}_{date}.csv",
    "merged": "Fig3_Merged_Neighborhood_H3_Resolution{resolution}_{date}.csv",
    "mass": "Fig3_Mass_Neighborhood_H3_Resolution{resolution}_{date}.csv",
    "city_boundaries": "all_cities_boundaries.gpkg",
    "h3_boundaries": "all_cities_h3_grids.gpkg",
    "china_cities": "china_city_statistics.csv",
    "china_neighborhoods": "china_neighborhoods_cmi.csv",
    "final_map": "china_dual_layer_cmi_multiplier_map_{timestamp}.html"
}

# Output directories
OUTPUT_DIRS = {
    "neighborhood_data": DATA_DIR / "processed" / "neighborhood_data",
    "merged_data": DATA_DIR / "processed" / "merged_data", 
    "boundaries": DATA_DIR / "processed" / "boundaries",
    "statistics": RESULTS_DIR / "statistics",
    "maps": RESULTS_DIR / "maps"
}

# Input data sources
INPUT_DATA = {
    "china_boundaries": DATA_DIR / "raw" / "ChinaAdminProvince.gpkg",
    "country_classification": DATA_DIR / "raw" / "country_classification.csv",
    "gee_urban_centres": "users/kh3657/GHS_STAT_UCDB2015"
}

# Building material intensity parameters
BUILDING_CLASSES = {
    'LW': 'Lightweight (<3m)',
    'RS': 'Residential single-family (3-12m)', 
    'RM': 'Residential multi-family (12-50m)',
    'NR': 'Non-residential (3-50m)',
    'HR': 'High-rise (50-100m)'
}

# Material intensities (tonnes per m³ of built volume)
MATERIAL_INTENSITIES = {
    # Steel intensities by building class and climate
    'steel': {
        'LW': {'temperate': 0.020, 'tropical': 0.018, 'continental': 0.022, 'default': 0.020},
        'RS': {'temperate': 0.045, 'tropical': 0.040, 'continental': 0.050, 'default': 0.045},
        'RM': {'temperate': 0.065, 'tropical': 0.058, 'continental': 0.072, 'default': 0.065},
        'NR': {'temperate': 0.080, 'tropical': 0.070, 'continental': 0.090, 'default': 0.080},
        'HR': {'temperate': 0.120, 'tropical': 0.105, 'continental': 0.135, 'default': 0.120}
    },
    # Concrete intensities by building class and climate
    'concrete': {
        'LW': {'temperate': 0.250, 'tropical': 0.220, 'continental': 0.280, 'default': 0.250},
        'RS': {'temperate': 0.400, 'tropical': 0.360, 'continental': 0.440, 'default': 0.400},
        'RM': {'temperate': 0.600, 'tropical': 0.540, 'continental': 0.660, 'default': 0.600},
        'NR': {'temperate': 0.750, 'tropical': 0.675, 'continental': 0.825, 'default': 0.750},
        'HR': {'temperate': 1.100, 'tropical': 0.990, 'continental': 1.210, 'default': 1.100}
    }
}

# Road material parameters
ROAD_MATERIALS = {
    'asphalt_thickness_m': 0.15,  # 15cm asphalt layer
    'concrete_thickness_m': 0.20,  # 20cm concrete base
    'asphalt_density_tonnes_m3': 2.3,
    'concrete_density_tonnes_m3': 2.4
}

# Climate classification mapping
CLIMATE_MAPPING = {
    'A': 'tropical',    # Tropical climates
    'B': 'arid',        # Arid climates  
    'C': 'temperate',   # Temperate climates
    'D': 'continental', # Continental climates
    'E': 'polar'        # Polar climates
}

# Statistical analysis parameters
FIXED_SLOPE = 0.75  # Fixed slope from literature for neighborhood-level scaling
PERCENTILE_RANGE = (5, 95)  # Percentile range for outlier detection

# UMIC countries (will be loaded from CSV)
UMIC_COUNTRIES = []  # Populated from country_classification.csv

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'DEBUG', 
            'class': 'logging.FileHandler',
            'filename': 'pipeline.log',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

def ensure_directories():
    """Ensure all required directories exist."""
    for dir_path in OUTPUT_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Also ensure raw data directory exists
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)

def get_timestamped_filename(template_key, **kwargs):
    """Get a filename with current timestamp."""
    from datetime import datetime
    
    template = FILENAME_TEMPLATES[template_key]
    timestamp = datetime.now().strftime(DATE_FORMAT)
    
    return template.format(
        resolution=H3_RESOLUTION,
        date=timestamp,
        timestamp=timestamp,
        **kwargs
    )

def load_umic_countries():
    """Load UMIC countries from classification file."""
    global UMIC_COUNTRIES
    
    import pandas as pd
    
    try:
        class_data = pd.read_csv(INPUT_DATA["country_classification"])
        UMIC_COUNTRIES = class_data[class_data['Income group'] == 'Upper middle income']['Code'].tolist()
        return UMIC_COUNTRIES
    except FileNotFoundError:
        print(f"Warning: Country classification file not found at {INPUT_DATA['country_classification']}")
        return []

if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"H3 resolution: {H3_RESOLUTION}")
    print(f"Output directories created: {len(OUTPUT_DIRS)}")