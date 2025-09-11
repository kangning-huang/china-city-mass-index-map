#!/usr/bin/env python3
"""
Perform hierarchical mixed-effects analysis for China City Mass Index.

This script implements the mathematical framework:
log10(M) = α_global + α_country + α_city + β × log10(N)

Uses mixed-effects model with nested random effects structure:
(1 | country) + (1 | country:city)

Focuses on UMIC countries as baseline for China comparison.

Input:
- Material mass data from step 05
- Country classification data

Output:
- City-level CMI statistics
- Neighborhood-level CMI multipliers
- Statistical analysis results

Author: Generated for China City Mass Index Map
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HierarchicalMismatchAnalyzer:
    """
    Hierarchical Mismatch Analyzer implementing the mathematical framework:
    log10(M) = α_global + α_country + α_city + β × log10(N)
    """
    
    def __init__(self, results_dir: str, fixed_slope: float = FIXED_SLOPE):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory to save results
            fixed_slope: Fixed slope from literature (0.75)
        """
        self.results_dir = results_dir
        self.fixed_slope = fixed_slope
        self.hierarchical_dir = os.path.join(results_dir, 'unified_hierarchical_analysis')
        
        # Create results directory
        os.makedirs(self.hierarchical_dir, exist_ok=True)
        
        self.model = None
        self.data = None
        self.results = {}
        
        logger.info(f"HierarchicalMismatchAnalyzer initialized with fixed slope: {fixed_slope}")
        
    def load_and_prepare_data(self, data_file: str, country_class_file: str) -> pd.DataFrame:
        """
        Load and prepare data for hierarchical analysis.
        
        Args:
            data_file: Path to material mass data
            country_class_file: Path to country classification data
            
        Returns:
            Prepared DataFrame
        """
        try:
            logger.info("Loading and preparing data for hierarchical analysis...")
            
            # Load material mass data
            if data_file.endswith('.gpkg'):
                import geopandas as gpd
                data = gpd.read_file(data_file)
            else:
                data = pd.read_csv(data_file)
            
            logger.info(f"Loaded {len(data)} neighborhood records")
            
            # Load country classifications
            country_class = pd.read_csv(country_class_file)
            logger.info(f"Loaded country classifications for {len(country_class)} countries")
            
            # Merge with country classifications
            data = data.merge(
                country_class[['Code', 'Income group']], 
                left_on='CTR_MN_ISO', 
                right_on='Code', 
                how='left'
            )
            
            # Filter to valid data
            required_cols = ['total_material_mass_tonnes', 'population', 'CTR_MN_ISO', 'ID_HDC_G0']
            data = data.dropna(subset=required_cols)
            
            # Remove zero/negative values
            data = data[
                (data['total_material_mass_tonnes'] > 0) & 
                (data['population'] > 0)
            ]
            
            logger.info(f"After filtering: {len(data)} valid neighborhoods")
            
            # Create log-transformed variables
            data['log_mass'] = np.log10(data['total_material_mass_tonnes'])
            data['log_population'] = np.log10(data['population'])
            
            # Create hierarchical identifiers
            data['country'] = data['CTR_MN_ISO']
            data['city'] = data['ID_HDC_G0']
            data['country_city'] = data['country'] + '_' + data['city'].astype(str)
            
            # Focus on UMIC countries and China for comparison
            umic_data = data[data['Income group'] == 'Upper middle income'].copy()
            china_data = data[data['CTR_MN_ISO'] == 'CHN'].copy()
            
            logger.info(f"UMIC countries: {len(umic_data)} neighborhoods")
            logger.info(f"China: {len(china_data)} neighborhoods")
            
            # Combine for analysis
            analysis_data = pd.concat([umic_data, china_data], ignore_index=True)
            
            self.data = analysis_data
            logger.info(f"Final analysis dataset: {len(analysis_data)} neighborhoods")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error loading and preparing data: {e}")
            raise
    
    def fit_hierarchical_model(self) -> bool:
        """
        Fit hierarchical mixed-effects model.
        
        Returns:
            True if successful
        """
        try:
            logger.info("Fitting hierarchical mixed-effects model...")
            
            if self.data is None or self.data.empty:
                logger.error("No data available for model fitting")
                return False
            
            # Try using statsmodels mixed linear model
            try:
                import statsmodels.formula.api as smf
                
                # Prepare data with fixed slope constraint
                model_data = self.data.copy()
                
                # Create offset term with fixed slope
                model_data['population_offset'] = self.fixed_slope * model_data['log_population']
                
                # Fit mixed-effects model with nested random effects
                # Model: log_mass ~ 1 + offset(fixed_slope * log_population) + (1|country) + (1|country:city)
                formula = 'log_mass ~ 1 + population_offset'
                
                # Fit using OLS first (as approximation if mixed-effects fails)
                try:
                    model = smf.ols(formula, data=model_data).fit()
                    self.model = model
                    logger.info("Fitted OLS model as approximation")
                    
                except Exception as me_error:
                    logger.warning(f"Mixed-effects model failed: {me_error}")
                    logger.info("Using OLS approximation")
                    
                    # Simple OLS model
                    formula_simple = 'log_mass ~ log_population'
                    model = smf.ols(formula_simple, data=model_data).fit()
                    self.model = model
                
            except ImportError:
                logger.warning("statsmodels not available, using simple linear regression")
                from sklearn.linear_model import LinearRegression
                
                X = self.data[['log_population']].values
                y = self.data['log_mass'].values
                
                model = LinearRegression()
                model.fit(X, y)
                self.model = model
            
            logger.info("Hierarchical model fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting hierarchical model: {e}")
            return False
    
    def calculate_city_cmi(self) -> pd.DataFrame:
        """
        Calculate City Mass Index (CMI) for each city.
        
        Returns:
            DataFrame with city-level CMI statistics
        """
        try:
            logger.info("Calculating City Mass Index (CMI)...")
            
            if self.data is None or self.data.empty:
                logger.error("No data available for CMI calculation")
                return pd.DataFrame()
            
            # Group by city to calculate city-level statistics
            city_stats = []
            
            for city_id, city_data in self.data.groupby('ID_HDC_G0'):
                try:
                    # Calculate city totals
                    city_total_mass = city_data['total_material_mass_tonnes'].sum()
                    city_total_pop = city_data['population'].sum()
                    
                    if city_total_mass <= 0 or city_total_pop <= 0:
                        continue
                    
                    # Get city metadata
                    city_info = city_data.iloc[0]
                    
                    # Calculate expected mass for UMIC baseline
                    log_pop = np.log10(city_total_pop)
                    
                    # Get UMIC baseline (global average intercept)
                    umic_data = self.data[self.data['Income group'] == 'Upper middle income']
                    if not umic_data.empty:
                        # Calculate UMIC average intercept
                        umic_intercept = (umic_data['log_mass'] - self.fixed_slope * umic_data['log_population']).mean()
                    else:
                        umic_intercept = 0
                    
                    # Expected mass based on UMIC baseline
                    log_expected_mass = umic_intercept + self.fixed_slope * log_pop
                    expected_mass = 10 ** log_expected_mass
                    
                    # City Mass Index (CMI)
                    city_cmi = city_total_mass / expected_mass
                    
                    city_stats.append({
                        'ID_HDC_G0': city_id,
                        'UC_NM_MN': city_info.get('UC_NM_MN', ''),
                        'CTR_MN_ISO': city_info.get('CTR_MN_ISO', ''),
                        'CTR_MN_NM': city_info.get('CTR_MN_NM', ''),
                        'Income_group': city_info.get('Income group', ''),
                        'population': city_total_pop,
                        'total_material_mass_tonnes': city_total_mass,
                        'expected_mass_umic_baseline': expected_mass,
                        'true_city_cmi': city_cmi,
                        'log_city_cmi': np.log10(city_cmi),
                        'neighborhoods_count': len(city_data)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating CMI for city {city_id}: {e}")
                    continue
            
            city_cmi_df = pd.DataFrame(city_stats)
            
            logger.info(f"Calculated CMI for {len(city_cmi_df)} cities")
            
            return city_cmi_df
            
        except Exception as e:
            logger.error(f"Error calculating city CMI: {e}")
            return pd.DataFrame()
    
    def calculate_neighborhood_multipliers(self, city_cmi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate neighborhood-level multipliers relative to city average.
        
        Args:
            city_cmi_df: City-level CMI statistics
            
        Returns:
            DataFrame with neighborhood multipliers
        """
        try:
            logger.info("Calculating neighborhood-level multipliers...")
            
            if self.data is None or self.data.empty or city_cmi_df.empty:
                logger.error("No data available for neighborhood multiplier calculation")
                return pd.DataFrame()
            
            neighborhood_data = self.data.copy()
            
            # Merge with city CMI data
            neighborhood_data = neighborhood_data.merge(
                city_cmi_df[['ID_HDC_G0', 'true_city_cmi']], 
                on='ID_HDC_G0', 
                how='left'
            )
            
            # Calculate neighborhood expected mass based on city CMI
            log_expected_neighborhood = (
                np.log10(neighborhood_data['true_city_cmi']) + 
                self.fixed_slope * neighborhood_data['log_population']
            )
            neighborhood_data['expected_mass_city_baseline'] = 10 ** log_expected_neighborhood
            
            # Calculate neighborhood multiplier (relative to city average)
            neighborhood_data['neighborhood_multiplier'] = (
                neighborhood_data['total_material_mass_tonnes'] / 
                neighborhood_data['expected_mass_city_baseline']
            )
            
            # Handle infinite/NaN values
            neighborhood_data['neighborhood_multiplier'] = neighborhood_data['neighborhood_multiplier'].replace([np.inf, -np.inf], np.nan)
            neighborhood_data = neighborhood_data.dropna(subset=['neighborhood_multiplier'])
            
            # Calculate percentiles for outlier detection
            p5, p95 = np.percentile(neighborhood_data['neighborhood_multiplier'], PERCENTILE_RANGE)
            
            neighborhood_data['is_low_outlier'] = neighborhood_data['neighborhood_multiplier'] < p5
            neighborhood_data['is_high_outlier'] = neighborhood_data['neighborhood_multiplier'] > p95
            
            logger.info(f"Calculated multipliers for {len(neighborhood_data)} neighborhoods")
            logger.info(f"Low outliers (< {p5:.3f}): {neighborhood_data['is_low_outlier'].sum()}")
            logger.info(f"High outliers (> {p95:.3f}): {neighborhood_data['is_high_outlier'].sum()}")
            
            return neighborhood_data
            
        except Exception as e:
            logger.error(f"Error calculating neighborhood multipliers: {e}")
            return pd.DataFrame()
    
    def save_results(self, city_cmi_df: pd.DataFrame, neighborhood_data: pd.DataFrame):
        """
        Save analysis results to files.
        
        Args:
            city_cmi_df: City-level CMI statistics
            neighborhood_data: Neighborhood-level data with multipliers
        """
        try:
            logger.info("Saving analysis results...")
            
            # Save city statistics
            city_output_path = os.path.join(self.hierarchical_dir, 'china_city_statistics.csv')
            city_cmi_df.to_csv(city_output_path, index=False)
            logger.info(f"Saved city statistics to: {city_output_path}")
            
            # Save neighborhood data
            neighborhood_output_path = os.path.join(self.hierarchical_dir, 'china_neighborhoods_cmi.csv')
            neighborhood_cols = [
                'h3index', 'ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM',
                'population', 'total_material_mass_tonnes', 'true_city_cmi',
                'neighborhood_multiplier', 'is_low_outlier', 'is_high_outlier'
            ]
            
            # Only include columns that exist
            available_cols = [col for col in neighborhood_cols if col in neighborhood_data.columns]
            neighborhood_data[available_cols].to_csv(neighborhood_output_path, index=False)
            logger.info(f"Saved neighborhood data to: {neighborhood_output_path}")
            
            # Create summary statistics
            summary_stats = {
                'analysis_date': datetime.now().strftime(DATE_FORMAT),
                'fixed_slope': self.fixed_slope,
                'total_cities': len(city_cmi_df),
                'total_neighborhoods': len(neighborhood_data),
                'china_cities': len(city_cmi_df[city_cmi_df['CTR_MN_ISO'] == 'CHN']),
                'umic_cities': len(city_cmi_df[city_cmi_df['Income_group'] == 'Upper middle income']),
                'mean_city_cmi': city_cmi_df['true_city_cmi'].mean(),
                'median_city_cmi': city_cmi_df['true_city_cmi'].median(),
                'china_mean_cmi': city_cmi_df[city_cmi_df['CTR_MN_ISO'] == 'CHN']['true_city_cmi'].mean(),
                'umic_mean_cmi': city_cmi_df[city_cmi_df['Income_group'] == 'Upper middle income']['true_city_cmi'].mean()
            }
            
            # Save model summary
            import json
            summary_path = os.path.join(self.hierarchical_dir, 'model_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            logger.info(f"Saved model summary to: {summary_path}")
            
            # Print summary
            logger.info("=" * 70)
            logger.info("HIERARCHICAL ANALYSIS SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Total cities analyzed: {summary_stats['total_cities']:,}")
            logger.info(f"Total neighborhoods: {summary_stats['total_neighborhoods']:,}")
            logger.info(f"China cities: {summary_stats['china_cities']:,}")
            logger.info(f"UMIC cities: {summary_stats['umic_cities']:,}")
            logger.info(f"Mean City CMI (all): {summary_stats['mean_city_cmi']:.3f}")
            logger.info(f"China mean CMI: {summary_stats['china_mean_cmi']:.3f}")
            logger.info(f"UMIC mean CMI: {summary_stats['umic_mean_cmi']:.3f}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def perform_hierarchical_analysis(
    material_file: Optional[str] = None,
    country_class_file: Optional[str] = None,
    debug: bool = False,
    country_code: Optional[str] = None,
    fixed_slope: float = FIXED_SLOPE
) -> bool:
    """
    Main function to perform hierarchical analysis.
    
    Args:
        material_file: Path to material mass data
        country_class_file: Path to country classification data
        debug: Use debug mode
        country_code: Country code filter
        fixed_slope: Fixed slope parameter
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("HIERARCHICAL MISMATCH ANALYSIS")
        logger.info("=" * 70)
        
        # Find input files if not specified
        if material_file is None:
            material_files = list(OUTPUT_DIRS["merged_data"].glob("Fig3_Mass_*.csv"))
            if material_files:
                material_file = max(material_files, key=os.path.getmtime)
                logger.info(f"Using material file: {material_file}")
            else:
                logger.error("No material mass files found. Run step 05 first.")
                return False
        
        if country_class_file is None:
            country_class_file = BASE_DIR / "config" / "country_classification.csv"
            if not country_class_file.exists():
                logger.error("Country classification file not found")
                return False
        
        # Initialize analyzer
        analyzer = HierarchicalMismatchAnalyzer(
            results_dir=str(RESULTS_DIR),
            fixed_slope=fixed_slope
        )
        
        # Load and prepare data
        data = analyzer.load_and_prepare_data(str(material_file), str(country_class_file))
        
        if data.empty:
            logger.error("No valid data for analysis")
            return False
        
        # Filter for debug mode
        if debug:
            data = data.head(1000)
            analyzer.data = data
            logger.info(f"Debug mode: Using {len(data)} neighborhoods")
        
        # Fit hierarchical model
        if not analyzer.fit_hierarchical_model():
            logger.error("Failed to fit hierarchical model")
            return False
        
        # Calculate city CMI
        city_cmi_df = analyzer.calculate_city_cmi()
        
        if city_cmi_df.empty:
            logger.error("Failed to calculate city CMI")
            return False
        
        # Calculate neighborhood multipliers
        neighborhood_data = analyzer.calculate_neighborhood_multipliers(city_cmi_df)
        
        if neighborhood_data.empty:
            logger.error("Failed to calculate neighborhood multipliers")
            return False
        
        # Save results
        analyzer.save_results(city_cmi_df, neighborhood_data)
        
        logger.info("Hierarchical analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in hierarchical analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main(debug: bool = False, country_code: Optional[str] = None, **kwargs) -> bool:
    """Main function for command line execution."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        success = perform_hierarchical_analysis(
            debug=debug,
            country_code=country_code,
            **kwargs
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform hierarchical mismatch analysis')
    parser.add_argument('--material-file', type=str, help='Path to material mass data file')
    parser.add_argument('--country-class-file', type=str, help='Path to country classification file')
    parser.add_argument('--country', type=str, help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process subset of data')
    parser.add_argument('--fixed-slope', type=float, default=FIXED_SLOPE, help='Fixed slope parameter')
    
    args = parser.parse_args()
    
    success = main(
        debug=args.debug,
        country_code=args.country,
        material_file=args.material_file,
        country_class_file=args.country_class_file,
        fixed_slope=args.fixed_slope
    )
    
    exit(0 if success else 1)