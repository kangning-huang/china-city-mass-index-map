#!/usr/bin/env python3
"""
Main pipeline runner for the China City Mass Index Map generation.

This script orchestrates the complete pipeline from H3 grid generation
to final interactive map creation.

Usage:
    python run_pipeline.py --full-pipeline
    python run_pipeline.py --step hierarchical_analysis
    python run_pipeline.py --list-steps

Author: Generated for NYU China Grant project
Date: 2025
"""

import sys
import os
import argparse
import logging
import importlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config.settings import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / 'pipeline.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Main pipeline runner class."""
    
    def __init__(self):
        """Initialize the pipeline runner."""
        self.pipeline_steps = {
            'h3_neighborhoods': {
                'script': '01_create_h3_neighborhoods',
                'description': 'Create H3 hexagonal neighborhoods for all cities',
                'inputs': ['Google Earth Engine GHSL Urban Centres'],
                'outputs': ['H3 grid geometries with city information']
            },
            'building_extraction': {
                'script': '02_extract_building_data', 
                'description': 'Extract building volume data from Google Earth Engine',
                'inputs': ['H3 grids', 'Zhou2022 building dataset'],
                'outputs': ['Building volume per neighborhood']
            },
            'road_extraction': {
                'script': '03_extract_road_data',
                'description': 'Extract road network data from Google Earth Engine',
                'inputs': ['H3 grids', 'OpenStreetMap road data'],
                'outputs': ['Road areas and lengths per neighborhood']
            },
            'data_merging': {
                'script': '04_merge_infrastructure_data',
                'description': 'Combine building and infrastructure data',
                'inputs': ['Building data', 'Road data', 'Population data'],
                'outputs': ['Merged neighborhood dataset']
            },
            'material_calculation': {
                'script': '05_calculate_material_masses',
                'description': 'Calculate material masses using climate-adjusted intensities',
                'inputs': ['Merged infrastructure data', 'Material intensity database'],
                'outputs': ['Neighborhood-level material mass estimates']
            },
            'city_boundaries': {
                'script': '06_generate_city_boundaries',
                'description': 'Generate city boundary geometries from H3 grids',
                'inputs': ['H3 grids with data'],
                'outputs': ['City boundary shapefiles']
            },
            'hierarchical_analysis': {
                'script': '07_hierarchical_mismatch_analysis',
                'description': 'Perform hierarchical mixed-effects analysis',
                'inputs': ['Global neighborhood data', 'Country classifications'],
                'outputs': ['City and neighborhood CMI statistics']
            },
            'map_generation': {
                'script': '08_create_interactive_map',
                'description': 'Generate dual-layer interactive map',
                'inputs': ['City boundaries', 'City statistics', 'Neighborhood statistics'],
                'outputs': ['Interactive HTML map']
            }
        }
        
        # Ensure directories exist
        ensure_directories()
        
        logger.info("Pipeline runner initialized")
    
    def list_steps(self):
        """List all available pipeline steps."""
        print("\nAvailable Pipeline Steps:")
        print("=" * 60)
        
        for step_id, step_info in self.pipeline_steps.items():
            print(f"\n{step_id}:")
            print(f"  Description: {step_info['description']}")
            print(f"  Script: {step_info['script']}.py")
            print(f"  Inputs: {', '.join(step_info['inputs'])}")
            print(f"  Outputs: {', '.join(step_info['outputs'])}")
        
        print("\nUsage:")
        print("  python run_pipeline.py --step <step_name>")
        print("  python run_pipeline.py --full-pipeline")
        print("  python run_pipeline.py --debug  # Test with Bangladesh data")
    
    def run_step(self, 
                 step_name: str, 
                 debug: bool = False,
                 country: Optional[str] = None,
                 **kwargs) -> bool:
        """
        Run a single pipeline step.
        
        Args:
            step_name: Name of the step to run
            debug: Use debug mode
            country: Country code to filter
            **kwargs: Additional arguments for the step
            
        Returns:
            True if step completed successfully
        """
        try:
            if step_name not in self.pipeline_steps:
                logger.error(f"Unknown step: {step_name}")
                logger.info(f"Available steps: {list(self.pipeline_steps.keys())}")
                return False
            
            step_info = self.pipeline_steps[step_name]
            script_name = step_info['script']
            
            logger.info("=" * 70)
            logger.info(f"RUNNING STEP: {step_name.upper()}")
            logger.info(f"Description: {step_info['description']}")
            logger.info("=" * 70)
            
            # Import and run the script
            script_path = SCRIPTS_DIR / f"{script_name}.py"
            
            if not script_path.exists():
                logger.error(f"Script not found: {script_path}")
                logger.info(f"Available scripts in {SCRIPTS_DIR}:")
                for script in SCRIPTS_DIR.glob("*.py"):
                    logger.info(f"  - {script.name}")
                return False
            
            # Dynamic import and execution
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Call main function with appropriate arguments
            if hasattr(module, 'main'):
                success = module.main(debug=debug, country_code=country, **kwargs)
                
                if success:
                    logger.info(f"Step {step_name} completed successfully")
                    return True
                else:
                    logger.error(f"Step {step_name} failed")
                    return False
            else:
                logger.error(f"Script {script_name}.py does not have a main() function")
                return False
                
        except Exception as e:
            logger.error(f"Error running step {step_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def run_full_pipeline(self, 
                         debug: bool = False, 
                         country: Optional[str] = None,
                         skip_steps: List[str] = None) -> bool:
        """
        Run the complete pipeline from start to finish.
        
        Args:
            debug: Use debug mode
            country: Country code to filter  
            skip_steps: List of step names to skip
            
        Returns:
            True if all steps completed successfully
        """
        try:
            if skip_steps is None:
                skip_steps = []
            
            logger.info("=" * 70)
            logger.info("STARTING FULL PIPELINE EXECUTION")
            logger.info("=" * 70)
            logger.info(f"Debug mode: {debug}")
            logger.info(f"Country filter: {country or 'All countries'}")
            logger.info(f"Steps to skip: {skip_steps or 'None'}")
            logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Track execution time and success for each step
            step_results = {}
            total_start_time = datetime.now()
            
            for step_name in self.pipeline_steps.keys():
                if step_name in skip_steps:
                    logger.info(f"Skipping step: {step_name}")
                    continue
                
                step_start_time = datetime.now()
                
                success = self.run_step(step_name, debug=debug, country=country)
                
                step_duration = datetime.now() - step_start_time
                step_results[step_name] = {
                    'success': success,
                    'duration': step_duration
                }
                
                if not success:
                    logger.error(f"Pipeline failed at step: {step_name}")
                    self._print_step_summary(step_results)
                    return False
                
                logger.info(f"Step {step_name} completed in {step_duration}")
            
            total_duration = datetime.now() - total_start_time
            
            # Print summary
            self._print_pipeline_summary(step_results, total_duration)
            
            return True
                
        except Exception as e:
            logger.error(f"Error in full pipeline execution: {e}")
            return False
    
    def _print_step_summary(self, step_results: Dict):
        """Print summary of step results."""
        logger.info("=" * 70)
        logger.info("STEP EXECUTION SUMMARY")
        logger.info("=" * 70)
        
        for step_name, result in step_results.items():
            status = "✓" if result['success'] else "✗"
            logger.info(f"{status} {step_name}: {result['duration']}")
    
    def _print_pipeline_summary(self, step_results: Dict, total_duration):
        """Print final pipeline summary."""
        logger.info("=" * 70)
        logger.info("PIPELINE EXECUTION COMPLETED")
        logger.info("=" * 70)
        
        successful_steps = sum(1 for r in step_results.values() if r['success'])
        total_steps = len(step_results)
        
        logger.info(f"Total steps executed: {total_steps}")
        logger.info(f"Successful steps: {successful_steps}")
        logger.info(f"Failed steps: {total_steps - successful_steps}")
        logger.info(f"Total execution time: {total_duration}")
        
        if successful_steps == total_steps:
            logger.info("🎉 Full pipeline completed successfully!")
            
            # Print final output locations
            logger.info("\nFinal outputs:")
            for dir_name, dir_path in OUTPUT_DIRS.items():
                if dir_path.exists():
                    file_count = len(list(dir_path.glob("*")))
                    logger.info(f"  - {dir_name}: {dir_path} ({file_count} files)")
            
        else:
            logger.error("Pipeline completed with errors")
            self._print_step_summary(step_results)

def validate_pipeline_setup() -> bool:
    """Validate pipeline setup and requirements."""
    try:
        logger.info("Validating pipeline setup...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ is required")
            return False
        
        logger.info(f"✓ Python version: {sys.version}")
        
        # Check required directories
        logger.info("Checking directories...")
        ensure_directories()
        logger.info("✓ Directories created")
        
        # Check Google Earth Engine setup
        try:
            from utils.gee_utils import initialize_gee
            if initialize_gee(GEE_PROJECT_ID):
                logger.info("✓ Google Earth Engine initialized")
            else:
                logger.warning("⚠ Google Earth Engine setup may have issues")
        except Exception as e:
            logger.warning(f"⚠ Google Earth Engine: {e}")
        
        # Check required Python packages
        logger.info("Checking Python packages...")
        required_packages = [
            'pandas', 'geopandas', 'folium', 'h3', 'tobler',
            'statsmodels', 'numpy', 'matplotlib', 'seaborn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Install with: pip install -r requirements.txt")
            return False
        
        logger.info("✓ All required packages found")
        
        # Check configuration files
        config_files = [
            BASE_DIR / "config" / "settings.py",
            BASE_DIR / "config" / "country_classification.csv"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                logger.info(f"✓ Found: {config_file.name}")
            else:
                logger.warning(f"⚠ Missing: {config_file}")
        
        logger.info("🎉 Pipeline setup validation completed!")
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description='China City Mass Index Map Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --list-steps
  python run_pipeline.py --full-pipeline
  python run_pipeline.py --full-pipeline --debug
  python run_pipeline.py --step hierarchical_analysis
  python run_pipeline.py --full-pipeline --country CHN
        """
    )
    
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run the complete pipeline')
    parser.add_argument('--step', type=str,
                       help='Run a specific step')
    parser.add_argument('--list-steps', action='store_true',
                       help='List all available pipeline steps')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (uses Bangladesh data for testing)')
    parser.add_argument('--country', type=str,
                       help='ISO country code to process (e.g., CHN)')
    parser.add_argument('--skip-steps', nargs='+',
                       help='Steps to skip in full pipeline')
    parser.add_argument('--validate-setup', action='store_true',
                       help='Validate pipeline setup and requirements')
    
    args = parser.parse_args()
    
    # Initialize pipeline runner
    runner = PipelineRunner()
    
    try:
        # Handle different execution modes
        if args.list_steps:
            runner.list_steps()
            return True
        
        elif args.validate_setup:
            return validate_pipeline_setup()
        
        elif args.step:
            return runner.run_step(
                args.step, 
                debug=args.debug, 
                country=args.country
            )
        
        elif args.full_pipeline:
            return runner.run_full_pipeline(
                debug=args.debug,
                country=args.country,
                skip_steps=args.skip_steps
            )
        
        else:
            parser.print_help()
            return True
            
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)