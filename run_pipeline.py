#!/usr/bin/env python3
"""
Main execution script for China City Mass Index Map generation.

This script orchestrates the complete pipeline from raw data extraction 
to final map generation.

Usage:
    python run_pipeline.py --help
    python run_pipeline.py --full-pipeline
    python run_pipeline.py --step hierarchical_analysis
    python run_pipeline.py --country CHN --debug

Author: Generated for China City Mass Index Map
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from config.settings import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChinaCMIPipeline:
    """Main pipeline orchestrator for China City Mass Index Map generation."""
    
    def __init__(self, country_code: Optional[str] = None, debug: bool = False):
        """
        Initialize pipeline.
        
        Args:
            country_code (str, optional): ISO 3-letter country code
            debug (bool): Enable debug mode for testing
        """
        self.country_code = country_code
        self.debug = debug
        self.steps_completed = []
        
        # Ensure directories exist
        ensure_directories()
        
        logger.info("China City Mass Index Map Pipeline initialized")
        if country_code:
            logger.info(f"Filtering to country: {country_code}")
        if debug:
            logger.info("Debug mode enabled")
    
    def run_step(self, step_name: str) -> bool:
        """
        Run a specific pipeline step.
        
        Args:
            step_name (str): Name of the step to run
            
        Returns:
            bool: True if successful
        """
        step_methods = {
            'h3_neighborhoods': self._run_h3_neighborhoods,
            'building_extraction': self._run_building_extraction,
            'road_extraction': self._run_road_extraction,
            'data_merging': self._run_data_merging,
            'material_calculation': self._run_material_calculation,
            'city_boundaries': self._run_city_boundaries,
            'hierarchical_analysis': self._run_hierarchical_analysis,
            'map_generation': self._run_map_generation
        }
        
        if step_name not in step_methods:
            logger.error(f"Unknown step: {step_name}")
            logger.info(f"Available steps: {list(step_methods.keys())}")
            return False
        
        logger.info(f"Running step: {step_name}")
        try:
            success = step_methods[step_name]()
            if success:
                self.steps_completed.append(step_name)
                logger.info(f"Step {step_name} completed successfully")
            else:
                logger.error(f"Step {step_name} failed")
            return success
        except Exception as e:
            logger.error(f"Error in step {step_name}: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            bool: True if all steps successful
        """
        pipeline_steps = [
            'h3_neighborhoods',
            'building_extraction', 
            'road_extraction',
            'data_merging',
            'material_calculation',
            'city_boundaries',
            'hierarchical_analysis',
            'map_generation'
        ]
        
        logger.info("Starting full pipeline execution")
        logger.info(f"Pipeline steps: {' → '.join(pipeline_steps)}")
        
        start_time = datetime.now()
        
        for step in pipeline_steps:
            if not self.run_step(step):
                logger.error(f"Pipeline failed at step: {step}")
                return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Steps completed: {len(self.steps_completed)}")
        
        return True
    
    def _run_h3_neighborhoods(self) -> bool:
        """Run H3 neighborhood creation step."""
        try:
            from scripts import create_h3_neighborhoods
            return create_h3_neighborhoods.create_h3_neighborhoods(
                country_code=self.country_code,
                debug=self.debug
            )
        except ImportError:
            logger.warning("H3 neighborhoods script not available, skipping...")
            return True
    
    def _run_building_extraction(self) -> bool:
        """Run building data extraction step."""
        logger.warning("Building extraction step not yet implemented")
        return True
    
    def _run_road_extraction(self) -> bool:
        """Run road data extraction step.""" 
        logger.warning("Road extraction step not yet implemented")
        return True
    
    def _run_data_merging(self) -> bool:
        """Run data merging step."""
        logger.warning("Data merging step not yet implemented")
        return True
    
    def _run_material_calculation(self) -> bool:
        """Run material mass calculation step."""
        logger.warning("Material calculation step not yet implemented") 
        return True
    
    def _run_city_boundaries(self) -> bool:
        """Run city boundaries generation step."""
        logger.warning("City boundaries step not yet implemented")
        return True
    
    def _run_hierarchical_analysis(self) -> bool:
        """Run hierarchical mismatch analysis step."""
        logger.warning("Hierarchical analysis step not yet implemented")
        return True
    
    def _run_map_generation(self) -> bool:
        """Run final map generation step.""" 
        logger.warning("Map generation step not yet implemented")
        return True

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description='China City Mass Index Map Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --full-pipeline              # Run complete pipeline
  python run_pipeline.py --step hierarchical_analysis # Run single step  
  python run_pipeline.py --country CHN --debug        # Debug mode for China
  python run_pipeline.py --list-steps                 # Show available steps
        """
    )
    
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run the complete pipeline')
    parser.add_argument('--step', type=str,
                       help='Run a specific pipeline step')
    parser.add_argument('--country', type=str,
                       help='ISO 3-letter country code (e.g., CHN)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (uses Bangladesh for testing)')
    parser.add_argument('--list-steps', action='store_true',
                       help='List available pipeline steps')
    
    args = parser.parse_args()
    
    # Handle list steps
    if args.list_steps:
        steps = [
            'h3_neighborhoods', 'building_extraction', 'road_extraction',
            'data_merging', 'material_calculation', 'city_boundaries', 
            'hierarchical_analysis', 'map_generation'
        ]
        print("Available pipeline steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        return True
    
    # Validate arguments
    if not (args.full_pipeline or args.step):
        parser.error("Must specify either --full-pipeline or --step")
    
    if args.country and len(args.country) != 3:
        parser.error("Country code must be 3 letters (e.g., CHN)")
    
    # Initialize pipeline
    pipeline = ChinaCMIPipeline(
        country_code=args.country,
        debug=args.debug
    )
    
    # Run pipeline
    if args.full_pipeline:
        success = pipeline.run_full_pipeline()
    else:
        success = pipeline.run_step(args.step)
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        return True
    else:
        logger.error("Pipeline execution failed!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)