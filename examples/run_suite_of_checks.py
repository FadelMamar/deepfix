"""
DeepSight Data Validation Example

This example demonstrates how to run comprehensive data validation suites
using Deepchecks integration with DeepSight.

Features demonstrated:
- Data integrity validation
- Train/test validation
- Model evaluation (optional)
- Result parsing and analysis
- Integration with CLIP models for vision tasks

Usage:
    python examples/run_suite_of_checks.py

Requirements:
    - Required datasets available
    - Deepchecks installed
    - Optional: CLIP model for advanced validation
"""

import json
import logging
from pathlib import Path

# DeepSight imports
from deepsight.zoo.datasets.foodwaste import (get_label_mapping, 
translations_de_en, load_train_and_val_datasets)
from deepsight.zoo.timm_models import CLIPModel
from deepsight.core.data import ClassificationVisionDataLoader
from deepsight.integrations import DeepchecksRunner
from deepsight.utils import DeepchecksConfig
from deepsight.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO')
logger = get_logger(__name__)

def main():
    """
    Main function to run comprehensive data validation suites.
    
    This function:
    1. Loads the dataset and label mappings
    2. Configures Deepchecks validation
    3. Optionally sets up a CLIP model for advanced validation
    4. Runs validation suites
    5. Returns parsed results
    """
    try:
        logger.info("Starting DeepSight data validation example...")
        
        # Load dataset and label mappings
        ing2name, ing2label = get_label_mapping()
        train_dataset, val_dataset = load_train_and_val_datasets(image_size=1024)
        
        # Prepare ingredient names for CLIP model (if used)
        ingredients_en = ["a " + translations_de_en[t] for t in ing2name.values()]
        
        logger.info(f"Loaded dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        logger.info(f"Number of classes: {len(ing2name)}")

        # Configure Deepchecks validation
        config = DeepchecksConfig(
            train_test_validation=True,  # Validate train/test distribution
            data_integrity=True,  # Check data quality and integrity
            model_evaluation=False,  # Skip model evaluation for data-only validation
            save_results=True,  # Save results to disk
            save_display=False,  # Skip saving visualizations
            save_results_format='json',  # Save in JSON format
            parse_results=True,  # Parse results for analysis
            output_dir='results'  # Output directory
        )

        # Initialize Deepchecks runner
        runner = DeepchecksRunner(config)
        logger.info("Initialized Deepchecks runner")

        # Optional: Initialize CLIP model for advanced validation
        model = None  # CLIPModel('PE-Core-T-16-384', ingredients_en)
        if model is not None:
            logger.info("Using CLIP model for advanced validation")
        else:
            logger.info("Running validation without model (data-only checks)")

        # Prepare data loaders for Deepchecks
        vision_train_data = ClassificationVisionDataLoader.load_from_dataset(
            train_dataset, 
            batch_size=8, 
            shuffle=True, 
            model=model
        )
        vision_test_data = ClassificationVisionDataLoader.load_from_dataset(
            val_dataset, 
            batch_size=8, 
            shuffle=True, 
            model=model
        )
        
        logger.info("Running validation suites...")
        
        # Run validation suites
        results = runner.run_suites(
            train_data=vision_train_data, 
            test_data=vision_test_data
        )
        
        logger.info("Validation completed successfully!")
        logger.info(f"Results saved to: {config.output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()