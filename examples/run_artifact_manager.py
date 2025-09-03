"""
DeepSight Artifact Management Example

This example demonstrates how to manage and retrieve artifacts from MLflow
using DeepSight's artifact management system.

Features demonstrated:
- MLflow integration for artifact tracking
- Artifact registration and retrieval
- Local caching and download management
- Artifact metadata and content access

Usage:
    python examples/run_artifact_manager.py

Requirements:
    - MLflow server running on localhost:5000
    - Valid run_id with artifacts
    - SQLite database for local caching
"""

import logging
from pathlib import Path

# DeepSight imports
from deepsight.core import ArtifactsManager
from deepsight.integrations.mlflow import MLflowManager
from deepsight.core.artifacts import ArtifactPaths
from deepsight.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO')
logger = get_logger(__name__)

def main():
    """
    Main function demonstrating artifact management capabilities.
    
    This function:
    1. Connects to MLflow tracking server
    2. Initializes artifact manager with local caching
    3. Demonstrates artifact registration and retrieval
    4. Shows how to access artifact content and metadata
    """
    try:
        logger.info("Starting DeepSight artifact management example...")
        
        # Initialize MLflow manager
        mlflow_mgr = MLflowManager(
            tracking_uri="http://localhost:5000", 
            run_id="07c04cc42fd9461e98f7eb0bf42444fb",  # Replace with your run_id
            dwnd_dir="tmp"  # Local download directory
        )
        
        logger.info(f"Connected to MLflow at: {mlflow_mgr.tracking_uri}")
        logger.info(f"Using run_id: {mlflow_mgr.run_id}")
        
        # Initialize artifact manager with local SQLite cache
        artifacts = ArtifactsManager(
            sqlite_path="tmp/artifacts.db",
            mlflow_manager=mlflow_mgr,
        )
        
        logger.info("Initialized artifact manager with local caching")

        # Example 1: Register and download artifacts
        #artifacts.register_artifact(
        #     run_id=mlflow_mgr.run_id, 
        #     artifact_key=ArtifactPaths.TRAINING  # Replace with actual artifact key
        #)
        #local_path = artifacts.get_local_path(
        #     run_id=mlflow_mgr.run_id, 
        #     artifact_key=ArtifactPaths.TRAINING,  # Replace with actual artifact key
        #     download_if_missing=True
        #)
        # logger.info(f"Artifact downloaded to: {local_path}")

        # Example 2: Load training artifacts
        logger.info("Loading training artifacts...")
        artifact = artifacts.load_artifact(
            run_id=mlflow_mgr.run_id, 
            artifact_key=ArtifactPaths.TRAINING.value,
            download_if_missing=True
        )
        
        # Display artifact information
        artifact_dict = artifact.to_dict()
        logger.info("Training artifact loaded successfully!")
        logger.info(f"Artifact type: {type(artifact).__name__}")
        logger.info(f"Artifact keys: {list(artifact_dict.keys()) if isinstance(artifact_dict, dict) else 'N/A'}")
        
        # Print artifact content (be careful with large artifacts)
        print("\n" + "="*50)
        print("ARTIFACT CONTENT:")
        print("="*50)
        print(artifact_dict)
        
        logger.info("Artifact management example completed successfully!")
        
    except Exception as e:
        logger.error(f"Artifact management failed with error: {e}")
        raise

if __name__ == "__main__":
    main()