"""
DeepSight Query Generation Example

This example demonstrates how to generate intelligent prompts from artifacts
using DeepSight's query generation system.

Features demonstrated:
- Artifact loading and management
- Query generation from multiple artifacts
- Integration with AI providers
- Prompt building for analysis and insights

Usage:
    python examples/build_prompts.py

Requirements:
    - MLflow server running on localhost:5000
    - Valid run_id with Deepchecks and Training artifacts
    - Optional: AI provider credentials for query execution
"""

import logging
from pathlib import Path

# DeepSight imports
from deepsight.core.query import QueryGenerator
from deepsight.core import ArtifactsManager
from deepsight.integrations.mlflow import MLflowManager
from deepsight.core.artifacts import ArtifactPaths
from deepsight.core.query.intelligence import IntelligenceClient, ProviderType, Providers
from deepsight.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO')
logger = get_logger(__name__)

def main():
    """
    Main function demonstrating query generation from artifacts.
    
    This function:
    1. Connects to MLflow and loads artifacts
    2. Generates intelligent prompts from artifact data
    3. Optionally executes queries using AI providers
    4. Displays generated prompts and responses
    """
    try:
        logger.info("Starting DeepSight query generation example...")
        
        # Initialize MLflow manager
        mlflow_mgr = MLflowManager(
            tracking_uri="http://localhost:5000", 
            run_id="07c04cc42fd9461e98f7eb0bf42444fb",  # Replace with your run_id
            dwnd_dir="tmp"
        )
        
        logger.info(f"Connected to MLflow at: {mlflow_mgr.tracking_uri}")
        logger.info(f"Using run_id: {mlflow_mgr.run_id}")

        # Initialize artifact manager
        artifacts = ArtifactsManager(
            sqlite_path="tmp/artifacts.db",
            mlflow_manager=mlflow_mgr,
        )
        
        logger.info("Initialized artifact manager")

        # Load Deepchecks artifacts
        logger.info("Loading Deepchecks artifacts...")
        deepchecks_artifact = artifacts.load_artifact(
            run_id=mlflow_mgr.run_id, 
            artifact_key=ArtifactPaths.DEEPCHECKS,
            download_if_missing=True
        )
        
        # Load Training artifacts
        logger.info("Loading Training artifacts...")
        training_artifact = artifacts.load_artifact(
            run_id=mlflow_mgr.run_id, 
            artifact_key=ArtifactPaths.TRAINING,
            download_if_missing=True
        )
        
        logger.info("Artifacts loaded successfully!")

        # Generate query from artifacts
        logger.info("Generating query from artifacts...")
        query_generator = QueryGenerator()
        prompt = query_generator.build_prompt([deepchecks_artifact, training_artifact])
        
        # Display generated prompt
        print("\n" + "="*80)
        print("GENERATED PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80)
        
        # Optional: Execute query using AI provider
        # Uncomment the following lines to execute the query
        # logger.info("Executing query with AI provider...")
        # intel = IntelligenceClient()
        # response = intel.execute_query(prompt=prompt, provider_name=Providers.CURSOR)
        # 
        # print("\n" + "="*80)
        # print("AI RESPONSE:")
        # print("="*80)
        # print(response.content)
        # print("="*80)
        
        logger.info("Query generation example completed successfully!")
        
    except Exception as e:
        logger.error(f"Query generation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()