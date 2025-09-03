"""
Main orchestrator class for DeepSight Advisor.

This module provides the DeepSightAdvisor class that coordinates the complete
ML analysis pipeline from artifact loading to intelligent query execution.
"""

import time
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from ...core import ArtifactsManager
from ...integrations.mlflow import MLflowManager
from ...core.artifacts.datamodel import ArtifactPaths, DeepchecksArtifact, TrainingArtifacts
from ...core.query import QueryGenerator
from ...core.query.intelligence import IntelligenceClient, Providers
from ...utils.logging import get_logger

from .config import AdvisorConfig, load_config
from .result import AdvisorResult
from .errors import (
    AdvisorError, ConfigurationError, ArtifactError, 
    QueryError, OutputError, MLflowError, IntelligenceError
)


class DeepSightAdvisor:
    """
    Global orchestrator for DeepSight ML analysis pipeline.
    
    This class coordinates the complete workflow:
    1. Artifact loading and management
    2. Query generation from artifacts
    3. Intelligent analysis execution
    4. Result formatting and output
    """
    
    def __init__(self, config: Union[AdvisorConfig, str, Path, Dict[str, Any]]):
        """
        Initialize the advisor with configuration.
        
        Args:
            config: Configuration object, file path, or dictionary
        """
        self.logger = get_logger(__name__)
        
        # Load and validate configuration
        try:
            self.config = load_config(config)
            self.config.validate()
            self.logger.info("Configuration loaded and validated successfully")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("DeepSight Advisor initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Initialize MLflow manager
            self.mlflow_manager = MLflowManager(
                tracking_uri=self.config.mlflow.tracking_uri,
                run_id=self.config.mlflow.run_id,
                dwnd_dir=self.config.mlflow.download_dir
            )
            self.logger.info(f"MLflow manager initialized: {self.config.mlflow.tracking_uri}")
            
            # Initialize artifact manager
            self.artifact_manager = ArtifactsManager(
                sqlite_path=self.config.artifacts.sqlite_path,
                mlflow_manager=self.mlflow_manager
            )
            self.logger.info("Artifact manager initialized")
            
            # Initialize query generator
            self.query_generator = QueryGenerator()
            self.logger.info("Query generator initialized")
            
            # Initialize intelligence client
            self.intelligence_client = IntelligenceClient()
            self.logger.info("Intelligence client initialized")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize components: {e}")
    
    def run_analysis(self, run_id: Optional[str] = None) -> AdvisorResult:
        """
        Run complete analysis pipeline for given run_id.
        
        Args:
            run_id: Optional run ID to override config
            
        Returns:
            AdvisorResult containing all analysis information
        """
        start_time = time.time()
        
        # Use provided run_id or config run_id
        analysis_run_id = run_id or self.config.mlflow.run_id
        if not analysis_run_id:
            raise ConfigurationError("No run_id provided in config or parameter")
        
        # Initialize result object
        result = AdvisorResult(run_id=analysis_run_id)
        
        try:
            self.logger.info(f"Starting analysis for run_id: {analysis_run_id}")
            
            # Step 1: Load artifacts
            self.logger.info("Step 1: Loading artifacts...")
            artifacts = self.load_artifacts(analysis_run_id, result)
            
            # Step 2: Generate query
            self.logger.info("Step 2: Generating query...")
            prompt = self.generate_query(artifacts, result)
            
            # Step 3: Execute query (if auto_execute is enabled)
            if self.config.intelligence.auto_execute:
                self.logger.info("Step 3: Executing intelligence query...")
                self.execute_query(prompt, result)
            else:
                self.logger.info("Step 3: Skipping query execution (auto_execute=False)")
            
            # Step 4: Save results
            self.logger.info("Step 4: Saving results...")
            self.save_results(result)
            
            # Calculate total execution time
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"Analysis completed successfully in {result.execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.set_error(str(e))
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def load_artifacts(self, run_id: str, result: AdvisorResult) -> List[Union[DeepchecksArtifact, TrainingArtifacts]]:
        """
        Load and return artifacts for the given run_id.
        
        Args:
            run_id: MLflow run ID
            result: Result object to update with loading information
            
        Returns:
            List of loaded artifacts
        """
        start_time = time.time()
        artifacts = []
        
        try:
            for artifact_key in self.config.artifacts.artifact_keys:
                try:
                    self.logger.info(f"Loading artifact: {artifact_key.value}")
                    
                    artifact = self.artifact_manager.load_artifact(
                        run_id=run_id,
                        artifact_key=artifact_key,
                        download_if_missing=self.config.artifacts.download_if_missing
                    )
                    
                    artifacts.append(artifact)
                    result.add_artifact_loaded(artifact_key)
                    self.logger.info(f"Successfully loaded artifact: {artifact_key.value}")
                    
                except Exception as e:
                    error_msg = f"Failed to load artifact {artifact_key.value}: {e}"
                    self.logger.error(error_msg)
                    result.add_artifact_failed(artifact_key, str(e))
            
            if not artifacts:
                raise ArtifactError(
                    "No artifacts could be loaded",
                    run_id=run_id,
                    details={'requested_artifacts': [k.value for k in self.config.artifacts.artifact_keys]}
                )
            
            result.artifact_loading_time = time.time() - start_time
            self.logger.info(f"Artifact loading completed in {result.artifact_loading_time:.2f} seconds")
            return artifacts
            
        except Exception as e:
            result.artifact_loading_time = time.time() - start_time
            raise ArtifactError(f"Artifact loading failed: {e}", run_id=run_id)
    
    def generate_query(self, artifacts: List[Union[DeepchecksArtifact, TrainingArtifacts]], result: AdvisorResult) -> str:
        """
        Generate intelligent query from artifacts.
        
        Args:
            artifacts: List of loaded artifacts
            result: Result object to update with query information
            
        Returns:
            Generated prompt string
        """
        start_time = time.time()
        
        try:
            # Prepare context
            context = self.config.query.context or {}
            if self.config.query.custom_instructions:
                context['custom_instructions'] = self.config.query.custom_instructions
            
            # Generate prompt
            prompt = self.query_generator.build_prompt(artifacts, context)
            
            # Update result
            result.set_prompt(prompt)
            result.query_generation_time = time.time() - start_time
            
            self.logger.info(f"Query generated successfully in {result.query_generation_time:.2f} seconds")
            self.logger.info(f"Prompt length: {result.prompt_length} characters")
            
            return prompt
            
        except Exception as e:
            result.query_generation_time = time.time() - start_time
            raise QueryError(f"Query generation failed: {e}")
    
    def execute_query(self, prompt: str, result: AdvisorResult) -> None:
        """
        Execute query using configured intelligence provider.
        
        Args:
            prompt: Generated prompt to execute
            result: Result object to update with response information
        """
        start_time = time.time()
        
        try:
            # Prepare execution context
            context = self.config.intelligence.context or {}
            context.update({
                'run_id': result.run_id,
                'artifacts_loaded': result.artifacts_loaded,
                'prompt_length': result.prompt_length
            })
            
            # Execute query
            response = self.intelligence_client.execute_query(
                prompt=prompt,
                provider_type=self.config.intelligence.provider_type,
                provider_name=self.config.intelligence.provider_name,
                context=context
            )
            
            # Update result
            result.set_response(response)
            result.intelligence_execution_time = time.time() - start_time
            
            self.logger.info(f"Query executed successfully in {result.intelligence_execution_time:.2f} seconds")
            self.logger.info(f"Response length: {result.response_length} characters")
            
        except Exception as e:
            result.intelligence_execution_time = time.time() - start_time
            raise IntelligenceError(f"Query execution failed: {e}")
    
    def save_results(self, result: AdvisorResult) -> None:
        """
        Save analysis results to configured output directory.
        
        Args:
            result: Result object to save
        """
        try:
            output_dir = Path(self.config.output.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp-based filename prefix
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            base_filename = f"advisor_result_{result.run_id}_{timestamp_str}"
            
            # Save prompt if requested
            if self.config.output.save_prompt and result.prompt_generated:
                prompt_path = output_dir / f"{base_filename}_prompt.txt"
                with open(prompt_path, 'w') as f:
                    f.write(result.prompt_generated)
                result.add_output_path('prompt', prompt_path)
                self.logger.info(f"Prompt saved to: {prompt_path}")
            
            # Save response if requested
            if self.config.output.save_response and result.response_content:
                response_path = output_dir / f"{base_filename}_response.txt"
                with open(response_path, 'w') as f:
                    f.write(result.response_content)
                result.add_output_path('response', response_path)
                self.logger.info(f"Response saved to: {response_path}")
            
            # Save result in configured format
            if self.config.output.format == 'json':
                result_path = output_dir / f"{base_filename}.json"
                result.to_json(result_path, include_content=True)
            elif self.config.output.format == 'yaml':
                result_path = output_dir / f"{base_filename}.yaml"
                result.to_yaml(result_path, include_content=True)
            elif self.config.output.format == 'txt':
                result_path = output_dir / f"{base_filename}.txt"
                result.to_text(result_path)
            
            self.logger.info(f"Results saved to: {result_path}")
            
        except Exception as e:
            raise OutputError(f"Failed to save results: {e}")
    
    def get_summary(self, result: AdvisorResult) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.
        
        Args:
            result: Analysis result object
            
        Returns:
            Summary dictionary
        """
        return result.get_summary()
    
    def validate_configuration(self) -> None:
        """Validate the current configuration."""
        try:
            self.config.validate()
            self.logger.info("Configuration validation passed")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        try:
            # Create new config with updates
            current_dict = self.config.dict()
            current_dict.update(updates)
            
            # Reload configuration
            self.config = AdvisorConfig(**current_dict)
            self.config.validate()
            
            # Reinitialize components if needed
            self._initialize_components()
            
            self.logger.info("Configuration updated successfully")
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration: {e}")


# Convenience function for quick analysis
def run_analysis(
    run_id: str,
    tracking_uri: str = "http://localhost:5000",
    config_overrides: Optional[Dict[str, Any]] = None
) -> AdvisorResult:
    """
    Quick function to run analysis with minimal configuration.
    
    Args:
        run_id: MLflow run ID to analyze
        tracking_uri: MLflow tracking server URI
        config_overrides: Optional configuration overrides
        
    Returns:
        AdvisorResult containing analysis information
    """
    # Create basic config
    config = {
        "mlflow": {
            "tracking_uri": tracking_uri,
            "run_id": run_id
        }
    }
    
    # Apply overrides if provided
    if config_overrides:
        config.update(config_overrides)
    
    # Create advisor and run analysis
    advisor = DeepSightAdvisor(config)
    return advisor.run_analysis()
