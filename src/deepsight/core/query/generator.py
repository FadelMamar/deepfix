"""
Main QueryGenerator class for orchestrating prompt generation from existing Pydantic models.
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import traceback

from ...utils.logging import get_logger
from ...core.artifacts.datamodel import DeepchecksArtifact, TrainingArtifacts
from .builders import DeepchecksPromptBuilder, TrainingPromptBuilder
from .config import QueryGeneratorConfig
from ...utils.exceptions import PromptBuilderError


class QueryGenerator:
    """Main class for orchestrating prompt generation from existing Pydantic models."""
    
    def __init__(
        self, 
        config: Optional[Union[Dict[str, Any], QueryGeneratorConfig]] = None,
        config_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the QueryGenerator.
        
        Args:
            config: Optional configuration for the QueryGenerator
            config_path: Optional path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config, config_path)
        self.prompt_builders = self._initialize_prompt_builders()
        
        self.logger.info("QueryGenerator initialized successfully")
    
    def _load_config(
        self, 
        config: Optional[Dict[str, Any]]=None, 
        config_path: Optional[Union[str, Path]]=None
    ) -> QueryGeneratorConfig:
        """Load and validate configuration."""
        if config:
            return QueryGeneratorConfig.from_dict(config)
        if config_path:
            return QueryGeneratorConfig.from_file(config_path)
        return QueryGeneratorConfig()


    def build_prompt(
        self, 
        artifacts: List[Union[DeepchecksArtifact, TrainingArtifacts]], 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build structured prompt based on artifact types.
        """
        try:
            prompt_parts = []
            
            # Group artifacts by type
            artifact_groups = {}
            for artifact in artifacts:
                artifact_type = type(artifact).__name__
                if artifact_type not in artifact_groups:
                    artifact_groups[artifact_type] = []
                artifact_groups[artifact_type].append(artifact)
            
            # Build prompts for each artifact type
            for artifact_type, artifact_list in artifact_groups.items():
                builder = self._get_prompt_builder(artifact_type)
                if builder:
                    for artifact in artifact_list:
                        try:
                            prompt = builder.build_prompt(artifact, context)
                            prompt_parts.append(prompt)
                        except Exception:
                            raise PromptBuilderError(artifact_type, f"Failed to build prompt: {traceback.format_exc()}")
                else:
                    raise PromptBuilderError(artifact_type, "No prompt builder found for artifact type")
            
            if not prompt_parts:
                raise PromptBuilderError("all", "No prompts could be built from artifacts")
            
            # Combine all prompts
            full_prompt = "\n\n".join(prompt_parts)
            
            return full_prompt
            
        except Exception:
            raise PromptBuilderError("unknown", f"Unexpected error during prompt building: {traceback.format_exc()}")
    
    def _get_prompt_builder(self, artifact_type: str) -> Optional[Union[DeepchecksPromptBuilder, TrainingPromptBuilder]]:
        """Get appropriate prompt builder for the given artifact type."""
        for builder in self.prompt_builders:
            if builder.can_build(artifact_type):
                return builder
        return None
    
    def _initialize_prompt_builders(self) -> List[Union[DeepchecksPromptBuilder, TrainingPromptBuilder]]:
        """Initialize prompt builders based on configuration."""
        builders = []
        
        # Always include core builders
        builders.extend([
            DeepchecksPromptBuilder(),
            TrainingPromptBuilder()
        ])
        
        # Add custom builders from config if specified
        if 'custom_builders' in self.config.prompt_builders:
            for builder_config in self.config.prompt_builders['custom_builders']:
                # Implementation for custom builders would go here
                self.logger.debug(f"Custom builder config found: {builder_config}")
        
        return builders
    
