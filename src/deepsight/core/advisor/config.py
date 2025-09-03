"""
Configuration management for DeepSight Advisor.

This module provides comprehensive configuration classes for the advisor,
including YAML loading, validation, and default value management.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator

from ...core.artifacts.datamodel import ArtifactPaths
from ...core.query.intelligence.models import ProviderType, Providers


class MLflowConfig(BaseModel):
    """Configuration for MLflow integration."""
    
    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    run_id: str = Field(
        description="MLflow run ID to analyze"
    )
    download_dir: str = Field(
        default="mlflow_downloads",
        description="Local directory for downloading artifacts"
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="MLflow experiment name (optional)"
    )
    
    @field_validator('tracking_uri')
    def validate_tracking_uri(cls, v):
        if not v.startswith(('http://', 'https://', 'file://')):
            raise ValueError("tracking_uri must start with http://, https://, or file://")
        return v
    
    @field_validator('run_id')
    def validate_run_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("run_id cannot be empty")
        return v.strip()


class ArtifactConfig(BaseModel):
    """Configuration for artifact management."""
    
    artifact_keys: List[ArtifactPaths] = Field(
        default=[ArtifactPaths.TRAINING, ArtifactPaths.DEEPCHECKS],
        description="List of artifact types to load"
    )
    download_if_missing: bool = Field(
        default=True,
        description="Whether to download artifacts if not locally cached"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable local caching"
    )
    sqlite_path: str = Field(
        default="tmp/artifacts.db",
        description="Path to SQLite database for artifact caching"
    )
    
    @field_validator('artifact_keys')
    def validate_artifact_keys(cls, v):
        if not v:
            raise ValueError("artifact_keys cannot be empty")
        return v


class QueryConfig(BaseModel):
    """Configuration for query generation."""
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for query generation"
    )
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Custom instructions to append to generated prompts"
    )
    prompt_builders: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for custom prompt builders"
    )


class IntelligenceConfig(BaseModel):
    """Configuration for intelligence client."""
    
    provider_type: ProviderType = Field(
        default=ProviderType.CODING_AGENT,
        description="Type of intelligence provider to use"
    )
    provider_name: Optional[Providers] = Field(
        default=None,
        description="Specific provider name (auto-selected if None)"
    )
    auto_execute: bool = Field(
        default=True,
        description="Whether to automatically execute queries"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for query execution"
    )
    timeout: Optional[int] = Field(
        default=300,
        description="Timeout in seconds for query execution"
    )


class OutputConfig(BaseModel):
    """Configuration for output management."""
    
    save_prompt: bool = Field(
        default=True,
        description="Whether to save generated prompts"
    )
    save_response: bool = Field(
        default=True,
        description="Whether to save AI responses"
    )
    output_dir: str = Field(
        default="advisor_output",
        description="Directory to save outputs"
    )
    format: str = Field(
        default="txt",
        description="Output format (txt, json, yaml)"
    )
    verbose: bool = Field(
        default=True,
        description="Whether to enable verbose logging"
    )
    
    @field_validator('format')
    def validate_format(cls, v):
        allowed_formats = ['txt','json','yaml']
        if v.lower() not in allowed_formats:
            raise ValueError(f"format must be one of {allowed_formats}")
        return v.lower()


class AdvisorConfig(BaseModel):
    """Main configuration class for DeepSight Advisor."""
    
    mlflow: MLflowConfig = Field(
        description="MLflow configuration"
    )
    artifacts: ArtifactConfig = Field(
        default_factory=ArtifactConfig,
        description="Artifact management configuration"
    )
    query: QueryConfig = Field(
        default_factory=QueryConfig,
        description="Query generation configuration"
    )
    intelligence: IntelligenceConfig = Field(
        default_factory=IntelligenceConfig,
        description="Intelligence client configuration"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration"
    )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "AdvisorConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AdvisorConfig":
        """Create configuration from dictionary."""
        try:
            return cls(**config_dict)
        except Exception as e:
            raise ValueError(f"Error creating configuration from dict: {e}")
    
    def to_file(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
    
    def merge(self, other: "AdvisorConfig") -> "AdvisorConfig":
        """Merge another configuration into this one."""
        # Convert to dict, merge, and create new instance
        self_dict = self.model_dump()
        other_dict = other.model_dump()
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge(self_dict, other_dict)
        
        return self.__class__(**merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate(self) -> None:
        """Validate the complete configuration."""
        # Create output directory if it doesn't exist
        output_dir = Path(self.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        


def create_default_config(run_id: str, tracking_uri: str = "http://localhost:5000") -> AdvisorConfig:
    """Create a default configuration with minimal required parameters."""
    return AdvisorConfig(
        mlflow=MLflowConfig(
            tracking_uri=tracking_uri,
            run_id=run_id
        )
    )


def load_config(config_source: Union[str, Path, Dict[str, Any], AdvisorConfig]) -> AdvisorConfig:
    """Load configuration from various sources."""
    if isinstance(config_source, AdvisorConfig):
        return config_source
    elif isinstance(config_source, dict):
        return AdvisorConfig.from_dict(config_source)
    elif isinstance(config_source, (str, Path)):
        return AdvisorConfig.from_file(config_source)
    else:
        raise ValueError(f"Unsupported config source type: {type(config_source)}")
