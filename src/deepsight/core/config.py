"""
Configuration management for DeepSight Advisor.

This module provides comprehensive configuration classes for the advisor,
including YAML loading, validation, and default value management.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from .artifacts.datamodel import ArtifactPaths


class MLflowConfig(BaseModel):
    """Configuration for MLflow integration."""

    tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking server URI"
    )
    run_id: str = Field(description="MLflow run ID to analyze")
    download_dir: str = Field(
        default="mlflow_downloads",
        description="Local directory for downloading artifacts",
    )
    experiment_name: Optional[str] = Field(
        default=None, description="MLflow experiment name (optional)"
    )

    @field_validator("tracking_uri")
    def validate_tracking_uri(cls, v):
        if not v.startswith(("http://", "https://", "file://","sqlite://")):
            raise ValueError(
                "tracking_uri must start with http://, https://, or file://"
            )
        return v

    @field_validator("run_id")
    def validate_run_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("run_id cannot be empty")
        return v.strip()


class ArtifactConfig(BaseModel):
    """Configuration for artifact management."""

    artifact_keys: List[ArtifactPaths] = Field(
        default=[ArtifactPaths.TRAINING, ArtifactPaths.DEEPCHECKS],
        description="List of artifact types to load",
    )
    download_if_missing: bool = Field(
        default=True, description="Whether to download artifacts if not locally cached"
    )
    cache_enabled: bool = Field(
        default=True, description="Whether to enable local caching"
    )
    sqlite_path: str = Field(
        default="tmp/artifacts.db",
        description="Path to SQLite database for artifact caching",
    )

    @field_validator("artifact_keys")
    def validate_artifact_keys(cls, v):
        if not v:
            raise ValueError("artifact_keys cannot be empty")
        return v


class QueryConfig(BaseModel):
    """Configuration for query generation."""

    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context for query generation"
    )
    custom_instructions: Optional[str] = Field(
        default=None, description="Custom instructions to append to generated prompts"
    )
    prompt_builders: Optional[Dict[str, Any]] = Field(
        default=None, description="Configuration for custom prompt builders"
    )


class OutputConfig(BaseModel):
    """Configuration for output management."""

    save_prompt: bool = Field(
        default=True, description="Whether to save generated prompts"
    )
    save_response: bool = Field(
        default=True, description="Whether to save AI responses"
    )
    output_dir: str = Field(
        default="advisor_output", description="Directory to save outputs"
    )
    format: str = Field(default="txt", description="Output format (txt, json, yaml)")
    verbose: bool = Field(default=True, description="Whether to enable verbose logging")

    @field_validator("format")
    def validate_format(cls, v):
        allowed_formats = ["txt", "json", "yaml"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"format must be one of {allowed_formats}")
        return v.lower()
