"""
Configuration management for DeepSight MLOps Copilot.

This module handles loading, saving, and validating configuration files
in various formats (YAML, JSON) with schema validation and defaults.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path

ROOT = Path(__file__).parents[2]


class DeepchecksConfig(BaseModel):
    train_test_validation: bool = Field(default=True,description="Whether to run the train_test_validation suite")
    data_integrity: bool = Field(default=True,description="Whether to run the data_integrity suite")
    model_evaluation: bool = Field(default=False,description="Whether to run the model_evaluation suite")
    max_samples: Optional[int] = Field(default=None,description="Maximum number of samples to run the suites on")
    random_state: int = Field(default=42,description="Random seed to use for the suites")
    save_results: bool = Field(default=False,description="Whether to save the results")
    output_dir: Optional[str] = Field(default=None,description="Output directory to save the results")
    save_display: bool = Field(default=False,description="Whether to save the display")
    parse_results: bool = Field(default=False,description="Whether to parse the results")
    batch_size: int = Field(default=16,description="Batch size to use for the suites")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DeepchecksConfig":
        return cls(**config)

class DVCConfig(BaseModel):
    remote: str = Field(default="origin")
    data_path: str = Field(default="data/")
    repo_path: Optional[str] = None