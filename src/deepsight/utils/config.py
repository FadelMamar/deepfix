"""
Configuration management for DeepSight MLOps Copilot.

This module handles loading, saving, and validating configuration files
in various formats (YAML, JSON) with schema validation and defaults.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal
import yaml
import json
import os
from copy import deepcopy


class DeepchecksConfig(BaseModel):
    train_test_validation: bool = Field(default=True,description="Whether to run the train_test_validation suite")
    data_integrity: bool = Field(default=True,description="Whether to run the data_integrity suite")
    model_evaluation: bool = Field(default=False,description="Whether to run the model_evaluation suite")
    max_samples: Optional[int] = Field(default=None,description="Maximum number of samples to run the suites on")
    random_state: int = Field(default=42,description="Random seed to use for the suites")
    save_results: bool = Field(default=False,description="Whether to save the results")
    save_results_format: Literal["json", "html"] = Field(default="json",description="Format to save the results")
    output_dir: Optional[str] = Field(default=None,description="Output directory to save the results")
    save_display: bool = Field(default=False,description="Whether to save the display")

class DVCConfig(BaseModel):
    remote: str = Field(default="origin")
    data_path: str = Field(default="data/")
    repo_path: Optional[str] = None