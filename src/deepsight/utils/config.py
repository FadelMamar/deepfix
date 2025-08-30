"""
Configuration management for DeepSight MLOps Copilot.

This module handles loading, saving, and validating configuration files
in various formats (YAML, JSON) with schema validation and defaults.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
import json
import os
from copy import deepcopy


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration dictionary.
    
    Returns:
        Default configuration with all supported settings
    """
    return {
        'mlflow': {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            'experiment_name': 'cv_model_analysis',
            'registry_uri': None
        },
        'dvc': {
            'remote': 'origin',
            'data_path': 'data/',
            'repo_path': None
        },
        'deepchecks': {
            'suite': 'computer_vision',
            'custom_checks': [
                'overfitting_analysis',
                'data_drift_detection'
            ],
            'timeout': 300
        },
        'research': {
            'sources': ['arxiv', 'semantic_scholar'],
            'max_papers': 10,
            'keywords': ['overfitting', 'computer vision', 'regularization'],
            'cache_dir': Path.home() / '.deepsight' / 'research_cache'
        },
        'detection': {
            'thresholds': {
                'train_val_gap': 0.1,
                'learning_curve_slope': 0.05,
                'stability_threshold': 0.02,
                'performance_drop': 0.05
            },
            'metrics': ['accuracy', 'loss', 'f1_score', 'precision', 'recall'],
            'cross_validation': {
                'folds': 5,
                'stratified': True
            }
        },
        'output': {
            'format': ['html', 'pdf'],
            'template': 'default',
            'include_visualizations': True,
            'output_dir': Path.cwd() / 'deepsight_output'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': None
        }
    }


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Merge with defaults
        default_config = get_default_config()
        merged_config = merge_configs(default_config, config or {})
        
        return merged_config
        
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid configuration file format: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries recursively.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_config_value(config: Dict[str, Any], key_path: str, 
                    default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'mlflow.tracking_uri')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated key path (e.g., 'mlflow.tracking_uri')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def validate_config_structure(config: Dict[str, Any]) -> bool:
    """
    Validate the basic structure of a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if structure is valid
        
    Raises:
        ValueError: If configuration structure is invalid
    """
    required_sections = ['mlflow', 'detection', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section missing: {section}")
    
    # Validate detection thresholds
    thresholds = get_config_value(config, 'detection.thresholds', {})
    required_thresholds = ['train_val_gap', 'learning_curve_slope']
    
    for threshold in required_thresholds:
        if threshold not in thresholds:
            raise ValueError(f"Required threshold missing: detection.thresholds.{threshold}")
    
    return True
