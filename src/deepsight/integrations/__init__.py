"""
External tool integrations for DeepSight MLOps Copilot.

This module provides integrations with popular MLOps tools:
- MLflow for model registry and experiment tracking
- Deepchecks for automated model validation
- DVC for data versioning and access
- Research assistant for academic paper search
"""

from .mlflow_client import MLflowClient
from .deepchecks_runner import DeepchecksRunner
from .dvc_manager import DVCManager
from .research_assistant import ResearchAssistant

__all__ = [
    "MLflowClient",
    "DeepchecksRunner",
    "DVCManager", 
    "ResearchAssistant"
]
