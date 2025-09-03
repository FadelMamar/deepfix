"""
DeepSight Advisor - Global Orchestrator

This module provides the DeepSight Advisor, a global orchestrator that automates
the complete ML analysis pipeline from artifact loading to intelligent query
generation and execution.

The advisor serves as a unified interface for running comprehensive ML analysis
workflows with minimal configuration.
"""

from .config import (
    AdvisorConfig,
    MLflowConfig,
    ArtifactConfig,
    QueryConfig,
    IntelligenceConfig,
    OutputConfig
)
from .errors import (
    AdvisorError,
    ConfigurationError,
    ArtifactError,
    QueryError,
    OutputError
)
from .result import AdvisorResult
from .orchestrator import DeepSightAdvisor,run_analysis

__all__ = [
    # Configuration
    "AdvisorConfig",
    "MLflowConfig", 
    "ArtifactConfig",
    "QueryConfig",
    "IntelligenceConfig",
    "OutputConfig",
    
    # Errors
    "AdvisorError",
    "ConfigurationError",
    "ArtifactError", 
    "QueryError",
    "OutputError",
    
    # Core Classes
    "AdvisorResult",
    "DeepSightAdvisor",
    "run_analysis",
]
