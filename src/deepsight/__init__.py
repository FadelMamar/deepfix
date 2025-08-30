"""
DeepSight - MLOps Copilot for Computer Vision Overfitting Analysis

A Python library that provides systematic guidance for resolving overfitting issues 
in computer vision models through automated analysis, testing, and research assistance.
"""

__version__ = "0.1.0"
__author__ = "DeepSight Team"
__email__ = "team@deepsight.ai"

# Core imports for main API
from .core.analyzer import ModelAnalyzer
from .core.detector import OverfittingDetector
from .core.reporter import ReportGenerator

# Main entry point class
class CopilotAnalyzer:
    """Main entry point for the DeepSight MLOps Copilot."""
    
    def __init__(self, config_path: str = None):
        """Initialize the copilot analyzer with configuration."""
        pass
    
    def analyze_model(self, model_uri: str):
        """Analyze a model for overfitting issues."""
        pass

__all__ = [
    "CopilotAnalyzer",
    "ModelAnalyzer", 
    "OverfittingDetector",
    "ReportGenerator"
]
