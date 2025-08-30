"""
Deepchecks integration for automated model validation.

This module provides comprehensive Deepchecks integration including:
- Pre-configured computer vision test suites
- Custom overfitting detection checks
- Batch processing workflows
- Result parsing and analysis
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
from deepchecks.vision.suites import train_test_validation
from deepchecks.vision import VisionData
from deepchecks.core.suite import SuiteResult

class DeepchecksRunner:
    """
    Deepchecks integration for automated model validation and testing.
    
    Provides high-level interface for running Deepchecks suites and custom
    checks specifically designed for overfitting detection in CV models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Deepchecks runner with configuration.
        """
        self.config = config or {}
        self.suite_train_test_validation = train_test_validation()
        
    
    def run_suite_train_test_validation(self, train_data: VisionData, test_data: VisionData,max_num_samples: Optional[int] = None):
        return self.suite_train_test_validation.run(train_dataset=train_data, 
        test_dataset=test_data,
        max_num_samples=max_num_samples)
