"""
MLflow integration for model registry and experiment tracking.

This module provides comprehensive MLflow integration including:
- Model registry access and management
- Experiment tracking and comparison
- Artifact management and retrieval
- Metric aggregation and analysis
"""

from typing import Dict, List, Optional, Any, Tuple
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment, FileInfo
import pandas as pd
from pathlib import Path
import os
from omegaconf import OmegaConf

from ..core.artifacts.datamodel import DeepchecksArtifact
from ..utils.config import DeepchecksConfig


class MLflowManager:
    """
    MLflow integration client for model, experiment and artifact management.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None,
                experiment_name: Optional[str] = None,
                run_id: Optional[str] = None,
                dwnd_dir: str = "."
            ):
        """
        Initialize MLflow client with tracking configuration.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name for operations
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.current_experiment: Experiment = None
        self.current_run:Run = None
        self.dwnd_dir = dwnd_dir
        if experiment_name:
            self.set_experiment(experiment_name)
        
        if run_id:
            self.set_run(run_id)
    
    def set_experiment(self, experiment_name: str) -> None:
        """
        Set the current experiment for operations.
        
        Args:
            experiment_name: Name of the experiment to use
        """
        try:
            self.current_experiment = self.client.get_experiment_by_name(experiment_name)
        except Exception:
           raise ValueError(f"Experiment {experiment_name} not found")
    
    def set_run(self, run_id: str) -> None:
        """
        Set the current run for operations.
        """
        self.current_run = self.client.get_run(run_id)
        self.run_id = run_id
    
    def get_run_info(self) -> Dict[str, Any]:
        info = dict(id=self.current_run.info.run_id,
            experiment_id=self.current_run.info.experiment_id,
            user_id=self.current_run.info.user_id,
            status=self.current_run.info.status,
            duration=(self.current_run.info.end_time - self.current_run.info.start_time)/1000/60, # in minutes
        )
        return info

    def get_run_metrics(self) -> Dict[str, Any]:
        return self.current_run.data.metrics
    
    def _get_run_metric_history(self, metric_name: str) -> pd.DataFrame:
        metric_history = self.client.get_metric_history(self.run_id, metric_name)
        metric_history = pd.DataFrame([m.to_dictionary() for m in metric_history])
        metric_history = metric_history.sort_values(by='step', ascending=True)  
        return metric_history
    
    def get_run_metric_histories(self,metric_names: List[str]) -> pd.DataFrame:
        assert isinstance(metric_names, list), "Metric names must be a list"
        assert len(metric_names) > 0, "Metric names must be a non-empty list"
        assert all(isinstance(metric_name, str) for metric_name in metric_names), "Metric names must be a list of strings"
        df = pd.concat([self._get_run_metric_history(metric_name) for metric_name in metric_names]).reset_index(drop=True)
        return df

    def get_run_tags(self) -> Dict[str, Any]:
        return self.current_run.data.tags
            
    def get_model_checkpoint(self) -> str:
        best_checkpoint = self.client.download_artifacts(self.run_id, "best_checkpoint", dst_path=self.dwnd_dir)
        artifacts = list(Path(best_checkpoint).iterdir())
        assert len(artifacts) == 1, "There should be only one artifact in the best checkpoint"
        assert artifacts[0].is_file(), "The artifact should be a file"
        return str(artifacts[0])
    
    def get_run_parameters(self) -> pd.DataFrame:
        return self.current_run.data.params

    def get_deepchecks_artifacts(self) -> Tuple[DeepchecksConfig, DeepchecksArtifact]:
        deepchecks = self.client.download_artifacts(self.run_id, "deepchecks", dst_path=self.dwnd_dir)
        artifacts = list(Path(deepchecks).iterdir())
        assert len(artifacts) == 2, "There should be two artifacts in the deepchecks"
        # load config and artifacts
        config = os.path.join(deepchecks, "config.yaml")
        config = DeepchecksConfig.from_dict(dict(OmegaConf.load(config)))
        artifacts = os.path.join(deepchecks, "artifacts.yaml")
        artifacts = dict(OmegaConf.load(artifacts))
        artifacts = DeepchecksArtifact.from_dict(artifacts)
        return config, artifacts


