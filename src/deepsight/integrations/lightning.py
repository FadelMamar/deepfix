from lightning.pytorch.callbacks import Callback
import lightning as L
import mlflow
from pathlib import Path
from omegaconf import OmegaConf
import traceback
from torch.utils.data import Dataset
from deepchecks.vision import VisionData
from typing import Optional, Tuple, Callable
import torch
import tempfile
import os

from ..utils.logging import get_logger
from .deepchecks import DeepchecksRunner
from ..utils.config import DeepchecksConfig
from ..core.data import ClassificationVisionDataLoader
from ..core.artifacts.datamodel import ArtifactPaths

LOGGER = get_logger(__name__)


class DeepSightCallback(Callback):
    def __init__(
        self,
        dataset_name: str,
        deepchecks_config: DeepchecksConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.mlflow_run_id = None
        self.mlflow_experiment_id = None
        self.best_model_path: str = None
        self.best_model_score: float = None
        self.config: dict = config or {}
        self.dataset_name: str = dataset_name
        self.deepchecks_config: DeepchecksConfig = deepchecks_config

        self.train_dataset: Dataset = train_dataset
        self.val_dataset: Optional[Dataset] = val_dataset

    def setup(self, trainer, pl_module, stage):
        LOGGER.info(f"Setup callback for {stage} stage")

    @property
    def state(self):
        return dict(
            mlflow_run_id=self.mlflow_run_id,
            mlflow_experiment_id=self.mlflow_experiment_id,
            best_model_path=self.best_model_path,
            best_model_score=self.best_model_score,
        )

    def load_state_dict(self, state_dict):
        self.mlflow_run_id = state_dict.get("mlflow_run_id", None)
        self.mlflow_experiment_id = state_dict.get("mlflow_experiment_id", None)
        self.best_model_path = state_dict.get("best_model_path", None)
        self.best_model_score = state_dict.get("best_model_score", None)

    def state_dict(self):
        return self.state.copy()

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Get the run_id and experiment_id from the logger
        if self.mlflow_run_id is None:
            self.mlflow_run_id = getattr(pl_module.logger, "run_id", None)
            self.mlflow_experiment_id = getattr(pl_module.logger, "experiment_id", None)
            if self.mlflow_run_id is not None:
                LOGGER.info(f"MLflow run_id: {self.mlflow_run_id}")
                LOGGER.info(f"MLflow experiment_id: {self.mlflow_experiment_id}")
            else:
                LOGGER.warning("No mlflow logger found")
                self.mlflow_run_id = "None"

    def run_deepchecks(self, pl_module: L.LightningModule) -> None:
        deepchecks_runner = DeepchecksRunner(self.deepchecks_config)
        self.train_dataset, self.val_dataset = self.load_vision_data(
            train_data=self.train_dataset,
            val_data=self.val_dataset,
            batch_size=self.deepchecks_config.batch_size,
            model=pl_module.predict_step,
        )
        artifacts = deepchecks_runner.run_suites(
            train_data=self.train_dataset,
            test_data=self.val_dataset,
            dataset_name=self.dataset_name,
        )
        if self.mlflow_run_id:
            with mlflow.start_run(run_id=self.mlflow_run_id, log_system_metrics=True):
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        artifacts.config = self.deepchecks_config
                        artifacts_path = os.path.join(
                            tmp_dir, ArtifactPaths.DEEPCHECKS_ARTIFACTS.value
                        )
                        OmegaConf.save(artifacts.to_dict(), artifacts_path)
                        mlflow.log_artifact(
                            artifacts_path, ArtifactPaths.DEEPCHECKS.value
                        )
                except Exception:
                    LOGGER.error(f"Error logging deepchecks: {traceback.format_exc()}")
        return None

    def load_vision_data(
        self,
        train_data: Dataset,
        batch_size: int,
        val_data: Optional[Dataset] = None,
        model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[VisionData, VisionData]:
        train_data = ClassificationVisionDataLoader.load_from_dataset(
            train_data, batch_size=batch_size, model=model
        )
        if val_data is not None:
            val_data = ClassificationVisionDataLoader.load_from_dataset(
                val_data, batch_size=batch_size, model=model
            )
        return train_data, val_data

    def log_model(self, trainer: L.Trainer) -> None:
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.best_model_score = trainer.checkpoint_callback.best_model_score
        if self.mlflow_run_id and self.best_model_path:
            with mlflow.start_run(run_id=self.mlflow_run_id, log_system_metrics=True):
                try:
                    mlflow.log_metric("best_model_score", self.best_model_score)
                    mlflow.log_artifact(
                        str(self.best_model_path), ArtifactPaths.MODEL_CHECKPOINT.value
                    )
                except Exception:
                    LOGGER.error(
                        f"Error logging model checkpoint: {traceback.format_exc()}"
                    )
        else:
            LOGGER.warning("No mlflow run_id or best_model_path found")

    # TODO: make sure that on_fit_end pl_module is the best model, automatically loaded by trainer
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.log_model(trainer)
        self.run_deepchecks(pl_module)
