from lightning.pytorch.callbacks import Callback
import lightning as L
from omegaconf import OmegaConf
import traceback
from torch.utils.data import Dataset
from typing import Optional
import tempfile
import os

from ..utils.logging import get_logger
from ..core.config import DeepchecksConfig
from .mlflow import MLflowManager
from ..core.artifacts.datamodel import ArtifactPaths,DeepchecksArtifacts
from ..core.pipelines.data_ingestion import DataIngestor
from ..core.pipelines.checks import Checks
from ..core.pipelines.base import Pipeline

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
        self.mlflow_manager: MLflowManager = None
        self.checks_runner = Pipeline([DataIngestor(),Checks()])


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
            tracking_uri = getattr(pl_module.logger, "_tracking_uri", None)
            if self.mlflow_run_id is not None:
                LOGGER.info(f"MLflow run_id: {self.mlflow_run_id}")
                LOGGER.info(f"MLflow experiment_id: {self.mlflow_experiment_id}")
                self.mlflow_manager = MLflowManager(run_id=self.mlflow_run_id,tracking_uri=tracking_uri)
            else:
                LOGGER.warning("No mlflow logger found")
                self.mlflow_run_id = None

    def run(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # log best model
        self._log_model(trainer)

        # run deepchecks
        self.checks_runner.run(
            train_data=self.train_dataset,
            test_data=self.val_dataset,
            dataset_name=self.dataset_name,
            deepchecks_config=self.deepchecks_config,
            model=pl_module.predict_step,
            batch_size=self.deepchecks_config.batch_size
        )
        artifacts = self.checks_runner.context.get("deepchecks_artifacts", None)
        if artifacts is None:
            LOGGER.warning("No deepchecks artifacts found")
            return None
        self._log_deepchecks(artifacts)
            
        return None
    
    def _log_deepchecks(self, artifacts:DeepchecksArtifacts) -> None:
        if self.mlflow_manager is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifacts.config = self.deepchecks_config
                artifacts_path = os.path.join(
                    tmp_dir, ArtifactPaths.DEEPCHECKS_ARTIFACTS.value
                )
                OmegaConf.save(artifacts.to_dict(), artifacts_path)
                try:
                    self.mlflow_manager.add_artifact(
                        artifact_key=ArtifactPaths.DEEPCHECKS.value,
                        local_path=artifacts_path,
                    )
                except Exception:
                    LOGGER.error(f"Error logging deepchecks: {traceback.format_exc()}")
        else:
            LOGGER.warning("No mlflow manager found")

    def _log_model(self, trainer: L.Trainer) -> None:
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.best_model_score = trainer.checkpoint_callback.best_model_score
        if self.mlflow_run_id and self.best_model_path:
            try:
                self.mlflow_manager.add_artifact(
                    artifact_key=ArtifactPaths.MODEL_CHECKPOINT.value,
                    local_path=str(self.best_model_path),
                )
            except Exception:
                LOGGER.error(
                    f"Error logging model checkpoint: {traceback.format_exc()}"
                )

        else:
            LOGGER.warning("No mlflow run_id or best_model_path found")

    # TODO: make sure that on_fit_end pl_module is the best model, automatically loaded by trainer
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.run(trainer=trainer,pl_module=pl_module)
