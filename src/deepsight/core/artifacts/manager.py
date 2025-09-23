from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import shutil
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml
import traceback

from .repository import ArtifactRepository
from .services import ChecksumService
from .datamodel import (
    ArtifactRecord,
    ArtifactStatus,
    ArtifactPath,
    DeepchecksArtifacts,
    TrainingArtifacts,
    DatasetArtifacts,
)
from ..config import DeepchecksConfig
#from ...integrations import MLflowManager


class ArtifactsManager:
    def __init__(
        self,
        sqlite_path: str,
        mlflow_manager#: MLflowManager,
    ) -> None:
        from ...integrations import MLflowManager
        self.repo = ArtifactRepository(sqlite_path)
        self.checksum = ChecksumService()
        self.mlflow:MLflowManager = mlflow_manager

    def register_artifact(
        self,
        run_id: str,
        artifact_key: Union[str, ArtifactPath],
        local_path: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        record = ArtifactRecord(
            run_id=run_id,
            mlflow_run_id=self.mlflow.run_id,
            artifact_key=artifact_key.value,
            source_uri=source_uri or self.mlflow.tracking_uri,
            local_path=local_path,
            status=ArtifactStatus.REGISTERED,
            metadata_json=metadata,
            tags_json=tags,
        )
        return self.repo.upsert(record)

    def ensure_downloaded(self, run_id: str, artifact_key: str) -> Path:
        local_path = self.mlflow.get_local_path(artifact_key, download_if_missing=False)

        rec = self.repo.get(run_id, artifact_key)
        if rec and rec.local_path and Path(rec.local_path).exists():
            self.repo.touch_access(run_id, artifact_key)
            return Path(rec.local_path)

        downloaded_dir = self.mlflow.get_local_path(
            artifact_key, download_if_missing=True
        )
        candidate = Path(downloaded_dir)
        final_path = candidate if candidate.is_file() else local_path
        if candidate.is_dir():
            final_path = candidate

        checksum = None
        size_bytes = None
        if final_path.is_file():
            checksum = self.checksum.compute_sha256(str(final_path))
            size_bytes = final_path.stat().st_size

        # update or create record
        rec = self.repo.update_local_path(
            run_id=run_id,
            artifact_key=artifact_key,
            local_path=str(final_path),
            status=ArtifactStatus.DOWNLOADED,
        )
        if rec is None:
            self.register_artifact(
                run_id=run_id,
                artifact_key=artifact_key,
                local_path=str(final_path),
                source_uri=self.mlflow.tracking_uri,
            )

        # update checksum/size if record exists
        existing = self.repo.get(run_id, artifact_key)
        if existing is not None:
            existing.checksum_sha256 = checksum
            existing.size_bytes = size_bytes
            existing.updated_at = datetime.now()
            self.repo.upsert(existing)

        return final_path

    def get_local_path(
        self,
        run_id: str,
        artifact_key: Union[str, ArtifactPath],
        download_if_missing: bool = True,
    ) -> Optional[Path]:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        artifact_key = artifact_key.value
        rec = self.repo.get(run_id, artifact_key)
        if rec and rec.local_path and Path(rec.local_path).exists():
            self.repo.touch_access(run_id, artifact_key)
            return Path(rec.local_path)
        if download_if_missing:
            return self.ensure_downloaded(run_id, artifact_key)
        return None

    def load_artifact(
        self,
        run_id: str,
        artifact_key: Union[str, ArtifactPath],
        download_if_missing: bool = True,
    ) -> Union[DeepchecksArtifacts, TrainingArtifacts, str]:
        artifact_key = (
            ArtifactPath(artifact_key)
            if isinstance(artifact_key, str)
            else artifact_key
        )
        path = self.get_local_path(run_id, artifact_key.value, download_if_missing)
        if artifact_key == ArtifactPath.DEEPCHECKS:
            return self._load_deepchecks_artifacts(path)
        elif artifact_key == ArtifactPath.TRAINING:
            return self._load_training_artifacts(path)
        elif artifact_key == ArtifactPath.MODEL_CHECKPOINT:
            return self._load_model_checkpoint(path)
        elif artifact_key == ArtifactPath.DATASET:
            return self._load_dataset_artifacts(path)
        else:
            raise ValueError(f"Artifact key {artifact_key} not supported")

    def _load_training_artifacts(self, local_path: str) -> TrainingArtifacts:
        metrics = os.path.join(local_path, ArtifactPath.TRAINING_METRICS.value)
        params = os.path.join(local_path, ArtifactPath.TRAINING_PARAMS.value)
        if not os.path.exists(params):
            return self.mlflow.get_training_artifacts()
        with open(params, "r") as f:
            params = yaml.safe_load(f)
        return TrainingArtifacts(
            metrics_path=metrics,
            metrics_values=pd.read_csv(metrics),
            params=params,
        )

    def _load_deepchecks_artifacts(self, local_path: str) -> DeepchecksArtifacts:
        artifacts = os.path.join(local_path, ArtifactPath.DEEPCHECKS_ARTIFACTS.value)
        artifacts = DeepchecksArtifacts.from_file(artifacts)
        if artifacts.config is None:
            config = os.path.join(local_path, ArtifactPath.DEEPCHECKS_CONFIG.value)
            config = DeepchecksConfig.from_file(config)
            artifacts.config = config
        return artifacts

    def _load_model_checkpoint(self, local_path: str) -> str:
        best_checkpoint = os.path.join(local_path, ArtifactPath.MODEL_CHECKPOINT.value)
        artifacts = list(Path(best_checkpoint).iterdir())
        assert len(artifacts) == 1, (
            "There should be only one artifact in the best checkpoint"
        )
        assert artifacts[0].is_file(), (
            "The artifact should be a file, but got a directory."
        )
        return str(artifacts[0])
    
    def _load_dataset_artifacts(self, local_path: str) -> DatasetArtifacts:
        artifacts = os.path.join(local_path, ArtifactPath.DATASET_METADATA.value)
        artifacts = DatasetArtifacts.from_file(artifacts)
        return artifacts

    def list_artifacts(
        self,
        run_id: str,
        prefix: Optional[str] = None,
        status: Optional[ArtifactStatus] = None,
    ) -> List[ArtifactRecord]:
        return self.repo.list_by_run(run_id, prefix=prefix, status=status)

    def remove_local_copy(self, run_id: str, artifact_key: str) -> None:
        rec = self.repo.get(run_id, artifact_key)
        if not rec or not rec.local_path:
            return
        p = Path(rec.local_path)
        if p.exists():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)
        self.repo.update_local_path(
            run_id, artifact_key, None, ArtifactStatus.REGISTERED
        )
