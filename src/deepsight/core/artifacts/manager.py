from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from .repository import ArtifactRepository
from .services import LocalPathResolver, ChecksumService
from .datamodel import ArtifactRecord, ArtifactStatus
from ...integrations.mlflow import MLflowManager


class ArtifactsManager:
    def __init__(
        self,
        sqlite_path: str,
        artifacts_root: str,
        mlflow_manager: MLflowManager,
    ) -> None:
        self.repo = ArtifactRepository(sqlite_path)
        self.path_resolver = LocalPathResolver(artifacts_root)
        self.checksum = ChecksumService()
        self.mlflow = mlflow_manager

    def register_artifact(
        self,
        run_id: str,
        artifact_key: str,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            run_id=run_id,
            mlflow_run_id=self.mlflow.run_id,
            artifact_key=artifact_key,
            source_uri=source_uri,
            status=ArtifactStatus.REGISTERED,
            metadata_json=metadata,
            tags_json=tags,
        )
        return self.repo.upsert(record)

    def ensure_downloaded(self, run_id: str, artifact_key: str) -> Path:
        local_path = self.path_resolver.resolve(run_id, artifact_key)
        
        rec = self.repo.get(run_id, artifact_key)
        if rec and rec.local_path and Path(rec.local_path).exists():
            self.repo.touch_access(run_id, artifact_key)
            return Path(rec.local_path)

        downloaded_dir = self.mlflow.client.download_artifacts(
            run_id, artifact_key, dst_path=str(local_path.parent)
        )
        # MLflow may download into a directory; if artifact_key points to file keep path
        candidate = Path(downloaded_dir)
        final_path = candidate if candidate.is_file() else local_path
        if candidate.is_dir():
            # If a directory, set final path to the directory path
            final_path = candidate

        checksum = None
        size_bytes = None
        if final_path.is_file():
            checksum = self.checksum.compute_sha256(str(final_path))
            size_bytes = final_path.stat().st_size

        self.repo.update_local_path(
            run_id=run_id,
            artifact_key=artifact_key,
            local_path=str(final_path),
            status=ArtifactStatus.DOWNLOADED,
        )

        # update checksum/size if record exists
        existing = self.repo.get(run_id, artifact_key)
        if existing is not None:
            existing.checksum_sha256 = checksum
            existing.size_bytes = size_bytes
            existing.updated_at = datetime.now()
            self.repo.upsert(existing)

        return final_path

    def get_local_path(self, run_id: str, artifact_key: str, download_if_missing: bool = True) -> Optional[Path]:
        rec = self.repo.get(run_id, artifact_key)
        if rec and rec.local_path and Path(rec.local_path).exists():
            self.repo.touch_access(run_id, artifact_key)
            return Path(rec.local_path)
        if download_if_missing:
            return self.ensure_downloaded(run_id, artifact_key)
        return None

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
        self.repo.update_local_path(run_id, artifact_key, None, ArtifactStatus.REGISTERED)

