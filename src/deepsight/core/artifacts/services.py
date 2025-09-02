from __future__ import annotations

import hashlib
from pathlib import Path


class LocalPathResolver:
    def __init__(self, artifacts_root: str):
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def resolve(self, run_id: str, artifact_key: str) -> Path:
        safe_key = artifact_key.replace("/", "_").replace("\\", "_")
        path = self.artifacts_root / run_id / safe_key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class ChecksumService:
    def compute_sha256(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


