from deepsight.core import ArtifactsManager
from deepsight.integrations.mlflow import MLflowManager
from deepsight.core.artifacts import ArtifactPaths

mlflow_mgr = MLflowManager(tracking_uri="http://localhost:5000", 
                            run_id="07c04cc42fd9461e98f7eb0bf42444fb", 
                            dwnd_dir="tmp"
                        )
artifacts = ArtifactsManager(
    sqlite_path="tmp/artifacts.db",
    artifacts_root="tmp/artifacts",
    mlflow_manager=mlflow_mgr,
)

# Register and download
#artifacts.register_artifact(run_id=mlflow_mgr.run_id, artifact_key="invalid")
#local = artifacts.get_local_path(run_id=mlflow_mgr.run_id, artifact_key="invalid")  # downloads if missing
#print(local)

artifact = artifacts.load_artifact(run_id=mlflow_mgr.run_id, 
                                    artifact_key=ArtifactPaths.TRAINING.value,
                                    download_if_missing=True
                                )
print(artifact.to_dict())