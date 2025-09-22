from typing import Optional

from .base import Step
from ...utils.logging import get_logger
from ..artifacts import ArtifactsManager,ArtifactPaths
from ...integrations import MLflowManager

LOGGER = get_logger(__name__)

class LoadArtifact(Step):
    
    def __init__(self,artifact_key:ArtifactPaths,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        self.mlflow_manager = mlflow_manager
        self.sqlite_path = artifact_sqlite_path
        self.artifact_key=artifact_key

    def run(self,context:dict)->dict:
        mlflow_mgr = self.mlflow_manager or context.get('mlflow_manager')
        sqlite_path = self.sqlite_path or context.get('sqlite_path')
        artifact_mgr = ArtifactsManager(
            sqlite_path=sqlite_path,
            mlflow_manager=mlflow_mgr
        )
        assert mlflow_mgr.run_id is not None, "MLflow run_id must be set in MLflowManager"
        LOGGER.info(f"Loading artifact: {self.artifact_key} for run_id: {mlflow_mgr.run_id}")
        artifact = artifact_mgr.load_artifact(
            run_id=mlflow_mgr.run_id,
            artifact_key=self.artifact_key,
            download_if_missing=True
        )
        if "artifacts" in context.keys():
            context['artifacts'].append(artifact)
        else:
            context['artifacts'] = [artifact]
        return context

class LoadTrainingArtifact(LoadArtifact):
    
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        super().__init__(artifact_key=ArtifactPaths.TRAINING,
                        mlflow_manager=mlflow_manager,
                        artifact_sqlite_path=artifact_sqlite_path
                    )

class LoadDeepchecksArtifacts(LoadArtifact):
    
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        super().__init__(artifact_key=ArtifactPaths.DEEPCHECKS,
                        mlflow_manager=mlflow_manager,
                        artifact_sqlite_path=artifact_sqlite_path
                    )           

class LoadModelCheckpoint(Step):

    def run(self,context:dict,mlflow_manager:MLflowManager)->dict:
        mlflow_mgr = mlflow_manager or context.get('mlflow_manager')
        path_to_ckpt = mlflow_mgr.get_model_checkpoint()
        context['model_path'] = path_to_ckpt
        return context
