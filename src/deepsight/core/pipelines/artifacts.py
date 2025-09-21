from typing import Optional

from .base import Step
from ..artifacts import ArtifactsManager,ArtifactPaths
from ...integrations import MLflowManager


class LoadTrainingArtifact(Step):
    
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        self.mlflow_manager = mlflow_manager
        self.sqlite_path = artifact_sqlite_path        

    def run(self,context:dict)->dict:
        mlflow_mgr = self.mlflow_manager or context.get('mlflow_manager')
        sqlite_path = self.sqlite_path or context.get('sqlite_path')
        artifact_mgr = ArtifactsManager(
            sqlite_path=sqlite_path,
            mlflow_manager=mlflow_mgr
        )
        artifact = artifact_mgr.load_artifact(
            run_id=mlflow_mgr.run_id,
            artifact_key=ArtifactPaths.TRAINING,
            download_if_missing=True
        )
        context['training_artifact'] = artifact
        return context


class LoadDeepchecksArtifacts(Step):
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        self.mlflow_manager = mlflow_manager
        self.sqlite_path = artifact_sqlite_path

    def run(self,context:dict)->dict:
        mlflow_mgr = self.mlflow_manager or context.get('mlflow_manager')
        sqlite_path = self.sqlite_path or context.get('sqlite_path')
        artifact_mgr = ArtifactsManager(
            sqlite_path=sqlite_path,
            mlflow_manager=mlflow_mgr
        )
        artifact = artifact_mgr.load_artifact(
            run_id=mlflow_mgr.run_id,
            artifact_key=ArtifactPaths.DEEPCHECKS,
            download_if_missing=True
        )
        context['deepchecks_artifact'] = artifact
        return context


class LoadModelCheckpoint(Step):

    def run(self,context:dict,mlflow_manager:MLflowManager)->dict:
        mlflow_mgr = mlflow_manager or context.get('mlflow_manager')
        path_to_ckpt = mlflow_mgr.get_model_checkpoint()
        context['model_path'] = path_to_ckpt
        return context
