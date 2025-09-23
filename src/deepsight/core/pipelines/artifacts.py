from typing import Optional, List
from omegaconf import OmegaConf
import traceback
import tempfile
import os
from torch.utils.data import Dataset
from .base import Step
from ...utils.logging import get_logger
from ..artifacts import ArtifactsManager,ArtifactPath,DeepchecksArtifacts
from ...integrations import MLflowManager

LOGGER = get_logger(__name__)

# Loading artifacts
class LoadArtifact(Step):
    
    def __init__(self,artifact_key:ArtifactPath,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        self.mlflow_manager = mlflow_manager
        self.sqlite_path = artifact_sqlite_path
        self.artifact_key=artifact_key

    def run(self,context:dict,**kwargs)->dict:
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
        if self.artifact_key == ArtifactPath.MODEL_CHECKPOINT:
            context['model_path'] = artifact

        if "artifacts" in context.keys():
            context['artifacts'].append(artifact)
        else:
            context['artifacts'] = [artifact]       
    
        return context

class LoadTrainingArtifact(LoadArtifact):
    
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        super().__init__(artifact_key=ArtifactPath.TRAINING,
                        mlflow_manager=mlflow_manager,
                        artifact_sqlite_path=artifact_sqlite_path
                    )

class LoadDeepchecksArtifacts(LoadArtifact):
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        super().__init__(artifact_key=ArtifactPath.DEEPCHECKS,
                        mlflow_manager=mlflow_manager,
                        artifact_sqlite_path=artifact_sqlite_path
                    )           

class LoadModelCheckpoint(Step):
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,artifact_sqlite_path:Optional[str]=None):
        super().__init__(artifact_key=ArtifactPath.MODEL_CHECKPOINT,
                        mlflow_manager=mlflow_manager,
                        artifact_sqlite_path=artifact_sqlite_path
                    )

# Logging artifacts
class LogArtifact(Step):
    def __init__(self,artifact_key:ArtifactPath,mlflow_manager:Optional[MLflowManager]=None):
        self.artifact_key=artifact_key
        self.mlflow_manager = mlflow_manager

class LogTrainingArtifact(LogArtifact):
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None):
        super().__init__(artifact_key=ArtifactPath.TRAINING,mlflow_manager=mlflow_manager)
    def run(self,context:dict,metric_names:List[str]=None,**kwargs)->dict:
        mlflow_mgr = self.mlflow_manager or context.get('mlflow_manager')    
        metric_names = metric_names or context.get('metric_names',None)    
        if metric_names is None:
            LOGGER.warning("metric_names not provided, will not log metric histories")
            return context
        mlflow_mgr.get_run_metric_histories(metric_names=metric_names,log_as_artifact=True)
        mlflow_mgr.get_run_parameters(log_as_artifact=True)
        return context
    
class LogChecksArtifacts(LogArtifact):
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,):
        super().__init__(artifact_key=ArtifactPath.DEEPCHECKS,mlflow_manager=mlflow_manager
                    )
    def run(self,context:dict,
            checks_artifacts:Optional[DeepchecksArtifacts]=None,**kwargs
        )->dict:
        mlflow_mgr = self.mlflow_manager or context.get('mlflow_manager')
        checks_artifacts = checks_artifacts or context.get('checks_artifacts')

        assert checks_artifacts is not None, "checks_artifacts must be provided in context or as an argument"
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_path = os.path.join(
                tmp_dir, ArtifactPath.DEEPCHECKS_ARTIFACTS.value
            )
            OmegaConf.save(checks_artifacts.to_dict(), artifacts_path)
            try:
                mlflow_mgr.add_artifact(
                    artifact_key=ArtifactPath.DEEPCHECKS.value,
                    local_path=artifacts_path,
                )
            except Exception:
                LOGGER.error(f"Error logging deepchecks: {traceback.format_exc()}")
        return context
    
class LogModelCheckpoint(LogArtifact):
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None):
        super().__init__(artifact_key=ArtifactPath.MODEL_CHECKPOINT,mlflow_manager=mlflow_manager
                    )
    def run(self,context:dict,checkpoint_artifact_path:Optional[str]=None,**kwargs)->dict:
        mlflow_mgr = self.mlflow_manager or context.get('mlflow_manager')
        local_path = checkpoint_artifact_path or context.get('checkpoint_artifact_path')
        if local_path is None:
            LOGGER.warning("checkpoint_artifact_path not provided, will not log model checkpoint")
            return context
        mlflow_mgr.add_artifact(
                    artifact_key=self.artifact_key,
                    local_path=local_path,
                )
        context["checkpoint_artifact_path"] = checkpoint_artifact_path
        return context

class LogDatasetMetadata(LogArtifact):
    def __init__(self,mlflow_manager:Optional[MLflowManager]=None,dataset_name:Optional[str]=None):
        super().__init__(artifact_key=ArtifactPath.DATASET_METADATA,mlflow_manager=mlflow_manager
                    )
        self.dataset_name = dataset_name
    def run(self,
            context:dict,
            train_data:Optional[Dataset]=None,
            test_data:Optional[Dataset]=None,
            **kwargs
    ) -> dict:
        train_data = train_data or context.get("train_data")
        test_data = test_data or context.get("test_data")
        dataset_name = self.dataset_name or context.get("dataset_name")
        
        assert dataset_name is not None, "dataset_name must be provided in context or as an argument"

        self.mlflow_manager.log_dataset(dataset_name=dataset_name,train_data=train_data, test_data=test_data)
        context["dataset_name"] = dataset_name
        return context