import traceback
from typing import Callable, Optional, List
import torch
from torch.utils.data import Dataset

from .base import Pipeline, Step
from .loggers import (LogChecksArtifacts,
                        LogModelCheckpoint,
                        LogTrainingArtifact,
                        LogDatasetMetadata
                    )
from .loaders import LoadDatasetArtifact, LoadDeepchecksArtifacts, LoadModelCheckpoint, LoadTrainingArtifact
from .data_ingestion import DataIngestor
from .checks import Checks
from ..artifacts import ArtifactRepository, ArtifactPath
from ...integrations import MLflowManager
from ..config import DeepchecksConfig, DefaultPaths
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)

class TrainLoggingPipeline(Pipeline):
    def __init__(self,
                dataset_name:str,
                run_id:str,
                run_name:Optional[str]=None,
                experiment_name:Optional[str]=None,
                mlflow_tracking_uri:Optional[str]=None,
                sqlite_path:Optional[str]=None,
                batch_size:int=8,
                model_evaluation_checks:bool=True,
                model:Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                ):

        self.model_evaluation_checks = model_evaluation_checks
        
        mlflow_manager = MLflowManager(tracking_uri=mlflow_tracking_uri,
                               create_if_not_exists=True,
                               experiment_name=experiment_name,
                               run_id=run_id,
                               run_name=run_name,
                            )

        steps_evaluation = []
        sqlite_path = sqlite_path or DefaultPaths.ARTIFACTS_SQLITE_PATH
        cfg = dict(mlflow_manager=mlflow_manager,sqlite_path=sqlite_path)
        if self.model_evaluation_checks:
            deepchecks_config = DeepchecksConfig(model_evaluation=model_evaluation_checks,
                                                train_test_validation=False,
                                                data_integrity=False,
                                                batch_size=batch_size,
                                            )
            steps_evaluation.append(DataIngestor(batch_size=deepchecks_config.batch_size,model=model),
                                    Checks(deepchecks_config=deepchecks_config,
                                           dataset_name=dataset_name),
                                    LogChecksArtifacts(**cfg)
                                    )
        steps = [LogModelCheckpoint(**cfg),
                LogTrainingArtifact(**cfg),
                *steps_evaluation
            ]
        super().__init__(steps=steps)
    
    def run(self,
            metric_names:List[str],        
            checkpoint_artifact_path:str,
            train_data:Optional[Dataset]=None,
            test_data:Optional[Dataset]=None,
            )->dict:
        self.context = {}
        self.context['checkpoint_artifact_path'] = checkpoint_artifact_path
        self.context['metric_names'] = metric_names
        if self.model_evaluation_checks:
            assert train_data is not None, "train_data must be provided if model_evaluation_checks is True"
            self.context['train_data'] = train_data
            self.context['test_data'] = test_data
        return super().run(**self.context)


class ChecksPipeline(Pipeline):
    def __init__(self,
                 dataset_name:str,
                 train_test_validation:bool=True,
                 data_integrity:bool=True,
                 model_evaluation:bool=False,
                 model:Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                 mlflow_tracking_uri:Optional[str]=None,
                 sqlite_path:Optional[str]=None,
                 batch_size:int=16,
                 max_samples:Optional[int]=None,
                 random_state:int=42,
                 save_results:bool=False,
                 output_dir:Optional[str]=None,
                 log_artifacts:bool=True
                 ):
        deepchecks_config = DeepchecksConfig(train_test_validation=train_test_validation,
                                            data_integrity=data_integrity,
                                            model_evaluation=model_evaluation,
                                            batch_size=batch_size,
                                            max_samples=max_samples,
                                            random_state=random_state,
                                            save_results=save_results,
                                            output_dir=output_dir,
                                        )
        sqlite_path = sqlite_path or DefaultPaths.ARTIFACTS_SQLITE_PATH        
        steps = [DataIngestor(batch_size=deepchecks_config.batch_size,model=model),
                Checks(deepchecks_config=deepchecks_config,
                       dataset_name=dataset_name),
            ]
        if log_artifacts:
            mlflow_manager = MLflowManager(tracking_uri=mlflow_tracking_uri or DefaultPaths.MLFLOW_TRACKING_URI.value,
                               create_if_not_exists=True,
                               experiment_name=DefaultPaths.DATASETS_EXPERIMENT_NAME.value,
                               run_name=dataset_name,
                            )
            steps.append(LogChecksArtifacts(mlflow_manager=mlflow_manager,sqlite_path=sqlite_path))
        super().__init__(steps=steps)
    
    def run(self,
            train_data:Dataset,
            test_data:Optional[Dataset]=None,         
            )->dict:
        self.context = {}
        self.context['test_data'] = test_data
        self.context['train_data'] = train_data
        return super().run(**self.context)


class DatasetLoggingPipeline(Pipeline):
    def __init__(self,
                 dataset_name:str,
                 batch_size:int=16,
                 mlflow_tracking_uri:Optional[str]=None,
                 sqlite_path:Optional[str]=None,
                 train_test_validation:bool=True,
                 data_integrity:bool=True,
                 max_samples:Optional[int]=None,
                 random_state:int=42,
                 save_results:bool=False,
                 output_dir:Optional[str]=None,
                 dataset_experiment_name:Optional[str]=None,
                 ):

        sqlite_path = sqlite_path or DefaultPaths.ARTIFACTS_SQLITE_PATH
        if self.check_if_exists(dataset_name,sqlite_path):
            raise ValueError(f"Dataset {dataset_name} already exists in the database.")
        
        deepchecks_config = DeepchecksConfig(model_evaluation=False,
                                            train_test_validation=train_test_validation,
                                            data_integrity=data_integrity,
                                            batch_size=batch_size,
                                            max_samples=max_samples,
                                            random_state=random_state,
                                            save_results=save_results,
                                            output_dir=output_dir,
                                        )
        mlflow_manager = MLflowManager(tracking_uri=mlflow_tracking_uri or DefaultPaths.MLFLOW_TRACKING_URI.value,
                               create_if_not_exists=True,
                               experiment_name=dataset_experiment_name or DefaultPaths.DATASETS_EXPERIMENT_NAME.value,
                               run_name=dataset_name,
                            )        
        cfg = dict(mlflow_manager=mlflow_manager,sqlite_path=sqlite_path)
        steps = [LogDatasetMetadata(dataset_name=dataset_name,**cfg),
            ]        
        if train_test_validation or data_integrity:
            steps.extend([DataIngestor(batch_size=batch_size,model=None),
                        Checks(deepchecks_config=deepchecks_config,
                                dataset_name=dataset_name),
                        LogChecksArtifacts(**cfg)]
                        )
        super().__init__(steps=steps)
    
    def check_if_exists(self,dataset_name:str,sqlite_path:str)->bool:
        repo = ArtifactRepository(sqlite_path)
        return repo.get(dataset_name, ArtifactPath.DATASET.value) is not None
    
    def run(self,
            train_data:Dataset,
            test_data:Optional[Dataset]=None
            )->dict:
        self.context = {}
        self.context['test_data'] = test_data
        self.context['train_data'] = train_data
        for step in self.steps:
            step.run(context=self.context)
        return self.context


class ArtifactLoadingPipeline(Pipeline):
    def __init__(self,
                 mlflow_tracking_uri:str,
                 run_id:str,
                 sqlite_path:Optional[str]=None,                 
                 load_dataset_metadata:bool=True,
                 dataset_name:Optional[str]=None,
                 dataset_experiment_name:Optional[str]=None,
                 load_checks:bool=True,
                 load_model_checkpoint:bool=True,
                 load_training:bool=True,
                 ):
        
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.dataset_name = dataset_name
        self.sqlite_path = sqlite_path or DefaultPaths.ARTIFACTS_SQLITE_PATH
        self.dataset_experiment_name = dataset_experiment_name or DefaultPaths.DATASETS_EXPERIMENT_NAME.value
        self.load_model_checkpoint = load_model_checkpoint
        self.load_training = load_training
        self.load_checks = load_checks
        self.load_dataset_metadata = load_dataset_metadata

        self.mlflow_manager = MLflowManager(tracking_uri=self.mlflow_tracking_uri,
                    run_id=run_id,
                )
        super().__init__(steps=self._load_steps())
    
    def _load_steps(self)->list[Step]:
        steps = []
        cfg = dict(mlflow_manager=self.mlflow_manager,artifact_sqlite_path=self.sqlite_path)
        if self.load_dataset_metadata:
            assert isinstance(self.dataset_name, str), "dataset_name must be a string"
            mlflow_manager = MLflowManager(tracking_uri=self.mlflow_tracking_uri,
                                           experiment_name=self.dataset_experiment_name,
                                        )
            steps.append(LoadDatasetArtifact(dataset_name=self.dataset_name,artifact_sqlite_path=self.sqlite_path,mlflow_manager=mlflow_manager))
        if self.load_checks:
            steps.append(LoadDeepchecksArtifacts(**cfg))
        if self.load_model_checkpoint:
            steps.append(LoadModelCheckpoint(**cfg))
        if self.load_training:
            steps.append(LoadTrainingArtifact(**cfg))
        return steps

    
    def run(self,)->dict:
        self.context = {}
        return super().run(**self.context)