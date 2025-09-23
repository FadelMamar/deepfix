import traceback
from typing import Callable, Optional, List
import torch
from torch.utils.data import Dataset

from .base import Pipeline
from .artifacts import (LogChecksArtifacts,
                        LogModelCheckpoint,
                        LogTrainingArtifact,
                        LogDatasetMetadata
                    )
from .data_ingestion import DataIngestor
from .checks import Checks
from ...integrations import MLflowManager
from ..config import DeepchecksConfig, DefaultPaths
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)

class TrainLoggingPipeline(Pipeline):
    def __init__(self,
                mlflow_manager:MLflowManager,
                dataset_name:str,
                batch_size:int=16,
                model_evaluation_checks:bool=True,
                model:Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                ):

        steps_evaluation = []
        if model_evaluation_checks:
            deepchecks_config = DeepchecksConfig(model_evaluation=model_evaluation_checks,
                                                train_test_validation=False,
                                                data_integrity=False,
                                                batch_size=batch_size,
                                            )
            steps_evaluation.append(DataIngestor(batch_size=deepchecks_config.batch_size,model=model),
                                    Checks(deepchecks_config=deepchecks_config,
                                           dataset_name=dataset_name),
                                    LogChecksArtifacts(mlflow_manager=mlflow_manager)
                                    )
        steps = [LogModelCheckpoint(mlflow_manager=mlflow_manager),
                LogTrainingArtifact(mlflow_manager=mlflow_manager),
                *steps_evaluation
            ]
        super().__init__(steps=steps)
    
    def run(self,
            metric_names:List[str],        
            checkpoint_artifact_path:str
            )->dict:
        self.context = {}
        self.context['checkpoint_artifact_path'] = checkpoint_artifact_path
        self.context['metric_names'] = metric_names
        return super().run(**self.context)


class ChecksPipeline(Pipeline):
    def __init__(self,
                 dataset_name:str,
                 train_test_validation:bool=True,
                 data_integrity:bool=True,
                 model_evaluation:bool=False,
                 model:Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                 mlflow_tracking_uri:Optional[str]=None,
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
            steps.append(LogChecksArtifacts(mlflow_manager=mlflow_manager))
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
                 train_test_validation:bool=True,
                 data_integrity:bool=True,
                 max_samples:Optional[int]=None,
                 random_state:int=42,
                 save_results:bool=False,
                 output_dir:Optional[str]=None,
                 ):
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
                               experiment_name=DefaultPaths.DATASETS_EXPERIMENT_NAME.value,
                               run_name=dataset_name,
                            )
        steps = [LogDatasetMetadata(mlflow_manager=mlflow_manager,
                                   dataset_name=dataset_name),
                DataIngestor(batch_size=batch_size,model=None),
                Checks(deepchecks_config=deepchecks_config,
                        dataset_name=dataset_name),
                LogChecksArtifacts(mlflow_manager=mlflow_manager),
            ]        
        super().__init__(steps=steps)
    
    def run(self,
            train_data:Dataset,
            test_data:Optional[Dataset]=None
            )->dict:
        self.context = {}
        self.context['test_data'] = test_data
        self.context['train_data'] = train_data
        return super().run(**self.context)
    
