"""
Basic usage example for DeepSight Advisor.

This example demonstrates how to use the advisor with minimal configuration
to run a complete ML analysis pipeline.
"""

import fire
import torch
from torchvision.transforms.v2 import ToDtype

from deepsight.core.config import DeepchecksConfig
from deepsight.integrations import MLflowManager
from deepsight.zoo.datasets.foodwaste import load_train_and_val_datasets
from deepsight.zoo.timm_models import get_timm_model
from deepsight.core.pipelines.factory import TrainLoggingPipeline,DatasetLoggingPipeline,ChecksPipeline


tracking_uri="file:./mlflow_data"
    
mlflow_mgr = MLflowManager(tracking_uri=tracking_uri,
                               create_if_not_exists=True,
                               experiment_name="deepsight_example",
                               )
dataset_name = "cafetaria-foodwaste-lstroetmann"


def example_train_logging():
    train_logging_pipeline = TrainLoggingPipeline(mlflow_manager=mlflow_mgr,
                                                dataset_name=dataset_name,
                                                batch_size=8
                                                )
    train_logging_pipeline.run(metric_names=["loss","accuracy"],
                                checkpoint_artifact_path=None
                                    )

def example_dataset_logging():
    train_data, val_data = load_train_and_val_datasets(
        image_size=448,
        batch_size=8,
        num_workers=4,
        pin_memory=False,
    )
    mlflow_mgr = MLflowManager(tracking_uri=tracking_uri,
                               create_if_not_exists=True,
                               experiment_name="deepfix_datasets",
                               run_name=dataset_name,
                               )
    dataset_logging_pipeline = DatasetLoggingPipeline(mlflow_manager=mlflow_mgr,
                                                dataset_name=dataset_name,
                                                train_test_validation=True,
                                                data_integrity=True,
                                                batch_size=8
                                                )
    dataset_logging_pipeline.run(train_data=train_data,
                                test_data=val_data,
                                    )
    

def example_checks():    
    train_data, val_data = load_train_and_val_datasets(
        image_size=448,
        batch_size=8,
        num_workers=4,
        pin_memory=False,
    )

    #model = get_timm_model(model_name="timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k", pretrained=True, num_classes=train_data.num_classes)
    predictor = None #torch.nn.Sequential(ToDtype(torch.float32,scale=False),model, torch.nn.Softmax(dim=1)
    
    deepchecks_config = DeepchecksConfig(train_test_validation=True,
                                         data_integrity=True,
                                         model_evaluation=True,
                                         batch_size=8
                                        )
    checks_pipeline = ChecksPipeline(mlflow_manager=mlflow_mgr,
                                    dataset_name=dataset_name,
                                    deepchecks_config=deepchecks_config,
                                    model=predictor,
                                    batch_size=8,
                                    log_artifacts=True
                                    )    
    checks_pipeline.run(train_data=train_data,
                        test_data=val_data
                        )

if __name__ == "__main__":
    fire.Fire()
