from typing import Optional
import mlflow
import traceback

from .base import Step
from ..artifacts import ArtifactPaths
from ...utils.logging import get_logger
from ...integrations import MLflowManager

LOGGER = get_logger(__name__)

class ModelRegistration(Step):

    def run(self,best_model_path:str,run_id:Optional[str]=None,run_name:Optional[str]=None):
        pass
            
class ModelLoader(Step):

    def run(self,context:dict,run_id:str,tracking_uri:str,dwnd_dir:str="mlflow_downloads",experiment_name:Optional[str]=None)->dict:
        mgr = MLflowManager(tracking_uri=tracking_uri,
                            experiment_name=experiment_name,
                            run_id=run_id,
                            dwnd_dir=dwnd_dir
                        )
        path_to_ckpt = mgr.get_model_checkpoint()
        context['model_path'] = path_to_ckpt
        return context

    