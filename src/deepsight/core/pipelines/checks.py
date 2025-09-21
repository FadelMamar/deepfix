from deepchecks.vision import VisionData
from typing import Optional

from .base import Step
from ...integrations import DeepchecksRunner,DeepchecksConfig


class Checks(Step):

    def run(self,
            context:dict,
            dataset_name:str,
            deepchecks_config:Optional[DeepchecksConfig]=None,
            train_data:Optional[VisionData]=None,
            test_data:Optional[VisionData]=None)->dict:
        deepchecks_runner = DeepchecksRunner(config=deepchecks_config or context.get("deepchecks_config"))
        artifacts = deepchecks_runner.run_suites(
            train_data=train_data or context.get("train_data"),
            test_data=test_data or context.get("test_data"),
            dataset_name=dataset_name or context.get("dataset_name"),
        )
        context['deepchecks_artifacts'] = artifacts
        return context
    