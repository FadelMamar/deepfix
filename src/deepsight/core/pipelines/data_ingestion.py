from torch.utils.data import Dataset
from deepchecks.vision import VisionData
from typing import Optional, Tuple, Callable
import torch

from .base import Step
from ..data import ClassificationVisionDataLoader


class DataIngestor(Step):
    
    def run(
        self,
        context:dict,
        train_data:Optional[Dataset]=None,
        batch_size:Optional[int]=8,
        test_data:Optional[Dataset]=None,
        model:Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
    ) -> dict:
        
        train_data = ClassificationVisionDataLoader.load_from_dataset(
            train_data or context.get("train_data"), batch_size=batch_size, model= model or context.get('model')
        )
        if context.get("test_data") is not None:
            test_data = ClassificationVisionDataLoader.load_from_dataset(
                test_data or context.get("test_data"), batch_size=batch_size, model=model or context.get('model')
            )
        context['train_data'] = train_data
        context['test_data'] = test_data
        return context
