from deepchecks.vision import VisionData, BatchOutputFormat
from torch.utils.data import DataLoader
import torch
from typing import Dict, Optional

def classification_collate(data) -> BatchOutputFormat:
    # Extracting images and label and converting images of (N, C, H, W) into (N, H, W, C)
    images = torch.stack([x[0] for x in data]).permute(0, 2, 3, 1)
    labels = [x[1] for x in data]
    return BatchOutputFormat(images= images, labels= labels)

class ClassificationVisionDataLoader:
    def __init__(self,):
        pass
    
    @classmethod
    def load_from_dataloader(cls, dataloader: DataLoader,label_map: Optional[Dict[int, str]] = None) -> VisionData:
        vision_data = VisionData(dataloader, task_type='classification', collate_fn=classification_collate, label_map=label_map)
        # Visualize the data and verify it is in the correct format
        vision_data.head()
        return vision_data


