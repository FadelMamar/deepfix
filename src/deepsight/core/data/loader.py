from deepchecks.vision import VisionData, BatchOutputFormat
from torch.utils.data import DataLoader,Dataset
import torch
from typing import Dict, Optional

def classification_collate(data) -> BatchOutputFormat:
    # Extracting images and label and converting images of (N, C, H, W) into (N, H, W, C)
    images = torch.stack([x[0] for x in data]).permute(0, 2, 3, 1).cpu().numpy()
    labels = [x[1] for x in data]
    return BatchOutputFormat(images= images, labels= labels)

class ClassificationVisionDataLoader:
    def __init__(self,):
        pass
    
    @classmethod
    def load_from_dataset(cls,dataset: Dataset,batch_size: int = 8,shuffle: bool = True) -> VisionData:
        assert isinstance(dataset,Dataset), "dataset must be an instance of torch.utils.data.Dataset. Received: {}".format(type(dataset))
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=classification_collate)
        return cls.load_from_dataloader(dataloader)
    
    @classmethod
    def load_from_dataloader(cls, dataloader: DataLoader,label_map: Optional[Dict[int, str]] = None) -> VisionData:
        assert isinstance(dataloader,DataLoader), "dataloader must be an instance of torch.utils.data.DataLoader. Received: {}".format(type(dataloader))
        vision_data = VisionData(dataloader, task_type='classification',label_map=label_map)
        # Visualize the data and verify it is in the correct format
        vision_data.head()
        return vision_data


