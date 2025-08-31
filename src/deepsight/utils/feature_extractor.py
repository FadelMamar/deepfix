"""
Feature extraction for image filtering algorithms.

This module provides feature extraction capabilities for clustering
and filtering algorithms in object detection training data selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
import timm
import torch.nn as nn
import torchvision.transforms as T


class FeatureExtractor(nn.Module):
    """
    Feature extractor.
    """

    def __init__(
        self,
        backbone: str = "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the feature extractor.
        Args:
            model_name: timm model name (default: 'timm/vit_small_patch16_224.dino')
            device: Device to run inference on ('cpu', 'cuda',)
        """
        super().__init__()
        
        self.backbone = backbone
        self.model = None
        self.transform = None
        self.device = device
        self.pil_to_tensor = T.PILToTensor()
        
        self._set_model_and_transform()
        self.to_torchscript()

    def _set_model_and_transform(self)->str:
        self.model = timm.create_model(
                self.backbone, pretrained=True, num_classes=0,global_pool=""
            )
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        self.transform = nn.Sequential(*[t for t in transform.transforms if isinstance(t, (T.Normalize,T.Resize))])

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(self.device)

    @property
    def feature_dim(self) -> int:
        """
        Return the dimension of the extracted feature vector.
        """
        return self.model.num_features

    def forward(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features
        """
        images = self._load(images).to(self.device)
        return self._forward(images)
    
    def _load(self,images: Union[torch.Tensor, List[Image.Image]]):
        if isinstance(images, torch.Tensor):
            images = images.float()
            images = self.transform(images)
        else:
            for image in images:
                assert isinstance(image, Image.Image), f"Image must be a PIL Image. Received {type(image)}"
            images = torch.stack([self.pil_to_tensor(image.convert("RGB")) for image in images],dim=0)
            images = images.float()
            images = self.transform(images)
        return images

    def _forward(self,images:torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if "vit" in self.backbone: # get CLS token for ViT models
                x = self.model(images)[:,0,:]
            else:
                x = self.model(images)
        return x.cpu()
    
    def to_torchscript(self)->None:
        self.model = torch.jit.script(self.model)
    
