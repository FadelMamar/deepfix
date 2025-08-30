import timm
import torch
from torchvision import transforms as T
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from typing import List

def get_timm_model(model_name: str,pretrained: bool = True,num_classes: int = 10)->torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained,num_classes=num_classes)
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    trfs = [t for t in transform.transforms if isinstance(t,(T.Normalize,T.Resize,T.ToTensor))]
    model = torch.nn.Sequential(*trfs,model)
    return model

class CLIPModel(torch.nn.Module):
    def __init__(self, timm_model_name: str,labels_list: List[str],device:str="cpu"):
        super().__init__()
        self.model, self.preprocess = create_model_from_pretrained(f"hf-hub:timm/{timm_model_name}")
        self.tokenizer = get_tokenizer(f"hf-hub:timm/{timm_model_name}")
        self.labels_list = labels_list
        self.text = self.tokenizer(self.labels_list, context_length=self.model.context_length)
        self.transforms = [t for t in self.preprocess.transforms if isinstance(t,(T.Normalize,T.Resize))]
        self.preprocess = torch.nn.Sequential(*self.transforms)
        self.device = device

        self.model.to(self.device)
        self.text.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x.float().to(self.device)
        image = self.preprocess(x).to(self.device)
        with torch.inference_mode():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(self.text, normalize=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        text_probs = text_probs.cpu()
        return text_probs
    
    def predict(self, x: torch.Tensor)->torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        probs = self.forward(x)
        return probs.argmax(dim=-1)

