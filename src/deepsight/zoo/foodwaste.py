from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from itertools import chain

def get_label_mapping():
    ing2name = {}
    dataset = load_dataset("AI-ServicesBB/food-waste-dataset")
    for d in tqdm(chain(dataset['train'],dataset['test']),desc="Getting label mapping",total=len(dataset['train'])+len(dataset['test'])):
        for n,ing in zip(d['Artikelnummer'],d['Artikel']):
            if n not in ing2name:
                ing2name[n] = str(ing).lower().strip().replace(' ','_')
    ing2label = {k:i for i,k in enumerate(ing2name.keys())}
    return ing2name, ing2label

def create_classification_dataset(ing2label:dict,split='train',image_size:int=1024):
    return FoodWasteDataset(ing2label,split=split,image_size=image_size)

class FoodWasteDataset(Dataset):
    def __init__(self,ing2label:dict,split='train',image_size:int=1024):
        self.dataset = load_dataset("AI-ServicesBB/food-waste-dataset")[split]
        self.transform = T.Compose([
            T.PILToTensor(),
            T.Resize((image_size,image_size),interpolation=T.InterpolationMode.NEAREST),
        ])
        self.ing2label = ing2label

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        ing_ids = sample['Artikelnummer']
        weights = sample['Menge_Rückläufer']
        i = np.argmax(weights)
        ing_id = ing_ids[i]
        label = self.ing2label[ing_id]
        return image, label