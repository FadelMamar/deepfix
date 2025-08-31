from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from typing import Tuple,Dict

eng =['dressing_portion',
 'sauerkraut',
 'iceberg lettuce',
 'bell pepper',
 'potato pancakes',
 'red cabbage',
 'zucchini',
 'mushrooms',
 'lentils',
 'apple sauce',
 'cream',
 'green beans',
 'cauliflower',
 'peas',
 'carrots',
 'rice',
 'potato cubes',
 'potatoes',
 'brown sauce',
 'light sauce',
 'savoy cabbage',
 'chicken',
 'chicken strips',
 'gravy',
 'onion',
 'bread dumplings',
 'roast beef',
 'roast pork neck',
 'ham_mettwurst',
 'grilled sausage',
 'schnitzel',
 'vegetable cream',
 'pollock',
 'egg noodles',
 'mashed potatoes',
 'meatballs',
 'breaded fish fillet',
 'lentil stew',
 'tomato curry sauce',
 'malt beer mustard sauce',
 'coleslaw']

german = ['dressing_portion',
 'sauerkraut',
 'eisbergsalat',
 'paprika',
 'reibekuchen',
 'rotkohl',
 'zucchini',
 'pilze',
 'linsen',
 'apfelmus',
 'sahne',
 'grüne_bohnen',
 'blumenkohl',
 'erbsen',
 'möhre',
 'reis',
 'kartoffelwürfel',
 'kartoffeln',
 'braune_sauce',
 'helle_sauce',
 'wirsing',
 'hähnchen',
 'hähnchenstreifen',
 'bratenjus',
 'zwiebel',
 'semmelknödel',
 'rinderbraten',
 'schweinenackenbraten',
 'schinken_mettwurst',
 'rostbratwurst',
 'schnitzel',
 'pflanzencreme',
 'seelachs',
 'eierspätzle',
 'kartoffelpüree',
 'fleischbällchen_gebrüht',
 'paniertes_fischfilet',
 'linseneintopf',
 'tomaten-curry-sauce',
 'malzbier-senf-sauce',
 'krautsalat']

translations_de_en = dict(zip(german,eng))

class FoodWasteDataset(Dataset):
    def __init__(self,ing2label:OrderedDict,ing2name:OrderedDict,split='train',image_size:int=1024):
        self.dataset = load_dataset("AI-ServicesBB/food-waste-dataset")[split]
        self.transform = T.Compose([
            T.PILToTensor(),
            T.Resize((image_size,image_size),interpolation=T.InterpolationMode.NEAREST),
        ])
        self.ing2label = ing2label
        self.ing2name = ing2name
    
    @property
    def num_classes(self):
        return len(self.ing2label)
    
    @property
    def label_to_class_map(self):
        assert list(self.ing2label.keys()) == list(self.ing2name.keys()), "labels ordering and names ordering must be the same"
        return dict(zip(self.ing2label.values(),self.ing2name.keys()))

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

def get_label_mapping()->Tuple[Dict[int,str],Dict[int,int]]:
    ing2name = {}
    dataset = load_dataset("AI-ServicesBB/food-waste-dataset")
    for d in tqdm(chain(dataset['train'],dataset['test']),desc="Getting label mapping",total=len(dataset['train'])+len(dataset['test'])):
        for n,ing in zip(d['Artikelnummer'],d['Artikel']):
            if n not in ing2name:
                ing2name[n] = str(ing).lower().strip().replace(' ','_')
    ing2name = OrderedDict(sorted(ing2name.items()))
    ing2label = OrderedDict({k:i for i,k in enumerate(ing2name.keys())})
    return ing2name, ing2label

def create_classification_dataset(ing2label:OrderedDict,split='train',image_size:int=1024):
    return FoodWasteDataset(ing2label,split=split,image_size=image_size)

def load_train_and_val_datasets(image_size:int=1024)->Tuple[FoodWasteDataset,FoodWasteDataset]:
    _, ing2label = get_label_mapping()
    train_dataset = create_classification_dataset(ing2label,split='train',image_size=image_size)
    val_dataset = create_classification_dataset(ing2label,split='val',image_size=image_size)
    return train_dataset, val_dataset