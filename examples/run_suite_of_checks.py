from deepsight.zoo.foodwaste import get_label_mapping,translations_de_en,load_train_and_val_datasets
from deepsight.core.data import ClassificationVisionDataLoader
from deepsight.integrations import DeepchecksRunner
from deepsight.utils import DeepchecksConfig
from deepsight.zoo.timm_models import CLIPModel

import json

def main():
    ing2name, ing2label = get_label_mapping()
    train_dataset, val_dataset = load_train_and_val_datasets(image_size=1024)
    ingredients_en = [ "a " + translations_de_en[t] for t in ing2name.values()]

    config = DeepchecksConfig(train_test_validation=True,
                            data_integrity=True,
                            model_evaluation=False,
                            save_results=True,
                            save_display=False,
                            save_results_format='json',
                            parse_results=True,
                            output_dir='results')

    runner = DeepchecksRunner(config)

    model = None # CLIPModel('PE-Core-T-16-384',ingredients_en)

    vision_train_data = ClassificationVisionDataLoader.load_from_dataset(train_dataset,batch_size=8,shuffle=True,model=model)
    vision_test_data = ClassificationVisionDataLoader.load_from_dataset(val_dataset,batch_size=8,shuffle=True,model=model)
    results = runner.run_suites(train_data=vision_train_data,test_data=vision_test_data)
    return results

if __name__ == "__main__":
    main()