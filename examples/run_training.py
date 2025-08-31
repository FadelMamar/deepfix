from pathlib import Path

from deepsight.zoo.trainers.classification import ClassificationTrainer,ClassificationTrainerConfig
from deepsight.zoo.timm_models import TimmClassificationModel, ClassifierHead
from deepsight.utils.feature_extractor import FeatureExtractor
from deepsight.integrations.lightning import DeepSightCallback
from deepsight.utils.config import DeepchecksConfig


ROOT = Path(__file__).parents[1]
DEBUG = False

def load_foodwaste_dataset(embedding_model:FeatureExtractor):
    from deepsight.zoo.datasets.foodwaste import load_train_and_val_datasets, FoodWasteDatasetWithEmbeddings
    #train_dataset, val_dataset = load_train_and_val_datasets(embedding_model=embedding_model,num_workers=4)
    #train_dataset, val_dataset = load_train_and_val_datasets(embedding_model=embedding_model,num_workers=4)
    #train_dataset.save_embeddings(str(ROOT / "data" / "train_embeddings.pt"))
    #val_dataset.save_embeddings(str(ROOT / "data" / "val_embeddings.pt"))
    train_dataset = FoodWasteDatasetWithEmbeddings.from_embeddings(str(ROOT / "data" / "train_embeddings.pt"))
    val_dataset = FoodWasteDatasetWithEmbeddings.from_embeddings(str(ROOT / "data" / "val_embeddings.pt"))
    return train_dataset, val_dataset

def load_food_dataset(embedding_model:FeatureExtractor):
    from deepsight.zoo.datasets.food import load_train_and_val_datasets, FoodDatasetWithEmbeddings
    #train_dataset, val_dataset = load_train_and_val_datasets(embedding_model=embedding_model,
    #                                                        num_workers=4,
    #                                                       device="cpu",
    #                                                        image_size=518)
    #train_dataset.save_embeddings(str(ROOT / "data" / "food101_train_embeddings.pt"))
    #val_dataset.save_embeddings(str(ROOT / "data" / "food101_val_embeddings.pt"))
    train_dataset = FoodDatasetWithEmbeddings.from_embeddings(str(ROOT / "data" / "food101_train_embeddings.pt"))
    val_dataset = FoodDatasetWithEmbeddings.from_embeddings(str(ROOT / "data" / "food101_val_embeddings.pt"))
    return train_dataset, val_dataset

def main():
    embedding_model = FeatureExtractor(model_name="timm/vit_base_patch14_reg4_dinov2.lvd142m")

    train_dataset, val_dataset = load_food_dataset(embedding_model)

    config = ClassificationTrainerConfig(

        # Classes
        num_classes=train_dataset.num_classes,
        label_to_class_map=train_dataset.label_to_class_map,

        # Dataloading
        batch_size=8,
        num_workers=4,
        pin_memory=False,

        # Accelerator
        accelerator="auto",
        precision="bf16-mixed",

        # Val
        val_check_interval=1,
        
        # Training
        epochs=10,
        label_smoothing=0.0,
        lr=1e-3,
        lrf=1e-2,
        weight_decay=5e-3,
        reweight_classes=False,

        # Monitoring
        monitor="val_f1score",
        patience=10,
        min_delta=1e-3,
        mode="max",

        # Mlflow
        experiment_name="foodwaste_classification",
        run_name="default",
        log_best_model=True,
        tracking_uri="http://localhost:5000",
        
        # Checkpoint
        dirpath=str(ROOT / "checkpoints"),
        filename="best-{epoch:02d}",
        save_weights_only=True,
    )
    #model = TimmClassificationModel(
    #    model_name="timm/vit_base_patch14_reg4_dinov2.lvd142m",
    #    num_classes=train_dataset.num_classes,
    #    freeze_backbone=True,
    #    hidden_dim=128,
    #    num_layers=2,
    #    dropout=0.15
    #)
    deepchecks_config = DeepchecksConfig(
        train_test_validation=True,
        data_integrity=True,
        save_results=True,
        output_dir=str(ROOT / "deepchecks_results"),
        save_display=True,
        parse_results=True,
        batch_size=8,
        model_evaluation=True,
        max_samples=1000,
        random_state=42,
    )
    deepsight_callback = DeepSightCallback(config.model_dump(),
                                        dataset_name="food101",
                                        deepchecks_config=deepchecks_config,
                                        train_dataset=train_dataset,
                                        val_dataset=val_dataset
                            )

    model = ClassifierHead(input_dim=embedding_model.feature_dim, num_classes=train_dataset.num_classes,num_layers=2,hidden_dim=384,dropout=0.2)
    trainer = ClassificationTrainer(config)
    trainer.run(model=model,train_dataset=train_dataset, val_dataset=val_dataset,deepsight_callback=deepsight_callback,debug=DEBUG)

if __name__ == "__main__":
    main()