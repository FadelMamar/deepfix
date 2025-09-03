"""
DeepSight Training Example

This example demonstrates how to train a classification model using DeepSight's
training framework with comprehensive monitoring and validation.

Features demonstrated:
- Dataset loading with embeddings
- Model configuration and training
- MLflow integration for experiment tracking
- Deepchecks integration for data validation
- Lightning integration for training orchestration

Usage:
    python examples/run_training.py

Requirements:
    - MLflow server running on localhost:5000
    - Required datasets and embeddings available
    - CUDA-compatible GPU (optional, will fallback to CPU)
"""

import logging
from pathlib import Path
from typing import Optional

# DeepSight imports
from deepsight.zoo.trainers.classification import (
    ClassificationTrainer,
    ClassificationTrainerConfig,
)
from deepsight.zoo.timm_models import TimmClassificationModel, ClassifierHead
from deepsight.utils.feature_extractor import FeatureExtractor
from deepsight.integrations.lightning import DeepSightCallback
from deepsight.utils.config import DeepchecksConfig
from deepsight.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Configuration
ROOT = Path(__file__).parents[1]
DEBUG = True


def load_foodwaste_dataset(embedding_model: FeatureExtractor):
    """
    Load the FoodWaste dataset with pre-computed embeddings.

    Args:
        embedding_model: Feature extractor model for generating embeddings

    Returns:
        Tuple of (train_dataset, val_dataset) with embeddings
    """
    from deepsight.zoo.datasets.foodwaste import (
        load_train_and_val_datasets,
        FoodWasteDatasetWithEmbeddings,
    )

    # Option 1: Generate embeddings from scratch (uncomment if needed)
    # train_dataset, val_dataset = load_train_and_val_datasets(
    #     embedding_model=embedding_model,
    #     num_workers=4
    # )
    # train_dataset.save_embeddings(str(ROOT / "data" / "train_embeddings.pt"))
    # val_dataset.save_embeddings(str(ROOT / "data" / "val_embeddings.pt"))

    # Option 2: Load pre-computed embeddings (faster)
    train_dataset = FoodWasteDatasetWithEmbeddings.from_embeddings(
        str(ROOT / "data" / "train_embeddings.pt")
    )
    val_dataset = FoodWasteDatasetWithEmbeddings.from_embeddings(
        str(ROOT / "data" / "val_embeddings.pt")
    )

    logger.info(
        f"Loaded FoodWaste dataset: {len(train_dataset)} train, {len(val_dataset)} val samples"
    )
    return train_dataset, val_dataset


def load_food_dataset(embedding_model: Optional[FeatureExtractor] = None):
    """
    Load the Food-101 dataset with optional embeddings.

    Args:
        embedding_model: Optional feature extractor for generating embeddings

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from deepsight.zoo.datasets.food import (
        load_train_and_val_datasets,
        FoodDatasetWithEmbeddings,
    )

    # Load dataset with limited samples for faster training
    train_dataset, val_dataset = load_train_and_val_datasets(
        embedding_model=embedding_model,
        num_workers=4,
        split_size=":500",  # Limit to 500 samples for demo
        device="cpu",
        image_size=518,
    )

    # Option to save/load embeddings for faster subsequent runs
    # train_dataset.save_embeddings(str(ROOT / "data" / "food101_train_embeddings.pt"))
    # val_dataset.save_embeddings(str(ROOT / "data" / "food101_val_embeddings.pt"))
    # train_dataset = FoodDatasetWithEmbeddings.from_embeddings(str(ROOT / "data" / "food101_train_embeddings.pt"))
    # val_dataset = FoodDatasetWithEmbeddings.from_embeddings(str(ROOT / "data" / "food101_val_embeddings.pt"))

    logger.info(
        f"Loaded Food-101 dataset: {len(train_dataset)} train, {len(val_dataset)} val samples"
    )
    return train_dataset, val_dataset


def main():
    """
    Main training function that orchestrates the entire training pipeline.

    This function:
    1. Loads the dataset
    2. Configures the training parameters
    3. Sets up the model
    4. Configures monitoring and validation
    5. Runs the training process
    """
    try:
        logger.info("Starting DeepSight training example...")

        # Load dataset
        embedding_model = None  # FeatureExtractor(model_name="timm/vit_base_patch14_reg4_dinov2.lvd142m")
        train_dataset, val_dataset = load_food_dataset(embedding_model)

        # Configure training parameters
        config = ClassificationTrainerConfig(
            # Dataset configuration
            num_classes=train_dataset.num_classes,
            label_to_class_map=train_dataset.label_to_class_map,
            # Data loading configuration
            batch_size=8,
            num_workers=4,
            pin_memory=False,
            # Hardware configuration
            accelerator="auto",  # Automatically detect GPU/CPU
            precision="bf16-mixed",  # Mixed precision for efficiency
            # Validation configuration
            val_check_interval=1,  # Validate every epoch
            # Training hyperparameters
            epochs=5,
            label_smoothing=0.0,
            lr=1e-3,  # Learning rate
            lrf=1e-2,  # Final learning rate
            weight_decay=5e-4,
            reweight_classes=False,
            # Monitoring configuration
            monitor="val_f1score",  # Metric to monitor for early stopping
            patience=10,
            min_delta=1e-3,
            mode="max",
            # MLflow configuration
            experiment_name="foodwaste_classification",
            run_name="default",
            log_best_model=True,
            tracking_uri="http://localhost:5000",
            # Checkpoint configuration
            dirpath=str(ROOT / "checkpoints"),
            filename="best-{epoch:02d}",
            save_weights_only=True,
        )

        logger.info(
            f"Training configuration: {config.num_classes} classes, {config.epochs} epochs"
        )

        # Initialize model
        model = TimmClassificationModel(
            model_name="timm/mobilenetv4_hybrid_large.e600_r384_in1k",
            num_classes=train_dataset.num_classes,
            freeze_backbone=True,  # Freeze pretrained backbone
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
        )

        # Alternative: Use classifier head with embeddings
        # model = ClassifierHead(
        #     input_dim=embedding_model.feature_dim,
        #     num_classes=train_dataset.num_classes,
        #     num_layers=2,
        #     hidden_dim=384,
        #     dropout=0.2
        # )

        logger.info(f"Initialized model: {model.__class__.__name__}")

        # Configure Deepchecks for data validation
        deepchecks_config = DeepchecksConfig(
            train_test_validation=True,  # Validate train/test splits
            data_integrity=True,  # Check data quality
            save_results=True,  # Save validation results
            output_dir=str(ROOT / "deepchecks_results"),
            save_display=True,  # Save visualizations
            parse_results=True,  # Parse results for analysis
            batch_size=8,
            model_evaluation=True,  # Evaluate model performance
            max_samples=1000,  # Limit samples for faster processing
            random_state=42,
        )

        # Setup DeepSight callback for monitoring
        deepsight_callback = DeepSightCallback(
            config.model_dump(),
            dataset_name="food101",
            deepchecks_config=deepchecks_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        # Initialize trainer and run training
        trainer = ClassificationTrainer(config)
        logger.info("Starting training...")

        trainer.run(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            deepsight_callback=deepsight_callback,
            debug=DEBUG,
        )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
