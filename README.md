# 🔍 DeepSight - MLOps Copilot for Computer Vision

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-3.0+-orange.svg)](https://mlflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)

**DeepSight** is an intelligent MLOps copilot designed to provide systematic guidance for computer vision model training, debugging, and overfitting analysis. It integrates seamlessly with popular MLOps tools to deliver actionable insights through automated testing, research assistance, and comprehensive reporting.

## 🎯 Key Features

### 🔬 **Intelligent Model Analysis**
- **Overfitting Detection**: Automated detection and analysis of overfitting patterns in CV models
- **Performance Monitoring**: Real-time tracking of training metrics with intelligent alerts
- **Learning Curve Analysis**: Advanced analysis of model behavior during training

### 🛠️ **MLOps Tool Integration**
- **MLflow Integration**: Seamless experiment tracking, model registry, and artifact management
- **Deepchecks Validation**: Automated model validation with computer vision-specific checks
- **DVC Support**: Data versioning and pipeline management integration
- **PyTorch Lightning**: Advanced training framework with built-in best practices

### 📊 **Advanced Training Capabilities**
- **Classification Trainer**: Production-ready trainer with configurable hyperparameters
- **Multi-GPU Support**: Distributed training with automatic device detection
- **Model Zoo**: Pre-configured models including CLIP and TIMM architectures
- **Custom Datasets**: Built-in support for various CV datasets and formats

### 🔍 **Research Assistant**
- **Academic Paper Discovery**: Automated literature search based on model characteristics
- **Solution Recommendations**: Evidence-based suggestions for model improvement
- **Problem Classification**: Intelligent categorization of training issues

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/deepsight.git
cd deepsight

# Install with uv (recommended)
uv sync

# Or install with pip
uv pip install -e .
```

### Basic Usage

#### 1. Start MLflow Tracking Server
```bash
# Launch MLflow server
./scripts/launch_mlflow.sh

# Or manually
mlflow server --host 0.0.0.0 --port 5000
```

#### 2. Train a Classification Model
```python
from deepsight.zoo.foodwaste import load_train_and_val_datasets
from deepsight.zoo.trainers.classification import ClassificationTrainer, ClassificationTrainerConfig

# Load your datasets
train_dataset, val_dataset = load_train_and_val_datasets()

# Configure the trainer
config = ClassificationTrainerConfig(
    num_classes=train_dataset.num_classes,
    label_to_class_map=train_dataset.label_to_class_map,
    batch_size=16,
    epochs=10,
    lr=1e-3,
    experiment_name="my_cv_experiment",
    tracking_uri="http://localhost:5000"
)

# Train the model
trainer = ClassificationTrainer(config)
trainer.run(train_dataset=train_dataset, val_dataset=val_dataset)
```

#### 3. Run Model Validation Checks
```python
from deepsight.integrations import DeepchecksRunner
from deepsight.utils import DeepchecksConfig
from deepsight.core.data import ClassificationVisionDataLoader

# Configure validation checks
config = DeepchecksConfig(
    train_test_validation=True,
    data_integrity=True,
    model_evaluation=True,
    save_results=True,
    output_dir='results'
)

# Run comprehensive validation suite
runner = DeepchecksRunner(config)
results = runner.run_suites(train_data=train_data, test_data=test_data)
```

## 🏗️ Architecture

```
deepsight/
├── 📁 core/                    # Core analysis engine
│   ├── data/                   # Data loading and processing
│   ├── db/                     # Database integrations
│   ├── ingestors/              # Data ingestion utilities
│   ├── llms/                   # Language model integrations
│   ├── retrievers/             # Information retrieval
│   └── researcher.py           # Research assistant
├── 📁 integrations/            # External tool integrations
│   ├── deepchecks.py          # Deepchecks validation runner
│   ├── dvc.py                 # DVC data management
│   └── mlflow.py              # MLflow experiment tracking
├── 📁 zoo/                     # Model zoo and trainers
│   ├── trainers/              # Training frameworks
│   ├── foodwaste.py           # Sample dataset implementation
│   └── timm_models.py         # TIMM model integrations
├── 📁 utils/                   # Shared utilities
│   ├── config.py              # Configuration management
│   ├── logging.py             # Structured logging
│   └── validators.py          # Input validation
└── 📁 cli/                     # Command-line interface
    ├── main.py                # CLI entry point
    └── commands.py            # Command implementations
```

## 🔧 Configuration

DeepSight uses YAML configuration files for easy customization:

```yaml
# deepsight_config.yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "cv_model_analysis"

deepchecks:
  suite: "computer_vision"
  custom_checks:
    - "overfitting_analysis"
    - "data_drift_detection"

detection:
  thresholds:
    train_val_gap: 0.1
    learning_curve_slope: 0.05
  metrics: ["accuracy", "loss", "f1_score"]

output:
  format: ["html", "pdf"]
  include_visualizations: true
```

## 📦 Dependencies

### Core Technologies
- **Python 3.11+**: Modern Python features and performance
- **PyTorch 2.8.0**: Deep learning framework
- **PyTorch Lightning**: Advanced training framework
- **MLflow 3.3+**: Experiment tracking and model registry
- **Deepchecks**: Model validation and testing

### Computer Vision Stack
- **torchvision**: Image processing and models
- **timm**: State-of-the-art vision models
- **open-clip-torch**: CLIP model implementations
- **fiftyone**: Dataset visualization and management

### MLOps Tools
- **DVC**: Data version control
- **Hugging Face Hub**: Model and dataset hosting
- **OmegaConf**: Hierarchical configuration management

## 📊 Examples

Explore the `examples/` directory for comprehensive tutorials:

- **`run_training.py`**: Complete training pipeline example
- **`run_suite_of_checks.py`**: Model validation workflow
- **`vision_dataloader.ipynb`**: Interactive data exploration notebook

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-org/deepsight.git
cd deepsight
uv sync --dev

# Run tests
pytest src/tests/

# Format code
black src/
isort src/
```

## 📖 Documentation

- **[API Reference](docs/api/)**: Detailed API documentation
- **[User Guide](docs/guide/)**: Step-by-step tutorials
- **[Best Practices](docs/best-practices/)**: MLOps recommendations
- **[Examples](examples/)**: Practical usage examples

## 🔬 Research & Citations

DeepSight incorporates cutting-edge research in computer vision and MLOps. If you use DeepSight in your research, please cite:

```bibtex
@software{deepsight2024,
  title={DeepSight: MLOps Copilot for Computer Vision},
  author={DeepSight Team},
  year={2024},
  url={https://github.com/your-org/deepsight}
}
```

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/your-org/deepsight/issues)
- **Discussions**: Join the community on [GitHub Discussions](https://github.com/your-org/deepsight/discussions)
- **Email**: Contact the team at team@deepsight.ai

## 🌟 Acknowledgments

Special thanks to the open-source community and the following projects that make DeepSight possible:
- PyTorch and PyTorch Lightning teams
- MLflow community
- Deepchecks developers
- TIMM and OpenCLIP contributors

---

**Built with ❤️ for the Computer Vision and MLOps community**
