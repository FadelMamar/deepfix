# DeepSight Examples

This directory contains comprehensive examples demonstrating how to use DeepSight's MLOps capabilities. Each example is designed to be self-contained and educational, showing best practices for different aspects of the DeepSight platform.

## Overview

DeepSight is an MLOps Copilot that provides:
- **Automated Training**: End-to-end model training with monitoring
- **Data Validation**: Comprehensive data quality checks using Deepchecks
- **Artifact Management**: Intelligent artifact tracking and retrieval
- **Query Generation**: AI-powered analysis and insights from your ML artifacts

## Examples

### 1. Training Example (`run_training.py`)

**Purpose**: Demonstrates complete model training pipeline with monitoring and validation.

**Features**:
- Dataset loading with optional embeddings
- Model configuration and training
- MLflow integration for experiment tracking
- Deepchecks integration for data validation
- Lightning integration for training orchestration

**Usage**:
```bash
python examples/run_training.py
```

**Requirements**:
- MLflow server running on `localhost:5000`
- Required datasets and embeddings available
- CUDA-compatible GPU (optional, will fallback to CPU)

**Key Components**:
- `ClassificationTrainer`: Main training orchestrator
- `TimmClassificationModel`: Pre-trained vision models
- `DeepSightCallback`: Monitoring and validation integration
- `DeepchecksConfig`: Data validation configuration

### 2. Data Validation Example (`run_suite_of_checks.py`)

**Purpose**: Shows how to run comprehensive data validation suites using Deepchecks.

**Features**:
- Data integrity validation
- Train/test validation
- Model evaluation (optional)
- Result parsing and analysis
- Integration with CLIP models for vision tasks

**Usage**:
```bash
python examples/run_suite_of_checks.py
```

**Requirements**:
- Required datasets available
- Deepchecks installed
- Optional: CLIP model for advanced validation

**Key Components**:
- `DeepchecksRunner`: Validation suite orchestrator
- `ClassificationVisionDataLoader`: Data preparation for validation
- `DeepchecksConfig`: Validation configuration
- `CLIPModel`: Optional advanced validation model

### 3. Artifact Management Example (`run_artifact_manager.py`)

**Purpose**: Demonstrates artifact management and retrieval from MLflow.

**Features**:
- MLflow integration for artifact tracking
- Artifact registration and retrieval
- Local caching and download management
- Artifact metadata and content access

**Usage**:
```bash
python examples/run_artifact_manager.py
```

**Requirements**:
- MLflow server running on `localhost:5000`
- Valid run_id with artifacts
- SQLite database for local caching

**Key Components**:
- `ArtifactsManager`: Main artifact management interface
- `MLflowManager`: MLflow integration
- `ArtifactPaths`: Standard artifact path definitions

### 4. Query Generation Example (`build_prompts.py`)

**Purpose**: Shows how to generate intelligent prompts from artifacts for AI analysis.

**Features**:
- Artifact loading and management
- Query generation from multiple artifacts
- Integration with AI IntelligenceProviders
- Prompt building for analysis and insights

**Usage**:
```bash
python examples/build_prompts.py
```

**Requirements**:
- MLflow server running on `localhost:5000`
- Valid run_id with Deepchecks and Training artifacts
- Optional: AI provider credentials for query execution

**Key Components**:
- `PromptBuilder`: Prompt generation from artifacts
- `IntelligenceClient`: AI provider integration
- `ArtifactsManager`: Artifact loading and management

### 5. DeepSight Advisor Example (`advisor_usage.py`)

**Purpose**: Demonstrates the complete DeepSight Advisor workflow - a unified orchestrator that automates the entire ML analysis pipeline from artifact loading to AI-powered insights.

**Features**:
- Complete end-to-end ML analysis pipeline
- Multiple configuration methods (convenience function, config objects, dictionaries)
- Automated artifact loading and query generation
- AI-powered analysis execution with result saving
- Comprehensive logging and progress tracking

**Usage**:
```bash
# Example 1: Using the convenience function
uv run examples/advisor_usage.py example_1

# Example 2: Using full configuration objects
uv run examples/advisor_usage.py example_2

# Example 3: Using dictionary configuration
uv run examples/advisor_usage.py example_3
```

**Requirements**:
- MLflow server running on `localhost:5000`
- Valid run_id with Training and Deepchecks artifacts
- AI provider credentials for query execution
- DeepSight Advisor components installed

**Key Components**:
- `DeepSightAdvisor`: Main orchestrator class
- `AdvisorConfig`: Configuration management
- `MLflowConfig`: MLflow integration settings
- `run_analysis()`: Convenience function for quick analysis
- `setup_logging()`: Centralized logging system

**Configuration Examples**:

**Simple Usage** (Example 1):
```python
result = run_analysis(
    run_id="07c04cc42fd9461e98f7eb0bf42444fb",
    tracking_uri="http://localhost:5000"
)
```

**Full Configuration** (Example 2):
```python
config = AdvisorConfig(
    mlflow=MLflowConfig(
        tracking_uri="http://localhost:5000",
        run_id="07c04cc42fd9461e98f7eb0bf42444fb"
    )
)
advisor = DeepSightAdvisor(config)
result = advisor.run_analysis()
```

**Dictionary Configuration** (Example 3):
```python
config_dict = {
    "mlflow": {
        "tracking_uri": "http://localhost:5000",
        "run_id": "07c04cc42fd9461e98f7eb0bf42444fb"
    },
    "output": {
        "output_dir": "test_output",
        "format": "txt"
    }
}
advisor = DeepSightAdvisor(config_dict)
result = advisor.run_analysis()
```

**What the Advisor Does**:
1. **Artifact Loading**: Automatically loads Training and Deepchecks artifacts from MLflow
2. **Query Generation**: Creates intelligent prompts from loaded artifacts
3. **AI Analysis**: Executes queries using configured AI IntelligenceProviders
4. **Result Saving**: Saves prompts, responses, and formatted results to specified directories
5. **Progress Tracking**: Provides detailed logging throughout the entire process

**Output**:
- Timestamped log messages showing progress
- Analysis results saved to `advisor_output/` directory (or custom output directory)
- Structured result objects with execution metadata
- Success/failure status and performance metrics

## Getting Started

### Prerequisites

1. **Install DeepSight**:
   ```bash
   pip install deepsight
   ```

2. **Start MLflow Server**:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

3. **Prepare Data**:
   - Ensure required datasets are available
   - For training examples, prepare embeddings if needed

### Configuration

Each example uses consistent configuration patterns:

- **Logging**: Structured logging with INFO level
- **Error Handling**: Comprehensive try-catch blocks with detailed error messages
- **Configuration**: Clear, documented configuration objects
- **Paths**: Consistent use of `Path` objects for file operations

### Running Examples

1. **Start with Training**: Begin with `run_training.py` to generate artifacts
2. **Validate Data**: Use `run_suite_of_checks.py` to validate your data
3. **Manage Artifacts**: Use `run_artifact_manager.py` to explore artifacts
4. **Generate Insights**: Use `build_prompts.py` to get AI-powered analysis
5. **Complete Analysis**: Use `advisor_usage.py` for end-to-end automated analysis



## Troubleshooting

### Common Issues

1. **MLflow Connection Errors**:
   - Ensure MLflow server is running
   - Check tracking URI configuration
   - Verify network connectivity

2. **Import Errors**:
   - Ensure DeepSight is properly installed
   - Check Python path configuration
   - Verify all dependencies are installed

3. **Dataset Loading Issues**:
   - Check dataset paths and availability
   - Verify embedding files exist (if using pre-computed embeddings)
   - Ensure sufficient disk space

4. **Memory Issues**:
   - Reduce batch sizes
   - Use smaller datasets for testing
   - Enable mixed precision training

### Getting Help

- Check the logs for detailed error messages
- Verify all requirements are met
- Ensure MLflow server is accessible
- Check dataset and artifact availability

## Next Steps

After running these examples:

1. **Customize Configurations**: Modify parameters for your specific use case
2. **Add Your Data**: Integrate your own datasets and models
3. **Extend Functionality**: Build upon the examples for your specific needs
4. **Explore Advanced Features**: Dive deeper into DeepSight's capabilities

## Contributing

When adding new examples:

1. Follow the established patterns for imports, logging, and error handling
2. Include comprehensive docstrings and comments
3. Add clear usage instructions and requirements
4. Test examples thoroughly before submitting
5. Update this README with new example information
