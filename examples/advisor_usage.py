"""
Basic usage example for DeepSight Advisor.

This example demonstrates how to use the advisor with minimal configuration
to run a complete ML analysis pipeline.
"""

import fire
from deepsight.core.advisor import AdvisorConfig, MLflowConfig
from deepsight.core.advisor import DeepSightAdvisor, run_analysis
from deepsight.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def example_1():
    """Example 1: Using the convenience function"""
    logger.info("Example 1: Using the convenience function")
    result = run_analysis(
        run_id="07c04cc42fd9461e98f7eb0bf42444fb", tracking_uri="http://localhost:5000"
    )
    logger.info(f"Analysis completed: {result.success}")
    logger.info(f"Artifacts loaded: {len(result.artifacts_loaded)}")
    logger.info(f"Execution time: {result.execution_time:.2f}s")


def example_2():
    """Example 2: Using full configuration"""
    config = AdvisorConfig(
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5000",
            run_id="07c04cc42fd9461e98f7eb0bf42444fb",
        )
    )
    advisor = DeepSightAdvisor(config)
    result = advisor.run_analysis()
    logger.info(f"Analysis completed: {result.success}")
    logger.info(f"Summary: {advisor.get_summary(result)}")


def example_3():
    """Example 3: Using dictionary configuration"""
    config_dict = {
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "run_id": "07c04cc42fd9461e98f7eb0bf42444fb",
        },
        "output": {"output_dir": "test_output", "format": "txt"},
    }
    advisor = DeepSightAdvisor(config_dict)
    result = advisor.run_analysis()
    logger.info(f"Analysis completed: {result.success}")
    logger.info(f"Output files: {list(result.output_paths.keys())}")


if __name__ == "__main__":
    fire.Fire(
        {
            "example_1": example_1,
            "example_2": example_2,
            "example_3": example_3,
        }
    )
