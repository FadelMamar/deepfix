"""
Basic usage example for DeepSight Advisor.

This example demonstrates how to use the advisor with minimal configuration
to run a complete ML analysis pipeline.
"""

import fire
from deepsight.core.config import MLflowConfig,ArtifactConfig
from deepsight.core.query import IntelligenceConfig,CursorConfig,LLMConfig
from deepsight.core.advisor import DeepSightAdvisor, run_analysis,AdvisorConfig
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

    api_key = "your_api_key"  # Replace with your actual API key
    base_url = "https://api.openai.com/v1"  # Replace with your actual base URL
    model_name = "openai/gpt-4"  # Replace with your desired model name
    temperature = 0.7
    max_tokens = 1024

    config = AdvisorConfig(
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5000",
            run_id="07c04cc42fd9461e98f7eb0bf42444fb",
        ),
        artifacts=ArtifactConfig(),
        intelligence=IntelligenceConfig(provider_name="llm",
                                        timeout=30, 
                                        llm_config=LLMConfig(api_key=api_key,
                                                             base_url=base_url,
                                                             model_name=model_name,
                                                             temperature=temperature,
                                                             max_tokens=max_tokens
                                                            ),
                                        cursor_config=CursorConfig(model="auto")
        )
    )
    advisor = DeepSightAdvisor(config)
    result = advisor.run_analysis()
    logger.info(f"Analysis completed: {result.success}")
    logger.info(f"Summary: {advisor.get_summary(result)}")


if __name__ == "__main__":
    fire.Fire(
        {
            "example_1": example_1,
            "example_2": example_2,
        }
    )
