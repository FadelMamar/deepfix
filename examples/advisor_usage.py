"""
Basic usage example for DeepSight Advisor.

This example demonstrates how to use the advisor with minimal configuration
to run a complete ML analysis pipeline.
"""

import fire
import os
from deepsight.core.config import MLflowConfig,ArtifactConfig, OutputConfig, PromptConfig
from deepsight.core.query import IntelligenceConfig,CursorConfig,LLMConfig
from deepsight.core.advisor import DeepSightAdvisor, run_analysis,AdvisorConfig
from deepsight.utils.logging import setup_logging, get_logger
from dotenv import load_dotenv

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def example_1(run_id:str):
    """Example 1: Using the convenience function"""
    logger.info("Example 1: Using the convenience function")
    result = run_analysis(
        run_id=run_id, tracking_uri="http://localhost:5000"
    )
    logger.info(f"Analysis completed: {result.success}")
    logger.info(f"Artifacts loaded: {len(result.artifacts_loaded)}")
    logger.info(f"Execution time: {result.execution_time:.2f}s")


def example_2(env_file:str, run_id:str, dataset_name:str):
    """Example 2: Using full configuration"""
    load_dotenv(env_file,override=True)

    api_key = os.getenv("LLM_API_KEY")  # Replace with your actual API key
    base_url = "https://openrouter.ai/api/v1"  # Replace with your actual base URL
    model_name = "openai/x-ai/grok-4-fast:free"  # Replace with your desired model name
    temperature = 0.7
    max_tokens = 1024

    config = AdvisorConfig(
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5000",
            download_dir="mlflow_downloads",
        ),
        prompt=PromptConfig(custom_instructions=None),
        artifacts=ArtifactConfig(load_checks=True, 
                                 load_training=True,
                                 cache_enabled=True,
                                 load_model_checkpoint=True,
                                 load_dataset_metadata=True,
                                 dataset_name=dataset_name,
                                 sqlite_path="tmp/artifacts.db"
                                ),
        intelligence=IntelligenceConfig(provider_name="llm",
                                        timeout=30, 
                                        llm_config=LLMConfig(api_key=api_key,
                                                             base_url=base_url,
                                                             model_name=model_name,
                                                             temperature=temperature,
                                                             max_tokens=max_tokens
                                                            ),
                                        cursor_config=CursorConfig(model="auto")
        ),
        output=OutputConfig(output_dir="advisor_output", 
                            save_prompt=True, 
                            save_response=True,
                            format="txt"
                        ),
    )
    advisor = DeepSightAdvisor(config)
    result = advisor.run_analysis(run_id=run_id)
    if result is not None:
        logger.info(f"Summary: {result.get_summary()}")


if __name__ == "__main__":
    fire.Fire(
        {
            "example_1": example_1,
            "example_2": example_2,
        }
    )
