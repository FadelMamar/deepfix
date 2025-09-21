"""
Main PromptBuilder class for orchestrating prompt creation from existing Pydantic models.
"""

from typing import Optional, Dict, Any, List, Union
import traceback

from ....utils.logging import get_logger
from ...artifacts.datamodel import DeepchecksArtifacts, TrainingArtifacts
from . import DeepchecksPromptBuilder, TrainingPromptBuilder
from ....utils.exceptions import PromptBuilderError


class PromptBuilder:
    """Main class for orchestrating prompt creation from existing Pydantic models."""

    def __init__(
        self,
    ):
        """Initialize the PromptBuilder.

        Args:
            config: Optional configuration for the PromptBuilder
            config_path: Optional path to configuration file
        """
        self.logger = get_logger(__name__)
        self.prompt_builders = self._initialize_prompt_builders()

        self.logger.info("PromptBuilder initialized successfully")

    def build_prompt(
        self,
        artifacts: List[Union[DeepchecksArtifacts, TrainingArtifacts]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt based on artifact types."""
        try:
            assert len(artifacts) > 0, "No artifacts provided for prompt generation"
            prompt_parts = []

            # Group artifacts by type
            artifact_groups = {}
            for artifact in artifacts:
                artifact_type = type(artifact).__name__
                if artifact_type not in artifact_groups:
                    artifact_groups[artifact_type] = []
                artifact_groups[artifact_type].append(artifact)

            # Build prompts for each artifact type
            for artifact_type, artifact_list in artifact_groups.items():
                builder = self._get_prompt_builder(artifact_type)
                if builder:
                    for artifact in artifact_list:
                        try:
                            prompt = builder.build_prompt(artifact, context)
                            prompt_parts.append(prompt)
                        except Exception:
                            raise PromptBuilderError(
                                artifact_type,
                                f"Failed to build prompt: {traceback.format_exc()}",
                            )
                else:
                    raise PromptBuilderError(
                        artifact_type, "No prompt builder found for artifact type"
                    )

            if not prompt_parts:
                raise PromptBuilderError(
                    "all", "No prompts could be built from artifacts"
                )

            # Combine all prompts
            full_prompt = "\n\n".join(prompt_parts) + "\n\n" + self._get_instruction()

            return full_prompt

        except Exception:
            raise PromptBuilderError(
                "unknown",
                f"Unexpected error during prompt building: {traceback.format_exc()}",
            )

    def _get_instruction(self) -> str:
        return """ONLY ANSWER BASED ON THE PROVIDED INFORMATION. DO NOT MAKE UP ANYTHING or READ ANY OTHER FILES.
                 DO NOT CREATE ANY NEW FILES OR DIRECTORIES.
                 DO NOT EDIT ANY FILES.
                 DO NOT DELETE ANY FILES.
                 DO NOT RENAME ANY FILES.
                 DO NOT MOVE ANY FILES.
                 DO NOT COPY ANY FILES.
                 DO NOT PASTE ANY FILES.
                 ANSWER IN PLAIN TEXT.              
        """

    def _get_prompt_builder(
        self, artifact_type: str
    ) -> Optional[Union[DeepchecksPromptBuilder, TrainingPromptBuilder]]:
        """Get appropriate prompt builder for the given artifact type."""
        for builder in self.prompt_builders:
            if builder.can_build(artifact_type):
                return builder
        return None

    def _initialize_prompt_builders(
        self,
    ) -> List[Union[DeepchecksPromptBuilder, TrainingPromptBuilder]]:
        """Initialize prompt builders based on configuration."""
        builders = [DeepchecksPromptBuilder(), TrainingPromptBuilder()]
        return builders
