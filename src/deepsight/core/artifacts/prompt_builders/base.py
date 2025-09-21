"""
Base prompt builder for PromptBuilder.

This module provides the abstract base class for all prompt builders.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, List
from ..datamodel import DeepchecksArtifacts, TrainingArtifacts


class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""

    @abstractmethod
    def can_build(self, artifact_type: str) -> bool:
        """Check if this builder can handle the artifact type.

        Args:
            artifact_type: The type of artifact (e.g., 'DeepchecksArtifacts', 'TrainingArtifacts')

        Returns:
            True if this builder can handle the artifact type, False otherwise
        """
        pass

    @abstractmethod
    def build_prompt(
        self,
        artifact: Union[DeepchecksArtifacts, TrainingArtifacts],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt from artifact.

        Args:
            artifact: The artifact to build a prompt for
            context: Additional context for the prompt

        Returns:
            A structured prompt string ready for LLM completion
        """
        pass

    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context dictionary into a readable string.

        Args:
            context: Context dictionary to format

        Returns:
            Formatted context string
        """
        if not context:
            return ""

        context_parts = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                context_parts.append(f"- {key}: {value}")
            else:
                context_parts.append(f"- {key}: {value}")

        return "\n".join(context_parts)

    def _get_query_instructions(self) -> List[str]:
        """Get query-specific instructions based on query type."""
        base_instructions = [
            "\nPlease provide:",
            "1. Summary of data quality issues",
            "2. Recommendations for model improvement",
            "3. Next steps for validation",
        ]

        base_instructions.extend(
            [
                "4. Specific overfitting indicators if found",
                "5. Regularization recommendations if necessary",
            ]
        )
        base_instructions.extend(
            ["6. Performance degradation analysis", "7. Model optimization suggestions"]
        )
        base_instructions.extend(
            [
                "8. Data preprocessing recommendations if necessary",
                "9. Data collection improvements if necessary",
            ]
        )

        return base_instructions
