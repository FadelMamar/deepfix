from typing import Dict, Any, Optional, List, Union
import time
import traceback

from ...models import (
    IntelligenceResponse,
    ProviderType,
    IntelligenceProviderError,
    Providers,
    Capabilities,
)
from ..base import BaseProvider

from deepsight.integrations import Cursor
from deepsight.integrations.cursor import CursorConfig


class CursorAgentProvider(BaseProvider):
    def __init__(self, config: Union[Dict[str, Any], CursorConfig]):
        self.config = CursorConfig(**config) if isinstance(config, dict) else config
        self.provider_type = ProviderType.CODING_AGENT
        self.agent = Cursor(**self.config.model_dump())

    def execute(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> IntelligenceResponse:
        start = time.time()
        try:
            enhanced_prompt = self._enhance_prompt_for_coding(prompt, context or {})
            resp = self.agent.query(prompt=enhanced_prompt)
            latency_ms = int((time.time() - start) * 1000)
            return IntelligenceResponse(
                content=resp,
                provider=Providers.CURSOR,
                provider_type=ProviderType.CODING_AGENT,
                latency_ms=latency_ms,
            )
        except Exception:
            raise IntelligenceProviderError(
                f"Cursor agent failed: {traceback.format_exc()}"
            )

    def _enhance_prompt_for_coding(self, prompt: str, context: Dict[str, Any]) -> str:
        parts = [
            "You are an expert MLOps engineer and data scientist.",
            "Focus on providing actionable code solutions and implementation guidance.",
            "",
            prompt,
        ]
        if context.get("code_context"):
            parts.insert(-1, f"\nCode context: {context['code_context']}")
        return "\n".join(parts)

    def get_capabilities(self) -> List[Capabilities]:
        return [
            Capabilities.CODE_GENERATION,
            Capabilities.DEBUGGING,
            Capabilities.REASONING,
            Capabilities.TEXT_GENERATION,
        ]
