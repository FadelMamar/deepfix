from typing import Dict, Any, Optional, Union
import time

from .models import IntelligenceResponse, ProviderType, IntelligenceProviderError, Providers
from .providers.llm.dspy import DspyLLMProvider
from .providers.coding_agent.cursor import CursorAgentProvider


class IntelligenceClient:
    """Synchronous multi-provider client for LLMs and coding agents.

    Accepts a prompt string and executes it against the selected provider.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.providers: Dict[Providers, Any] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        # LLM via DSPy wrapper
        llm_cfg = (self.config.get(ProviderType.LLM.value) or {}).get("dspy") or {}
        if llm_cfg.get("enabled", True):
            self.providers[Providers.DSPY] = DspyLLMProvider(config=llm_cfg)

        # Coding Agent(s)
        agent_cfg = (self.config.get(ProviderType.CODING_AGENT.value) or {}).get("cursor") or {}
        if agent_cfg.get("enabled", True):
            self.providers[Providers.CURSOR] = CursorAgentProvider(config=agent_cfg)

    def execute_query(
        self,
        prompt: str,
        provider_type: Union[ProviderType, str] = ProviderType.CODING_AGENT,
        provider_name: Optional[Union[str, Providers]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntelligenceResponse:
        provider_name = Providers(provider_name) if isinstance(provider_name, str) else provider_name
        provider_type = ProviderType(provider_type) if isinstance(provider_type, str) else provider_type
        provider = self._select_provider(provider_type, provider_name)
        return provider.execute(prompt, context or {})

    def _select_provider(self, provider_type: ProviderType, provider_name: Optional[Providers]=None) -> CursorAgentProvider | DspyLLMProvider:
        if provider_name:
            if provider_name in self.providers:
                return self.providers[provider_name]
            raise IntelligenceProviderError(f"Provider '{provider_name}' not found")

        # Auto-select based on type
        for name, provider in self.providers.items():
            if getattr(provider, "provider_type", None) == provider_type:
                return provider

        raise IntelligenceProviderError(f"No provider available for type: {provider_type}")


