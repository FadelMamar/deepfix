from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    LLM = "llm"
    CODING_AGENT = "coding_agent"


class Providers(Enum):
    CURSOR = "cursor_agent"
    DSPY = "dspy_llm"


class Capabilities(Enum):
    TEXT_GENERATION = "text_generation"
    CODE_ANALYSIS = "code_analysis"
    REASONING = "reasoning"
    ML_INSIGHTS = "ml_insights"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    IMAGE_UNDERSTANDING = "image_understanding"


class IntelligenceResponse(BaseModel):
    content: str
    provider: str
    provider_type: ProviderType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[int] = None


class IntelligenceProviderError(RuntimeError):
    pass
