"""
Cursor API Integration for DeepSight

This module provides a comprehensive wrapper around the Cursor Background Agent API
to handle LLM queries and reasoning for the DeepSight project.
"""

from .cursor import CursorAgent, LLMQueryHandler
from .config import CursorConfig
from .exceptions import (
    CursorAPIError, CursorAuthError, CursorValidationError,
    CursorRateLimitError, CursorAgentNotFoundError
)
from .models import (
    AgentStatus, ImageData, Prompt, SourceConfig, TargetConfig,
    AgentRequest, AgentResponse, AgentStatusResponse, LLMQuery, LLMResponse
)
from .utils import encode_image_to_base64, validate_image_data, create_prompt_template

__all__ = [
    "CursorAgent",
    "CursorConfig", 
    "LLMQueryHandler",
    "CursorAPIError",
    "CursorAuthError", 
    "CursorValidationError",
    "CursorRateLimitError",
    "CursorAgentNotFoundError",
    "AgentStatus",
    "ImageData",
    "Prompt",
    "SourceConfig",
    "TargetConfig",
    "AgentRequest",
    "AgentResponse", 
    "AgentStatusResponse",
    "LLMQuery",
    "LLMResponse",
    "encode_image_to_base64",
    "validate_image_data",
    "create_prompt_template",
]
