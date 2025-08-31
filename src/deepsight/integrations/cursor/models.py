"""
Data models for Cursor API integration.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    CREATING = "CREATING"
    RUNNING = "RUNNING" 
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ImageData(BaseModel):
    """Image data for prompts."""
    data: str = Field(..., description="Base64 encoded image data")
    dimension: Dict[str, int] = Field(..., description="Image dimensions (width, height)")


class Prompt(BaseModel):
    """Prompt data for agent requests."""
    text: str = Field(..., description="Text prompt for the agent")
    images: Optional[List[ImageData]] = Field(default=None, description="Optional image attachments")


class SourceConfig(BaseModel):
    """Source repository configuration."""
    repository: str = Field(..., description="GitHub repository URL")
    ref: str = Field(default="main", description="Branch or ref to use")


class TargetConfig(BaseModel):
    """Target configuration for agent results."""
    branch_name: str = Field(..., description="Target branch name")
    url: str = Field(..., description="Agent URL")
    auto_create_pr: bool = Field(default=False, description="Whether to auto-create PR")


class AgentRequest(BaseModel):
    """Request model for creating an agent."""
    prompt: Prompt = Field(..., description="Prompt configuration")
    source: SourceConfig = Field(..., description="Source repository configuration")


class AgentResponse(BaseModel):
    """Response model for agent operations."""
    id: str = Field(..., description="Agent unique identifier")
    name: str = Field(..., description="Agent name")
    status: AgentStatus = Field(..., description="Current agent status")
    source: SourceConfig = Field(..., description="Source configuration")
    target: TargetConfig = Field(..., description="Target configuration")
    created_at: datetime = Field(..., description="Creation timestamp")


class AgentStatusResponse(BaseModel):
    """Response model for agent status queries."""
    id: str = Field(..., description="Agent unique identifier")
    status: AgentStatus = Field(..., description="Current agent status")
    progress: Optional[Dict[str, Any]] = Field(default=None, description="Progress information")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    result_url: Optional[str] = Field(default=None, description="URL to agent results")


class LLMQuery(BaseModel):
    """Model for LLM query requests."""
    query: str = Field(..., description="The query text")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    images: Optional[List[ImageData]] = Field(default=None, description="Optional image attachments")
    repository_context: Optional[SourceConfig] = Field(default=None, description="Repository context")


class LLMResponse(BaseModel):
    """Model for LLM query responses."""
    query_id: str = Field(..., description="Unique query identifier")
    agent_id: str = Field(..., description="Associated agent ID")
    response: str = Field(..., description="LLM response text")
    reasoning_steps: Optional[List[str]] = Field(default=None, description="Step-by-step reasoning")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
