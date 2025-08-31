"""
Configuration management for Cursor API integration using Pydantic.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
import os


class CursorConfig(BaseModel):
    """Configuration for Cursor API client."""
    
    # API Configuration
    api_token: str = Field(..., description="Cursor API authentication token")
    base_url: str = Field(
        default="https://api.cursor.com/v0",
        description="Base URL for Cursor API"
    )
            
    # Request Configuration
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_requests: bool = Field(
        default=True,
        description="Whether to log API requests/responses"
    )
    
    class Config:
        env_prefix = "CURSOR_"
        
    @field_validator("api_token")
    def validate_api_token(cls, v):
        if not v or not v.strip():
            raise ValueError("API token cannot be empty")
        return v.strip()
    
    @field_validator("base_url")
    def validate_base_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v.rstrip("/")
    
    @field_validator("timeout")
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @field_validator("max_retries")
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError("Max retries cannot be negative")
        return v
    
    @classmethod
    def from_env(cls) -> "CursorConfig":
        """Create configuration from environment variables."""
        return cls(
            api_token=os.getenv("CURSOR_API_TOKEN", ""),
            base_url=os.getenv("CURSOR_BASE_URL", "https://api.cursor.com/v0"),
            timeout=int(os.getenv("CURSOR_TIMEOUT", "30")),
            max_retries=int(os.getenv("CURSOR_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("CURSOR_RETRY_DELAY", "1.0")),
            log_level=os.getenv("CURSOR_LOG_LEVEL", "INFO"),
            log_requests=os.getenv("CURSOR_LOG_REQUESTS", "true").lower() == "true",
        )
