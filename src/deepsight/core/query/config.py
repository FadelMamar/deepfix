"""
Configuration management for QueryGenerator.
"""

from typing import Dict, Any
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from enum import Enum


class QueryType(Enum):
    """Types of queries that can be generated."""
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    DEBUG = "debug"
    OTHER = "other"
    

class QueryGeneratorConfig(BaseModel):
    """Configuration for QueryGenerator."""
    
    # Prompt building configuration
    prompt_builders: Dict[str, Any] = Field(default_factory=dict)
    max_prompt_length: int = Field(default=4000, ge=100, le=10000)
    detail_level: str = Field(default="comprehensive")
    
    # LLM client configuration
    llm: Dict[str, Any] = Field(default_factory=dict)
    
    # File loading configuration
    file_loading: Dict[str, Any] = Field(default_factory=dict)
    
    # Logging configuration
    logging: Dict[str, Any] = Field(default_factory=dict)

    query_type: QueryType = Field(default=QueryType.ANALYSIS)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "QueryGeneratorConfig":
        """Create a QueryGeneratorConfig from a dictionary."""
        return cls(**config)

    @classmethod
    def from_file(cls, file_path: str) -> "QueryGeneratorConfig":
        """Create a QueryGeneratorConfig from a file."""
        return cls.from_dict(OmegaConf.load(file_path))

