"""Configuration management for Cursor CLI integration."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class CursorConfig(BaseModel):
    """Configuration for Cursor CLI integration."""
    
    model: str = Field(default="auto",description="Model to use for Cursor CLI integration")
    output_format: str = Field(default="text",description="Output format to use for Cursor CLI integration",
                          examples=["text", "json", "stream-json"])
    timeout: int = Field(default=300,description="Timeout to use for Cursor CLI integration")
    cli_path: str = Field(default="cursor-agent",description="Path to the Cursor CLI executable")
    working_directory: Optional[str] = Field(default=None,description="Working directory to use for Cursor CLI integration")
        
    def to_cli_args(self) -> list[str]:
        """Convert configuration to CLI arguments."""
        args = [
            self.cli_path,
            "-p",  # non-interactive mode
        ]
        
        # Add model if specified
        if self.model:
            args.extend(["--model", self.model])
        
        # Add output format if specified
        if self.output_format:
            args.extend(["--output-format", self.output_format])

        
        return args
