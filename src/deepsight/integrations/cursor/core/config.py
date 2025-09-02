"""Configuration management for Cursor CLI wrapper."""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for Cursor CLI wrapper."""
    
    model: str = "auto"
    output_format: str = "text"
    timeout: int = 300
    cli_path: str = "cursor-agent"
    working_directory: Optional[str] = None
    additional_args: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.output_format not in ["text", "json", "markdown"]:
            raise ValueError("Output format must be one of: text, json, markdown")
    
    def to_cli_args(self) -> list[str]:
        """Convert configuration to CLI arguments."""
        args = [
            self.cli_path,
            "-p",  # prompt flag
        ]
        
        # Add model if specified
        if self.model:
            args.extend(["--model", self.model])
        
        # Add output format if specified
        if self.output_format:
            args.extend(["--output-format", self.output_format])
        
        # Add additional arguments
        if self.additional_args:
            for key, value in self.additional_args.items():
                if isinstance(value, bool):
                    if value:
                        args.append(f"--{key}")
                else:
                    args.extend([f"--{key}", str(value)])
        
        return args
