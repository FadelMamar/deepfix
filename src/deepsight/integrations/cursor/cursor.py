"""Main Cursor CLI integration class."""

from typing import Optional, Dict, Any
from .config import CursorConfig
from .process import ProcessManager
from .errors import CursorResponseError


class Cursor:
    """Main integration class for Cursor CLI non-interactive mode."""
    
    def __init__(
        self,
        model: str = "auto",
        output_format: str = "text",
        timeout: int = 300,
        cli_path: str = "cursor-agent",
        working_directory: Optional[str] = None,
        **kwargs
    ):
        """Initialize Cursor integration.
        
        Args:
            model: AI model to use (e.g., "auto", "gpt-5", "sonnet-4", "opus-4.1")
            output_format: Output format ("text", "json", "markdown")
            timeout: Timeout in seconds for CLI operations
            cli_path: Path to cursor-agent executable
            working_directory: Working directory for CLI operations
            **kwargs: Additional arguments to pass to Cursor CLI
        """
        self.config = CursorConfig(
            model=model,
            output_format=output_format,
            timeout=timeout,
            cli_path=cli_path,
            working_directory=working_directory,
            additional_args=kwargs if kwargs else None
        )
        self.process_manager = ProcessManager(cli_path=cli_path)
    
    def query(self, prompt: str) -> str:
        """Send a query to Cursor CLI and return the response.
        
        Args:
            prompt: The prompt/query to send to Cursor
            
        Returns:
            The response from Cursor CLI
            
        Raises:
            CursorError: If there's an error with the CLI operation
            CursorResponseError: If the response indicates an error
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Get CLI arguments from config
        cli_args = self.config.to_cli_args()
        
        # Execute the command
        stdout, stderr, return_code = self.process_manager.execute(
            args=cli_args,
            prompt=prompt,
            timeout=self.config.timeout,
            working_directory=self.config.working_directory
        )
        
        # Check for errors
        if return_code != 0:
            error_msg = stderr.strip() if stderr.strip() else "Unknown error"
            raise CursorResponseError(f"Cursor CLI returned error: {error_msg}")
        
        # Return the response
        response = stdout.strip()
        if not response:
            raise CursorResponseError("Empty response from Cursor CLI")
        
        return response
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                # Add to additional_args if not a standard config parameter
                if self.config.additional_args is None:
                    self.config.additional_args = {}
                self.config.additional_args[key] = value
