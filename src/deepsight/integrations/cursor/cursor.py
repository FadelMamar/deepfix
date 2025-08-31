"""
Main Cursor API client implementation.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
import httpx
from datetime import datetime

from .config import CursorConfig
from .models import (
    AgentRequest, AgentResponse, AgentStatusResponse, LLMQuery, LLMResponse,
    Prompt, SourceConfig, TargetConfig, AgentStatus, ImageData
)
from .exceptions import (
    CursorAPIError, CursorAuthError, CursorValidationError,
    CursorRateLimitError, CursorAgentNotFoundError
)


class CursorAgent:
    """Main client for interacting with the Cursor API."""
    
    def __init__(self, config: Optional[CursorConfig] = None):
        """Initialize the Cursor API client.
        
        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or CursorConfig.from_env()
        self.logger = self._setup_logging()
        self._client: Optional[httpx.AsyncClient] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            CursorAPIError: For API errors
        """
        url = f"{endpoint}"
        retry_count = 0
        
        while retry_count <= self.config.max_retries:
            try:
                if self.config.log_requests:
                    self.logger.debug(f"Making {method} request to {url}")
                
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                )
                
                if self.config.log_requests:
                    self.logger.debug(f"Response status: {response.status_code}")
                
                if response.is_success:
                    return response.json()
                
                # Handle specific error cases
                if response.status_code == 401:
                    raise CursorAuthError(
                        "Authentication failed. Check your API token.",
                        status_code=response.status_code,
                        response_data=response.json() if response.content else None
                    )
                elif response.status_code == 403:
                    raise CursorAuthError(
                        "Access forbidden. Check your permissions.",
                        status_code=response.status_code,
                        response_data=response.json() if response.content else None
                    )
                elif response.status_code == 429:
                    if retry_count < self.config.max_retries:
                        delay = self.config.retry_delay * (2 ** retry_count)
                        self.logger.warning(f"Rate limited, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        retry_count += 1
                        continue
                    else:
                        raise CursorRateLimitError(
                            "Rate limit exceeded",
                            status_code=response.status_code,
                            response_data=response.json() if response.content else None
                        )
                else:
                    raise CursorAPIError(
                        f"API request failed: {response.status_code}",
                        status_code=response.status_code,
                        response_data=response.json() if response.content else None
                    )
                    
            except httpx.RequestError as e:
                if retry_count < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry_count)
                    self.logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue
                else:
                    raise CursorAPIError(f"Request failed after {self.config.max_retries} retries: {e}")
    
    async def create_agent(
        self,
        prompt: str,
        repository_url: Optional[str] = None,
        branch: Optional[str] = None,
        images: Optional[List[ImageData]] = None,
    ) -> AgentResponse:
        """Create a new agent.
        
        Args:
            prompt: Text prompt for the agent
            repository_url: GitHub repository URL (uses default if None)
            branch: Branch/ref to use (uses default if None)
            images: Optional image attachments
            
        Returns:
            Agent response data
        """
        if not prompt.strip():
            raise CursorValidationError("Prompt cannot be empty")
        
        if not repository_url:
            raise CursorValidationError("Repository URL is required")
        
        request_data = AgentRequest(
            prompt=Prompt(text=prompt, images=images),
            source=SourceConfig(
                repository=repository_url,
                ref=branch or "main"
            )
        )
        
        response_data = await self._make_request(
            method="POST",
            endpoint="/agents",
            data=request_data.model_dump(exclude_none=True)
        )
        
        return AgentResponse(**response_data)
    
    async def get_agent_status(self, agent_id: str) -> AgentStatusResponse:
        """Get agent status.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent status data
        """
        if not agent_id:
            raise CursorValidationError("Agent ID cannot be empty")
        
        try:
            response_data = await self._make_request(
                method="GET",
                endpoint=f"/agents/{agent_id}"
            )
            return AgentStatusResponse(**response_data)
        except CursorAPIError as e:
            if e.status_code == 404:
                raise CursorAgentNotFoundError(f"Agent {agent_id} not found")
            raise
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if successful
        """
        if not agent_id:
            raise CursorValidationError("Agent ID cannot be empty")
        
        try:
            await self._make_request(
                method="DELETE",
                endpoint=f"/agents/{agent_id}"
            )
            return True
        except CursorAPIError as e:
            if e.status_code == 404:
                raise CursorAgentNotFoundError(f"Agent {agent_id} not found")
            raise
    
    async def wait_for_completion(
        self,
        agent_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> AgentStatusResponse:
        """Wait for agent to complete.
        
        Args:
            agent_id: Agent identifier
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds
            
        Returns:
            Final agent status
        """
        start_time = time.time()
        
        while True:
            status = await self.get_agent_status(agent_id)
            
            if status.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
                return status
            
            if timeout and (time.time() - start_time) > timeout:
                raise CursorAPIError(f"Timeout waiting for agent {agent_id} to complete")
            
            await asyncio.sleep(poll_interval)
    
    async def add_follow_up(
        self,
        agent_id: str,
        follow_up_prompt: str,
        images: Optional[List[ImageData]] = None,
    ) -> bool:
        """Add a follow-up to an existing agent.
        
        Args:
            agent_id: Agent identifier
            follow_up_prompt: Additional prompt text
            images: Optional image attachments
            
        Returns:
            True if successful
        """
        if not agent_id:
            raise CursorValidationError("Agent ID cannot be empty")
        if not follow_up_prompt.strip():
            raise CursorValidationError("Follow-up prompt cannot be empty")
        
        follow_up_data = {
            "prompt": {
                "text": follow_up_prompt,
                "images": [img.model_dump() for img in images] if images else None
            }
        }
        
        try:
            await self._make_request(
                method="POST",
                endpoint=f"/agents/{agent_id}/follow-up",
                data=follow_up_data
            )
            return True
        except CursorAPIError as e:
            if e.status_code == 404:
                raise CursorAgentNotFoundError(f"Agent {agent_id} not found")
            raise


class LLMQueryHandler:
    """Handler for LLM queries using Cursor agents."""
    
    def __init__(self, cursor_client: CursorAgent):
        """Initialize query handler.
        
        Args:
            cursor_client: Cursor API client instance
        """
        self.client = cursor_client
        self.logger = cursor_client.logger
    
    async def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        images: Optional[List[ImageData]] = None,
        repository_url: Optional[str] = None,
        branch: Optional[str] = None,
        wait_for_completion: bool = True,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Execute an LLM query.
        
        Args:
            query: The query text
            context: Additional context data
            images: Optional image attachments
            repository_url: Repository context
            branch: Branch context
            wait_for_completion: Whether to wait for completion
            timeout: Maximum wait time
            
        Returns:
            LLM response
        """
        query_id = str(uuid.uuid4())
        
        # Create agent with the query
        prompt = self._build_prompt(query, context)
        agent = await self.client.create_agent(
            prompt=prompt,
            repository_url=repository_url,
            branch=branch,
            images=images,
        )
        
        self.logger.info(f"Created agent {agent.id} for query {query_id}")
        
        if wait_for_completion:
            final_status = await self.client.wait_for_completion(
                agent.id, timeout=timeout
            )
            
            if final_status.status == AgentStatus.FAILED:
                raise CursorAPIError(f"Agent failed: {final_status.error}")
            
            # Extract response (this would need to be implemented based on actual API)
            response_text = f"Agent {agent.id} completed successfully"
            
        else:
            response_text = f"Agent {agent.id} created and running"
        
        return LLMResponse(
            query_id=query_id,
            agent_id=agent.id,
            response=response_text,
            reasoning_steps=None,  # TODO: Extract from agent results
            confidence=None,  # TODO: Extract from agent results
            metadata={
                "agent_status": agent.status.value,
                "created_at": agent.created_at.isoformat(),
                "context": context,
            }
        )
    
    def _build_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt from query and context.
        
        Args:
            query: Base query
            context: Additional context
            
        Returns:
            Formatted prompt
        """
        prompt_parts = [query]
        
        if context:
            prompt_parts.append("\nAdditional context:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
        
        return "\n".join(prompt_parts)
