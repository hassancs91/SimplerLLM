from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_providers.llm_response_models import LLMFullResponse
from SimplerLLM.utils.custom_verbose import verbose_print

class LLMConfig(BaseModel):
    """Configuration for a single LLM instance"""
    provider: str  # Name of the provider from LLMProvider enum
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    api_key: Optional[str] = None

class Message(BaseModel):
    """A single message in the conversation"""
    role: str  # "system", "user", or "assistant"
    content: str

class MultiLLMRequest(BaseModel):
    """Request model for the multi-LLM chat endpoint"""
    # List of previous messages in the conversation
    history: List[Message] = Field(default_factory=list)
    
    # New user message
    message: str
    
    # System prompt (if not included in history)
    system_prompt: Optional[str] = "You are a helpful AI assistant"
    
    # LLM configurations - can be a single config or multiple
    llm_configs: Union[LLMConfig, Dict[str, LLMConfig]]
    
    # Maximum tokens to generate (with a cap of 2048)
    max_tokens: int = 300
    
    # Top-p parameter for sampling
    top_p: float = 1.0
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v > 2048:
            return 2048
        return v

class DynamicLLM:
    """
    A class that provides a stateless interface for chatting with multiple LLM models
    and dynamically switching between providers and models.
    """
    
    @staticmethod
    def chat_with_multiple_llms(
        request: MultiLLMRequest, 
        verbose: bool = False
    ) -> Dict[str, Union[str, LLMFullResponse]]:
        """
        Process a chat request with one or more LLM configurations.
        
        Args:
            request: The MultiLLMRequest containing all necessary parameters
            verbose: Whether to print verbose output
            
        Returns:
            Dict mapping LLM identifiers to their responses
        """
        if verbose:
            verbose_print("Processing multi-LLM chat request", "info")
            
        # Prepare the full message history including the new message
        full_history = list(request.history)  # Create a copy
        
        # Add system message if not already in history
        if not any(msg.role == "system" for msg in full_history):
            full_history.insert(0, Message(role="system", content=request.system_prompt))
        
        # Add the new user message
        full_history.append(Message(role="user", content=request.message))
        
        # Format messages for LLM processing
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in full_history]
        
        responses = {}
        
        # Handle both single and multiple LLM configurations
        configs = request.llm_configs
        if isinstance(configs, LLMConfig):
            # Single LLM configuration
            configs = {"default": configs}
        
        # Process each LLM configuration
        for llm_id, config in configs.items():
            try:
                if verbose:
                    verbose_print(f"Processing LLM configuration for {llm_id}", "info")
                    
                # Convert provider string to enum
                provider_enum = getattr(LLMProvider, config.provider)
                
                # Create LLM instance
                llm = LLM.create(
                    provider=provider_enum,
                    model_name=config.model,
                    temperature=config.temperature,
                    top_p=config.top_p or request.top_p,  # Use config-specific top_p if provided, otherwise use request-level top_p
                    api_key=config.api_key,
                    verbose=verbose
                )
                
                if verbose:
                    verbose_print(f"Created LLM instance for {config.provider} with model {config.model}", "debug")
                
                # Generate response
                response = llm.generate_response(
                    messages=formatted_messages,
                    max_tokens=request.max_tokens,
                    top_p=config.top_p or request.top_p  # Ensure top_p is passed to generate_response
                )
                
                responses[llm_id] = response
                
                if verbose:
                    verbose_print(f"Generated response for {llm_id}", "info")
                    
            except Exception as e:
                if verbose:
                    verbose_print(f"Error processing {llm_id}: {str(e)}", "error")
                responses[llm_id] = f"Error: {str(e)}"
        
        return responses
    
    @staticmethod
    async def chat_with_multiple_llms_async(
        request: MultiLLMRequest, 
        verbose: bool = False
    ) -> Dict[str, Union[str, LLMFullResponse]]:
        """
        Asynchronously process a chat request with one or more LLM configurations.
        
        Args:
            request: The MultiLLMRequest containing all necessary parameters
            verbose: Whether to print verbose output
            
        Returns:
            Dict mapping LLM identifiers to their responses
        """
        if verbose:
            verbose_print("Processing async multi-LLM chat request", "info")
            
        # Prepare the full message history including the new message
        full_history = list(request.history)  # Create a copy
        
        # Add system message if not already in history
        if not any(msg.role == "system" for msg in full_history):
            full_history.insert(0, Message(role="system", content=request.system_prompt))
        
        # Add the new user message
        full_history.append(Message(role="user", content=request.message))
        
        # Format messages for LLM processing
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in full_history]
        
        responses = {}
        
        # Handle both single and multiple LLM configurations
        configs = request.llm_configs
        if isinstance(configs, LLMConfig):
            # Single LLM configuration
            configs = {"default": configs}
        
        # Process each LLM configuration
        for llm_id, config in configs.items():
            try:
                if verbose:
                    verbose_print(f"Processing async LLM configuration for {llm_id}", "info")
                    
                # Convert provider string to enum
                provider_enum = getattr(LLMProvider, config.provider)
                
                # Create LLM instance
                llm = LLM.create(
                    provider=provider_enum,
                    model_name=config.model,
                    temperature=config.temperature,
                    top_p=config.top_p or request.top_p,
                    api_key=config.api_key,
                    verbose=verbose
                )
                
                if verbose:
                    verbose_print(f"Created LLM instance for {config.provider} with model {config.model}", "debug")
                
                # Generate response asynchronously
                response = await llm.generate_response_async(
                    messages=formatted_messages,
                    max_tokens=request.max_tokens,
                    top_p=config.top_p or request.top_p
                )
                
                responses[llm_id] = response
                
                if verbose:
                    verbose_print(f"Generated async response for {llm_id}", "info")
                    
            except Exception as e:
                if verbose:
                    verbose_print(f"Error processing {llm_id} asynchronously: {str(e)}", "error")
                responses[llm_id] = f"Error: {str(e)}"
        
        return responses
