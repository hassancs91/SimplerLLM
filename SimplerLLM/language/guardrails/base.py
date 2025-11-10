"""
Base classes and protocols for the Guardrails system.

This module provides the core abstractions for implementing guardrails
in SimplerLLM. Guardrails can be applied before (input) or after (output)
LLM generation to enforce safety, quality, and compliance requirements.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class GuardrailAction(Enum):
    """Actions a guardrail can take when processing content."""

    ALLOW = "allow"           # Continue normally without modifications
    MODIFY = "modify"         # Modify content and continue processing
    BLOCK = "block"           # Block the request/response entirely
    WARN = "warn"             # Log warning but continue processing


@dataclass
class GuardrailResult:
    """
    Result of a guardrail execution.

    Attributes:
        action: The action to take (ALLOW, MODIFY, BLOCK, WARN)
        passed: Whether the guardrail validation passed
        message: Optional message explaining the result
        modified_content: Modified content if action is MODIFY
        metadata: Additional metadata about the guardrail execution
        guardrail_name: Name of the guardrail that produced this result
    """
    action: GuardrailAction
    passed: bool
    message: Optional[str] = None
    modified_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    guardrail_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "guardrail": self.guardrail_name,
            "action": self.action.value,
            "passed": self.passed,
            "message": self.message,
            "metadata": self.metadata
        }


class InputGuardrail(ABC):
    """
    Base class for input/prompt guardrails.

    Input guardrails are executed before the LLM call to validate
    and optionally modify the prompt, system prompt, or messages.

    Example use cases:
    - Inject safety instructions into system prompt
    - Block certain topics or keywords
    - Detect and handle PII in user input
    - Add formatting instructions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the guardrail.

        Args:
            config: Configuration dictionary for the guardrail
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.name = self.__class__.__name__

    @abstractmethod
    def validate(self,
                 prompt: str,
                 system_prompt: str,
                 messages: Optional[List[Dict]] = None,
                 **kwargs) -> GuardrailResult:
        """
        Validate and optionally modify input before LLM call.

        Args:
            prompt: The user prompt/message
            system_prompt: The system prompt
            messages: Optional list of conversation messages
            **kwargs: Additional context from the LLM call

        Returns:
            GuardrailResult with action and optional modifications
        """
        pass

    @abstractmethod
    async def validate_async(self,
                            prompt: str,
                            system_prompt: str,
                            messages: Optional[List[Dict]] = None,
                            **kwargs) -> GuardrailResult:
        """
        Async version of validate.

        Args:
            prompt: The user prompt/message
            system_prompt: The system prompt
            messages: Optional list of conversation messages
            **kwargs: Additional context from the LLM call

        Returns:
            GuardrailResult with action and optional modifications
        """
        pass


class OutputGuardrail(ABC):
    """
    Base class for output/response guardrails.

    Output guardrails are executed after the LLM call to validate
    and optionally modify the generated response.

    Example use cases:
    - Content safety and moderation
    - PII detection and redaction
    - Format validation (JSON, XML, etc.)
    - Length or quality constraints
    - Toxicity filtering
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the guardrail.

        Args:
            config: Configuration dictionary for the guardrail
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.name = self.__class__.__name__

    @abstractmethod
    def validate(self,
                 response: str,
                 original_prompt: str = "",
                 **kwargs) -> GuardrailResult:
        """
        Validate and optionally modify output after LLM call.

        Args:
            response: The generated response text
            original_prompt: The original user prompt for context
            **kwargs: Additional context from the LLM call

        Returns:
            GuardrailResult with action and optional modifications
        """
        pass

    @abstractmethod
    async def validate_async(self,
                            response: str,
                            original_prompt: str = "",
                            **kwargs) -> GuardrailResult:
        """
        Async version of validate.

        Args:
            response: The generated response text
            original_prompt: The original user prompt for context
            **kwargs: Additional context from the LLM call

        Returns:
            GuardrailResult with action and optional modifications
        """
        pass


class CompositeGuardrail:
    """
    Combines multiple guardrails into a single guardrail.

    Useful for organizing related guardrails or creating
    reusable guardrail configurations.
    """

    def __init__(self, guardrails: List[Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize composite guardrail.

        Args:
            guardrails: List of InputGuardrail or OutputGuardrail instances
            config: Configuration for the composite
        """
        self.guardrails = guardrails
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.fail_fast = self.config.get("fail_fast", True)
        self.name = self.config.get("name", "CompositeGuardrail")

    def validate_all(self, *args, **kwargs) -> List[GuardrailResult]:
        """
        Execute all guardrails and return results.

        Returns:
            List of GuardrailResult from each guardrail
        """
        results = []
        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue

            result = guardrail.validate(*args, **kwargs)
            results.append(result)

            if self.fail_fast and result.action == GuardrailAction.BLOCK:
                break

        return results

    async def validate_all_async(self, *args, **kwargs) -> List[GuardrailResult]:
        """
        Execute all guardrails asynchronously and return results.

        Returns:
            List of GuardrailResult from each guardrail
        """
        results = []
        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue

            result = await guardrail.validate_async(*args, **kwargs)
            results.append(result)

            if self.fail_fast and result.action == GuardrailAction.BLOCK:
                break

        return results
