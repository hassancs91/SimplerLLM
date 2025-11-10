"""
SimplerLLM Guardrails Module

This module provides a comprehensive guardrails system for SimplerLLM that adds
safety, quality, and compliance checks to LLM interactions.

Guardrails can be applied before (input) or after (output) LLM generation to:
- Enforce safety and ethical guidelines
- Detect and handle PII
- Validate output formats
- Filter prohibited content
- Ensure quality constraints

Example:
    >>> from SimplerLLM.language.guardrails import (
    ...     GuardrailsLLM,
    ...     PromptInjectionGuardrail,
    ...     OutputPIIDetectionGuardrail
    ... )
    >>> from SimplerLLM.language.llm.base import LLM, LLMProvider
    >>>
    >>> # Create base LLM
    >>> llm = LLM.create(provider=LLMProvider.OPENAI, api_key="...")
    >>>
    >>> # Add guardrails
    >>> guardrailed_llm = GuardrailsLLM(
    ...     llm_instance=llm,
    ...     input_guardrails=[PromptInjectionGuardrail()],
    ...     output_guardrails=[OutputPIIDetectionGuardrail(config={"action_on_detect": "redact"})]
    ... )
    >>>
    >>> # Use like normal LLM
    >>> response = guardrailed_llm.generate_response(
    ...     prompt="Hello!",
    ...     full_response=True
    ... )
    >>> print(response.guardrails_metadata)
"""

# Core classes
from .base import (
    GuardrailAction,
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    CompositeGuardrail,
)

# Wrapper
from .wrapper import GuardrailsLLM

# Exceptions
from .exceptions import (
    GuardrailException,
    GuardrailBlockedException,
    GuardrailValidationException,
    GuardrailConfigurationException,
    GuardrailTimeoutException,
)

# Input guardrails
from .input_guardrails import (
    PromptInjectionGuardrail,
    TopicFilterGuardrail,
    InputPIIDetectionGuardrail,
)

# Output guardrails
from .output_guardrails import (
    FormatValidatorGuardrail,
    OutputPIIDetectionGuardrail,
    ContentSafetyGuardrail,
    LengthValidatorGuardrail,
)

__all__ = [
    # Core classes
    "GuardrailAction",
    "GuardrailResult",
    "InputGuardrail",
    "OutputGuardrail",
    "CompositeGuardrail",
    # Wrapper
    "GuardrailsLLM",
    # Exceptions
    "GuardrailException",
    "GuardrailBlockedException",
    "GuardrailValidationException",
    "GuardrailConfigurationException",
    "GuardrailTimeoutException",
    # Input guardrails
    "PromptInjectionGuardrail",
    "TopicFilterGuardrail",
    "InputPIIDetectionGuardrail",
    # Output guardrails
    "FormatValidatorGuardrail",
    "OutputPIIDetectionGuardrail",
    "ContentSafetyGuardrail",
    "LengthValidatorGuardrail",
]

# Version
__version__ = "1.0.0"
