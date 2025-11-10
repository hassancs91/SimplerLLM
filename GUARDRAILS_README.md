# SimplerLLM Guardrails

A comprehensive guardrails system for SimplerLLM that adds safety, quality, and compliance checks to LLM interactions.

## Overview

Guardrails provides a modular, pluggable layer that can:
- **Input Guardrails**: Validate and modify prompts before sending to LLMs
- **Output Guardrails**: Validate and modify responses after generation
- Work with **any LLM provider** (OpenAI, Anthropic, Gemini, etc.)
- Integrate seamlessly with **Pydantic JSON generation** and **ReliableLLM**
- Support both **synchronous and asynchronous** operations

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Built-in Guardrails](#built-in-guardrails)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)
7. [Custom Guardrails](#custom-guardrails)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)

---

## Installation

Guardrails is included in SimplerLLM. No additional installation required!

```python
from SimplerLLM.language.guardrails import GuardrailsLLM
```

---

## Quick Start

### Basic Example

```python
from SimplerLLM.language.llm.base import LLM, LLMProvider
from SimplerLLM.language.guardrails import (
    GuardrailsLLM,
    PromptInjectionGuardrail,
    OutputPIIDetectionGuardrail
)

# Create base LLM
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    api_key="your-api-key"
)

# Add guardrails
guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    input_guardrails=[
        PromptInjectionGuardrail()  # Adds safety rules
    ],
    output_guardrails=[
        OutputPIIDetectionGuardrail(config={
            "action_on_detect": "redact"  # Auto-redact PII
        })
    ]
)

# Use like normal LLM
response = guardrailed_llm.generate_response(
    model_name="gpt-4",
    prompt="Tell me about user john@example.com",
    full_response=True
)

print(response.generated_text)  # PII will be redacted
print(response.guardrails_metadata)  # See what guardrails did
```

### Works with Pydantic JSON

```python
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from SimplerLLM.language.guardrails import GuardrailsLLM, FormatValidatorGuardrail
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

# Wrap LLM with guardrails
guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    output_guardrails=[
        FormatValidatorGuardrail(config={
            "format_type": "json",
            "strict": True
        })
    ]
)

# Use with Pydantic generation - guardrails apply automatically!
result = generate_pydantic_json_model(
    model_class=UserProfile,
    llm_instance=guardrailed_llm,
    prompt="Generate a user profile"
)
```

### Works with ReliableLLM

```python
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.guardrails import GuardrailsLLM

# Wrap each LLM with guardrails
primary_guardrailed = GuardrailsLLM(primary_llm, input_guardrails=[...])
fallback_guardrailed = GuardrailsLLM(fallback_llm, input_guardrails=[...])

# Create reliable LLM with guardrailed instances
reliable_llm = ReliableLLM(
    primary_llm=primary_guardrailed,
    fallback_llm=fallback_guardrailed
)
```

---

## Core Concepts

### GuardrailsLLM Wrapper

`GuardrailsLLM` is a wrapper class that adds guardrails to any LLM instance. It:
- Inherits from `LLM` base class
- Transparently intercepts `generate_response()` and `generate_response_async()`
- Applies guardrails in sequence: **Input → LLM Call → Output**
- Returns enhanced responses with metadata

### Guardrail Types

#### Input Guardrails
Applied **before** LLM generation to:
- Inject safety instructions
- Block prohibited topics
- Detect/redact PII in prompts
- Validate or modify user input

#### Output Guardrails
Applied **after** LLM generation to:
- Validate response format
- Detect/redact PII in responses
- Check content safety
- Enforce length constraints

### Guardrail Actions

Each guardrail can take one of four actions:

- **ALLOW**: Content passes validation, continue normally
- **MODIFY**: Content modified (e.g., PII redacted), use modified version
- **BLOCK**: Content violates rules, raise `GuardrailBlockedException`
- **WARN**: Content has issues but allowed, log warning

---

## Built-in Guardrails

### Input Guardrails

#### 1. PromptInjectionGuardrail

Injects safety rules into the system prompt.

```python
from SimplerLLM.language.guardrails import PromptInjectionGuardrail

guardrail = PromptInjectionGuardrail(config={
    "safety_rules": "Always be helpful and harmless.",
    "position": "prepend",  # or "append"
    "custom_instructions": [
        "Decline harmful requests",
        "Be truthful and accurate"
    ]
})
```

**Use Cases**:
- Enforce ethical guidelines
- Add domain-specific instructions
- Ensure consistent behavior

#### 2. TopicFilterGuardrail

Blocks or filters prohibited topics/keywords.

```python
from SimplerLLM.language.guardrails import TopicFilterGuardrail

guardrail = TopicFilterGuardrail(config={
    "prohibited_topics": ["violence", "illegal activities", "hate speech"],
    "action_on_match": "block",  # or "warn", "modify"
    "case_sensitive": False,
    "match_whole_words": True
})
```

**Use Cases**:
- Prevent discussion of sensitive topics
- Filter inappropriate requests
- Enforce content policies

#### 3. InputPIIDetectionGuardrail

Detects PII in user prompts.

```python
from SimplerLLM.language.guardrails import InputPIIDetectionGuardrail

guardrail = InputPIIDetectionGuardrail(config={
    "action_on_detect": "redact",  # or "block", "warn"
    "pii_types": ["email", "phone", "ssn", "credit_card"],
    "redaction_text": "[REDACTED]"
})
```

**Use Cases**:
- Protect user privacy
- Compliance with data protection regulations
- Prevent accidental PII exposure

### Output Guardrails

#### 4. FormatValidatorGuardrail

Validates response format (JSON, XML, etc.).

```python
from SimplerLLM.language.guardrails import FormatValidatorGuardrail

guardrail = FormatValidatorGuardrail(config={
    "format_type": "json",  # or "xml", "markdown", "plain", "custom"
    "strict": True,
    "required_fields": ["name", "age"],  # For JSON
    "extract_json": True  # Try to extract JSON from text
})
```

**Use Cases**:
- Ensure structured output
- Validate API responses
- Enforce data contracts

#### 5. OutputPIIDetectionGuardrail

Detects and redacts PII in LLM responses.

```python
from SimplerLLM.language.guardrails import OutputPIIDetectionGuardrail

guardrail = OutputPIIDetectionGuardrail(config={
    "action_on_detect": "redact",
    "pii_types": ["email", "phone", "ssn"],
    "allow_examples": True,  # Allow example.com, 555-0100, etc.
    "include_type_in_redaction": False
})
```

**Use Cases**:
- Prevent PII leakage in responses
- Compliance with privacy regulations
- Protect sensitive information

#### 6. ContentSafetyGuardrail

Checks for unsafe content (profanity, violence, hate speech).

```python
from SimplerLLM.language.guardrails import ContentSafetyGuardrail

guardrail = ContentSafetyGuardrail(config={
    "action_on_detect": "block",  # or "warn", "modify"
    "check_profanity": True,
    "check_violence": True,
    "check_hate_speech": True,
    "severity_threshold": "medium",  # or "low", "high"
    "replacement_text": "[CONTENT_REMOVED]"
})
```

**Use Cases**:
- Content moderation
- Brand safety
- User protection

#### 7. LengthValidatorGuardrail

Enforces length constraints on responses.

```python
from SimplerLLM.language.guardrails import LengthValidatorGuardrail

guardrail = LengthValidatorGuardrail(config={
    "min_length": 10,
    "max_length": 500,
    "unit": "words",  # or "characters", "sentences"
    "action_on_violation": "truncate",  # or "block", "warn"
    "truncate_position": "end",  # or "middle"
    "truncation_indicator": "..."
})
```

**Use Cases**:
- Control response verbosity
- Ensure minimum quality
- Fit within UI constraints

---

## Configuration

### GuardrailsLLM Configuration

```python
config = {
    "fail_fast": True,        # Stop on first BLOCK (default: True)
    "include_metadata": True,  # Add metadata to response (default: True)
    "auto_modify": True,       # Apply MODIFY actions automatically (default: True)
    "skip_on_error": False     # Continue if guardrail fails (default: False)
}

guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    input_guardrails=[...],
    output_guardrails=[...],
    config=config
)
```

### Per-Guardrail Configuration

Each guardrail accepts a `config` dictionary:

```python
guardrail = SomeGuardrail(config={
    "enabled": True,      # Enable/disable this guardrail
    # ... guardrail-specific options
})
```

Disable a guardrail temporarily:

```python
guardrail.enabled = False
```

---

## Advanced Usage

### Multiple Guardrails

Chain multiple guardrails together:

```python
guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    input_guardrails=[
        PromptInjectionGuardrail(),
        TopicFilterGuardrail(config={"prohibited_topics": ["violence"]}),
        InputPIIDetectionGuardrail(config={"action_on_detect": "warn"})
    ],
    output_guardrails=[
        FormatValidatorGuardrail(config={"format_type": "json"}),
        OutputPIIDetectionGuardrail(config={"action_on_detect": "redact"}),
        ContentSafetyGuardrail(config={"action_on_detect": "block"}),
        LengthValidatorGuardrail(config={"max_length": 1000, "unit": "words"})
    ]
)
```

Guardrails are executed in order. Use `fail_fast=True` to stop on first block.

### Accessing Metadata

```python
response = guardrailed_llm.generate_response(
    prompt="Hello",
    full_response=True  # Required for metadata
)

# Access guardrails metadata
metadata = response.guardrails_metadata

print(f"Input guardrails: {metadata['input_guardrails']}")
print(f"Output guardrails: {metadata['output_guardrails']}")
print(f"Prompt modified: {metadata['prompt_modified']}")
print(f"Response modified: {metadata['response_modified']}")
print(f"All passed: {metadata['guardrails_passed']}")

# Inspect individual guardrail results
for result in metadata['input_guardrails']:
    print(f"{result['guardrail']}: {result['action']} - {result['message']}")
```

### Async Usage

```python
async def generate_with_guardrails():
    response = await guardrailed_llm.generate_response_async(
        model_name="gpt-4",
        prompt="Hello",
        full_response=True
    )
    return response

# Run async
import asyncio
response = asyncio.run(generate_with_guardrails())
```

### Error Handling

```python
from SimplerLLM.language.guardrails import (
    GuardrailBlockedException,
    GuardrailValidationException
)

try:
    response = guardrailed_llm.generate_response(
        prompt="Prohibited content"
    )
except GuardrailBlockedException as e:
    print(f"Content blocked by {e.guardrail_name}: {e}")
    print(f"Metadata: {e.metadata}")
except GuardrailValidationException as e:
    print(f"Guardrail error: {e}")
```

### Conditional Guardrails

Enable/disable guardrails based on conditions:

```python
# Create guardrail
pii_detector = OutputPIIDetectionGuardrail()

# Conditionally enable
if user.privacy_mode:
    pii_detector.enabled = True
else:
    pii_detector.enabled = False

guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    output_guardrails=[pii_detector]
)
```

---

## Custom Guardrails

### Creating a Custom Input Guardrail

```python
from SimplerLLM.language.guardrails.base import (
    InputGuardrail,
    GuardrailResult,
    GuardrailAction
)

class CustomPromptValidatorGuardrail(InputGuardrail):
    """Validates prompts meet custom criteria."""

    def validate(self, prompt, system_prompt, messages=None, **kwargs):
        # Your validation logic
        if len(prompt) < 10:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                passed=False,
                message="Prompt too short",
                guardrail_name=self.name
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            passed=True,
            message="Prompt valid",
            guardrail_name=self.name
        )

    async def validate_async(self, prompt, system_prompt, messages=None, **kwargs):
        # Async version
        return self.validate(prompt, system_prompt, messages, **kwargs)
```

### Creating a Custom Output Guardrail

```python
from SimplerLLM.language.guardrails.base import (
    OutputGuardrail,
    GuardrailResult,
    GuardrailAction
)

class ToxicityFilterGuardrail(OutputGuardrail):
    """Filters toxic content from responses."""

    def __init__(self, config=None):
        super().__init__(config)
        self.toxic_words = self.config.get("toxic_words", [])

    def validate(self, response, original_prompt="", **kwargs):
        # Check for toxic words
        for word in self.toxic_words:
            if word.lower() in response.lower():
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    message=f"Toxic content detected: {word}",
                    metadata={"detected_word": word},
                    guardrail_name=self.name
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            passed=True,
            message="No toxic content detected",
            guardrail_name=self.name
        )

    async def validate_async(self, response, original_prompt="", **kwargs):
        return self.validate(response, original_prompt, **kwargs)

# Use custom guardrail
custom_guardrail = ToxicityFilterGuardrail(config={
    "toxic_words": ["bad", "evil", "harmful"]
})

guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    output_guardrails=[custom_guardrail]
)
```

---

## Best Practices

### 1. **Start Simple**
Begin with 1-2 guardrails and add more as needed:

```python
# Good: Start with essential guardrails
guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    input_guardrails=[PromptInjectionGuardrail()],
    output_guardrails=[OutputPIIDetectionGuardrail()]
)
```

### 2. **Order Matters**
Place blocking guardrails early, modifying ones later:

```python
output_guardrails=[
    FormatValidatorGuardrail(),  # Block if format invalid
    ContentSafetyGuardrail(),     # Block if unsafe
    OutputPIIDetectionGuardrail(), # Redact PII (modify)
    LengthValidatorGuardrail()    # Truncate if needed (modify)
]
```

### 3. **Use Appropriate Actions**
- **BLOCK**: For strict violations (hate speech, illegal content)
- **WARN**: For monitoring (track patterns without blocking)
- **MODIFY**: For fixable issues (PII redaction, truncation)

### 4. **Monitor Metadata**
Always inspect `guardrails_metadata` in production:

```python
response = guardrailed_llm.generate_response(prompt="...", full_response=True)

# Log guardrail activity
if not response.guardrails_metadata['guardrails_passed']:
    logger.warning(f"Guardrails triggered: {response.guardrails_metadata}")
```

### 5. **Test Thoroughly**
Test guardrails with edge cases:

```python
# Test PII detection
test_prompts = [
    "My email is test@example.com",
    "Call me at 555-1234",
    "SSN: 123-45-6789"
]

for prompt in test_prompts:
    try:
        response = guardrailed_llm.generate_response(prompt=prompt)
        print(f"Response: {response}")
    except GuardrailBlockedException as e:
        print(f"Blocked: {e.message}")
```

### 6. **Performance Considerations**
- Guardrails add latency (typically <100ms per guardrail)
- Use `skip_on_error=True` for non-critical guardrails
- Disable guardrails in development if needed

### 7. **Composition**
Combine with other SimplerLLM features:

```python
# ReliableLLM + Guardrails
reliable_llm = ReliableLLM(
    primary_llm=GuardrailsLLM(primary, ...),
    fallback_llm=GuardrailsLLM(fallback, ...)
)

# MiniAgent + Guardrails
from SimplerLLM.language.flow import MiniAgent

agent = MiniAgent(
    llm_instance=guardrailed_llm,
    steps=[...]
)
```

---

## API Reference

### GuardrailsLLM

```python
class GuardrailsLLM(LLM):
    def __init__(
        self,
        llm_instance: LLM,
        input_guardrails: List[InputGuardrail] = None,
        output_guardrails: List[OutputGuardrail] = None,
        config: Dict[str, Any] = None
    )

    def generate_response(
        self,
        model_name: str = None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        **kwargs
    )

    async def generate_response_async(...)
```

### Base Classes

```python
class InputGuardrail(ABC):
    def validate(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> GuardrailResult

    async def validate_async(...) -> GuardrailResult


class OutputGuardrail(ABC):
    def validate(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> GuardrailResult

    async def validate_async(...) -> GuardrailResult


class GuardrailResult:
    action: GuardrailAction
    passed: bool
    message: Optional[str]
    modified_content: Optional[str]
    metadata: Dict[str, Any]
    guardrail_name: str
```

### Exceptions

```python
class GuardrailBlockedException(Exception):
    """Raised when content is blocked by a guardrail"""

class GuardrailValidationException(Exception):
    """Raised when guardrail validation fails unexpectedly"""

class GuardrailConfigurationException(Exception):
    """Raised when guardrail is misconfigured"""
```

---

## Examples

### Example 1: E-commerce Chatbot

```python
# Protect customer data and ensure quality responses
guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    input_guardrails=[
        PromptInjectionGuardrail(config={
            "safety_rules": "You are a helpful e-commerce assistant. Never share customer data."
        }),
        InputPIIDetectionGuardrail(config={"action_on_detect": "warn"})
    ],
    output_guardrails=[
        OutputPIIDetectionGuardrail(config={"action_on_detect": "redact"}),
        LengthValidatorGuardrail(config={"max_length": 200, "unit": "words"}),
        ContentSafetyGuardrail(config={"action_on_detect": "block"})
    ]
)
```

### Example 2: Healthcare Assistant

```python
# Strict compliance with health data regulations
guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    input_guardrails=[
        PromptInjectionGuardrail(config={
            "safety_rules": "Follow HIPAA guidelines. Never diagnose. Recommend seeing a doctor."
        }),
        TopicFilterGuardrail(config={
            "prohibited_topics": ["self-harm", "dangerous medical advice"],
            "action_on_match": "block"
        })
    ],
    output_guardrails=[
        OutputPIIDetectionGuardrail(config={
            "action_on_detect": "block",  # Never leak patient data
            "pii_types": ["ssn", "phone", "email"]
        }),
        ContentSafetyGuardrail(config={"severity_threshold": "low"})
    ]
)
```

### Example 3: API with Structured Output

```python
# Ensure valid JSON responses
guardrailed_llm = GuardrailsLLM(
    llm_instance=llm,
    output_guardrails=[
        FormatValidatorGuardrail(config={
            "format_type": "json",
            "strict": True,
            "required_fields": ["status", "data"],
            "extract_json": True
        })
    ]
)

# Use with Pydantic
from pydantic import BaseModel

class APIResponse(BaseModel):
    status: str
    data: dict

result = generate_pydantic_json_model(
    model_class=APIResponse,
    llm_instance=guardrailed_llm,
    prompt="Generate API response"
)
```

---

## Troubleshooting

### Guardrail is Not Triggering

1. Check if guardrail is enabled: `guardrail.enabled`
2. Verify configuration is correct
3. Enable metadata to see execution: `full_response=True`

### False Positives

1. Adjust sensitivity: e.g., `severity_threshold="high"`
2. Add exceptions: e.g., `allow_examples=True` for PII
3. Use `action_on_detect="warn"` instead of `"block"`

### Performance Issues

1. Reduce number of guardrails
2. Use simpler patterns in regex-based guardrails
3. Consider running guardrails in parallel (advanced)

---

## Contributing

Want to add a new guardrail? Follow the custom guardrail pattern and submit a PR!

---

## License

Part of SimplerLLM - see main library license.

---

## Support

For issues, questions, or feature requests, please open an issue on the SimplerLLM GitHub repository.

---

**Happy Safe LLM Building! 🛡️**
