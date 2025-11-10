"""
PII detection guardrail for output responses.
"""

import re
from typing import Optional, List, Dict
from SimplerLLM.language.guardrails.base import (
    OutputGuardrail,
    GuardrailResult,
    GuardrailAction
)


class OutputPIIDetectionGuardrail(OutputGuardrail):
    """
    Detects and optionally redacts PII in LLM-generated responses.

    This guardrail scans LLM output for common PII patterns like emails,
    phone numbers, SSNs, credit cards, etc. and can block, warn, or redact.

    Configuration:
        - action_on_detect (str): Action to take - 'block', 'warn', or 'redact' (default: 'redact')
        - redaction_text (str): Text to use for redaction (default: '[REDACTED]')
        - pii_types (list): Types of PII to detect (default: all)
          Available: email, phone, ssn, credit_card, ip_address, url
        - strict_mode (bool): Use stricter pattern matching (default: False)
        - allow_examples (bool): Allow PII in example formats (default: False)

    Example:
        >>> guardrail = OutputPIIDetectionGuardrail(config={
        ...     "action_on_detect": "redact",
        ...     "pii_types": ["email", "phone", "ssn"]
        ... })
        >>> result = guardrail.validate(
        ...     response="Contact John at john@example.com or 555-1234",
        ...     original_prompt="How to contact John?"
        ... )
    """

    # PII regex patterns (same as input)
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "url": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
    }

    # Common example domains to allow (if allow_examples is True)
    EXAMPLE_DOMAINS = [
        "example.com", "example.org", "example.net",
        "test.com", "demo.com", "sample.com",
        "localhost", "127.0.0.1"
    ]

    # Example phone numbers to allow
    EXAMPLE_PHONES = [
        "555-0100", "555-0199",  # Reserved for examples
        "123-456-7890", "000-000-0000"  # Obviously fake
    ]

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the output PII detection guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get configuration
        self.action_on_detect = self.config.get("action_on_detect", "redact")
        self.redaction_text = self.config.get("redaction_text", "[REDACTED]")
        self.pii_types = self.config.get("pii_types", list(self.PII_PATTERNS.keys()))
        self.strict_mode = self.config.get("strict_mode", False)
        self.allow_examples = self.config.get("allow_examples", False)

        # Validate action
        if self.action_on_detect not in ["block", "warn", "redact"]:
            self.action_on_detect = "redact"

        # Compile patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for enabled PII types."""
        self.compiled_patterns = {}

        for pii_type in self.pii_types:
            if pii_type in self.PII_PATTERNS:
                self.compiled_patterns[pii_type] = re.compile(self.PII_PATTERNS[pii_type])

    def _is_example_pii(self, pii_value: str, pii_type: str) -> bool:
        """
        Check if PII value is a common example/placeholder.

        Args:
            pii_value: The detected PII value
            pii_type: Type of PII

        Returns:
            True if it's likely an example value
        """
        if not self.allow_examples:
            return False

        pii_lower = pii_value.lower()

        if pii_type == "email":
            return any(domain in pii_lower for domain in self.EXAMPLE_DOMAINS)
        elif pii_type == "phone":
            return any(phone in pii_value for phone in self.EXAMPLE_PHONES)
        elif pii_type == "ssn":
            return pii_value in ["123-45-6789", "000-00-0000"]
        elif pii_type == "ip_address":
            return pii_value in ["127.0.0.1", "0.0.0.0"]

        return False

    def _detect_pii(self, text: str) -> List[Dict]:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of detected PII with details
        """
        detected = []

        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                pii_value = match.group()

                # Skip if it's an example value (if configured)
                if self._is_example_pii(pii_value, pii_type):
                    continue

                detected.append({
                    "type": pii_type,
                    "value": pii_value,
                    "position": match.span(),
                    "start": match.start(),
                    "end": match.end()
                })

        # Sort by position for easier redaction
        detected.sort(key=lambda x: x["start"])

        return detected

    def _redact_pii(self, text: str, detected_pii: List[Dict]) -> str:
        """
        Redact PII from text.

        Args:
            text: Original text
            detected_pii: List of detected PII

        Returns:
            Text with PII redacted
        """
        if not detected_pii:
            return text

        # Redact from end to beginning to preserve positions
        redacted = text
        for pii in reversed(detected_pii):
            start, end = pii["position"]
            # Add type information to redaction
            redaction = f"{self.redaction_text}"
            if self.config.get("include_type_in_redaction", False):
                redaction = f"[{pii['type'].upper()}_REDACTED]"
            redacted = redacted[:start] + redaction + redacted[end:]

        return redacted

    def validate(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> GuardrailResult:
        """
        Detect PII in LLM response.

        Args:
            response: Generated response to check
            original_prompt: Original user prompt (for context)
            **kwargs: Additional context

        Returns:
            GuardrailResult with appropriate action
        """
        if not response:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                passed=True,
                message="No response to check",
                guardrail_name=self.name
            )

        # Detect PII
        detected_pii = self._detect_pii(response)

        if detected_pii:
            # PII found in output
            pii_types = list(set(p["type"] for p in detected_pii))
            message = f"PII detected in output: {', '.join(pii_types)}"

            if self.action_on_detect == "block":
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    message=message,
                    metadata={
                        "detected_pii": detected_pii,
                        "pii_types": pii_types,
                        "count": len(detected_pii)
                    },
                    guardrail_name=self.name
                )
            elif self.action_on_detect == "warn":
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    passed=False,
                    message=message + " (warning only)",
                    metadata={
                        "detected_pii": detected_pii,
                        "pii_types": pii_types,
                        "count": len(detected_pii)
                    },
                    guardrail_name=self.name
                )
            else:  # redact
                redacted_response = self._redact_pii(response, detected_pii)
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    passed=False,
                    message=f"PII redacted from output: {', '.join(pii_types)}",
                    modified_content=redacted_response,
                    metadata={
                        "detected_pii": detected_pii,
                        "pii_types": pii_types,
                        "count": len(detected_pii)
                    },
                    guardrail_name=self.name
                )

        # No PII detected
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            passed=True,
            message="No PII detected in output",
            guardrail_name=self.name
        )

    async def validate_async(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> GuardrailResult:
        """
        Async version of validate.

        Since this guardrail doesn't make async calls, it just delegates
        to the sync version.
        """
        return self.validate(response, original_prompt, **kwargs)
