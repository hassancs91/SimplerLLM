"""
PII detection guardrail for input prompts.
"""

import re
from typing import Optional, List, Dict
from SimplerLLM.language.guardrails.base import (
    InputGuardrail,
    GuardrailResult,
    GuardrailAction
)


class InputPIIDetectionGuardrail(InputGuardrail):
    """
    Detects and optionally redacts PII (Personally Identifiable Information) in user prompts.

    This guardrail scans user input for common PII patterns like emails,
    phone numbers, SSNs, credit cards, etc. and can block, warn, or redact.

    Configuration:
        - action_on_detect (str): Action to take - 'block', 'warn', or 'redact' (default: 'warn')
        - redaction_text (str): Text to use for redaction (default: '[PII_REDACTED]')
        - pii_types (list): Types of PII to detect (default: all)
          Available: email, phone, ssn, credit_card, ip_address, url
        - strict_mode (bool): Use stricter pattern matching (default: False)

    Example:
        >>> guardrail = InputPIIDetectionGuardrail(config={
        ...     "action_on_detect": "redact",
        ...     "pii_types": ["email", "phone"]
        ... })
        >>> result = guardrail.validate(
        ...     prompt="Contact me at john@example.com or 555-1234",
        ...     system_prompt="You are helpful"
        ... )
    """

    # PII regex patterns
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "url": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
    }

    # Stricter patterns for more accurate detection
    STRICT_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the input PII detection guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get configuration
        self.action_on_detect = self.config.get("action_on_detect", "warn")
        self.redaction_text = self.config.get("redaction_text", "[PII_REDACTED]")
        self.pii_types = self.config.get("pii_types", list(self.PII_PATTERNS.keys()))
        self.strict_mode = self.config.get("strict_mode", False)

        # Validate action
        if self.action_on_detect not in ["block", "warn", "redact"]:
            self.action_on_detect = "warn"

        # Compile patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for enabled PII types."""
        self.compiled_patterns = {}

        pattern_dict = self.STRICT_PATTERNS if self.strict_mode else self.PII_PATTERNS

        for pii_type in self.pii_types:
            if pii_type in pattern_dict:
                self.compiled_patterns[pii_type] = re.compile(pattern_dict[pii_type])

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
                detected.append({
                    "type": pii_type,
                    "value": match.group(),
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
            redacted = redacted[:start] + self.redaction_text + redacted[end:]

        return redacted

    def validate(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> GuardrailResult:
        """
        Detect PII in user prompt.

        Args:
            prompt: User prompt to check
            system_prompt: System prompt (not checked)
            messages: Optional conversation messages
            **kwargs: Additional context

        Returns:
            GuardrailResult with appropriate action
        """
        if not prompt:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                passed=True,
                message="No prompt to check",
                guardrail_name=self.name
            )

        # Detect PII
        detected_pii = self._detect_pii(prompt)

        if detected_pii:
            # PII found
            pii_types = list(set(p["type"] for p in detected_pii))
            message = f"PII detected in input: {', '.join(pii_types)}"

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
                redacted_prompt = self._redact_pii(prompt, detected_pii)
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    passed=False,
                    message=f"PII redacted from input: {', '.join(pii_types)}",
                    modified_content=redacted_prompt,
                    metadata={
                        "detected_pii": detected_pii,
                        "pii_types": pii_types,
                        "count": len(detected_pii),
                        "target": "prompt"
                    },
                    guardrail_name=self.name
                )

        # No PII detected
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            passed=True,
            message="No PII detected in input",
            guardrail_name=self.name
        )

    async def validate_async(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> GuardrailResult:
        """
        Async version of validate.

        Since this guardrail doesn't make async calls, it just delegates
        to the sync version.
        """
        return self.validate(prompt, system_prompt, messages, **kwargs)
