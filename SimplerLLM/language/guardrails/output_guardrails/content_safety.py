"""
Content safety guardrail for output responses.
"""

import re
from typing import Optional, List, Dict
from SimplerLLM.language.guardrails.base import (
    OutputGuardrail,
    GuardrailResult,
    GuardrailAction
)


class ContentSafetyGuardrail(OutputGuardrail):
    """
    Checks LLM output for unsafe or inappropriate content.

    This guardrail scans generated responses for harmful content including
    profanity, violence, hate speech, and other inappropriate material.

    Configuration:
        - action_on_detect (str): Action to take - 'block', 'warn', or 'modify' (default: 'block')
        - check_profanity (bool): Check for profanity (default: True)
        - check_violence (bool): Check for violent content (default: True)
        - check_hate_speech (bool): Check for hate speech (default: True)
        - custom_unsafe_patterns (list): Additional patterns to check
        - severity_threshold (str): Minimum severity to act on - 'low', 'medium', 'high' (default: 'medium')
        - replacement_text (str): Text to use when modifying unsafe content (default: '[CONTENT_REMOVED]')

    Example:
        >>> guardrail = ContentSafetyGuardrail(config={
        ...     "action_on_detect": "block",
        ...     "check_profanity": True,
        ...     "severity_threshold": "medium"
        ... })
        >>> result = guardrail.validate(
        ...     response="This is a safe response",
        ...     original_prompt="Tell me a story"
        ... )
    """

    # Basic profanity patterns (simplified - in production, use a comprehensive list)
    PROFANITY_PATTERNS = [
        r'\b(?:damn|hell|crap)\b',  # Low severity
        r'\b(?:shit|ass|bitch)\b',  # Medium severity
        r'\b(?:fuck|cock|pussy|dick)\b',  # High severity
    ]

    # Violence patterns
    VIOLENCE_PATTERNS = [
        r'\b(?:kill|murder|assault|attack|harm|hurt|stab|shoot|beat)\s+(?:him|her|them|someone|people)\b',
        r'\b(?:how to|ways to)\s+(?:kill|murder|harm|hurt)\b',
        r'\b(?:violent|brutal|savage)\s+(?:attack|assault)\b',
    ]

    # Hate speech patterns (basic examples)
    HATE_SPEECH_PATTERNS = [
        r'\b(?:hate|despise|inferior)\s+(?:group|race|religion|gender)\b',
        r'\b(?:all|every)\s+\w+\s+(?:are|is)\s+(?:bad|evil|inferior|stupid)\b',
    ]

    SEVERITY_LEVELS = {
        "low": 1,
        "medium": 2,
        "high": 3
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the content safety guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get configuration
        self.action_on_detect = self.config.get("action_on_detect", "block")
        self.check_profanity = self.config.get("check_profanity", True)
        self.check_violence = self.config.get("check_violence", True)
        self.check_hate_speech = self.config.get("check_hate_speech", True)
        self.custom_unsafe_patterns = self.config.get("custom_unsafe_patterns", [])
        self.severity_threshold = self.config.get("severity_threshold", "medium")
        self.replacement_text = self.config.get("replacement_text", "[CONTENT_REMOVED]")

        # Validate action
        if self.action_on_detect not in ["block", "warn", "modify"]:
            self.action_on_detect = "block"

        # Validate severity threshold
        if self.severity_threshold not in self.SEVERITY_LEVELS:
            self.severity_threshold = "medium"

        # Compile patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for safety checks."""
        self.compiled_patterns = []

        # Add profanity patterns
        if self.check_profanity:
            for i, pattern_str in enumerate(self.PROFANITY_PATTERNS):
                severity = "low" if i == 0 else "medium" if i == 1 else "high"
                self.compiled_patterns.append({
                    "pattern": re.compile(pattern_str, re.IGNORECASE),
                    "type": "profanity",
                    "severity": severity
                })

        # Add violence patterns
        if self.check_violence:
            for pattern_str in self.VIOLENCE_PATTERNS:
                self.compiled_patterns.append({
                    "pattern": re.compile(pattern_str, re.IGNORECASE),
                    "type": "violence",
                    "severity": "high"
                })

        # Add hate speech patterns
        if self.check_hate_speech:
            for pattern_str in self.HATE_SPEECH_PATTERNS:
                self.compiled_patterns.append({
                    "pattern": re.compile(pattern_str, re.IGNORECASE),
                    "type": "hate_speech",
                    "severity": "high"
                })

        # Add custom patterns (default to high severity)
        for pattern_str in self.custom_unsafe_patterns:
            try:
                self.compiled_patterns.append({
                    "pattern": re.compile(pattern_str, re.IGNORECASE),
                    "type": "custom",
                    "severity": "high"
                })
            except re.error:
                # Skip invalid patterns
                pass

    def _check_content(self, text: str) -> List[Dict]:
        """
        Check text for unsafe content.

        Args:
            text: Text to check

        Returns:
            List of detected unsafe content with details
        """
        detected = []
        threshold_level = self.SEVERITY_LEVELS[self.severity_threshold]

        for pattern_info in self.compiled_patterns:
            pattern = pattern_info["pattern"]
            severity = pattern_info["severity"]
            content_type = pattern_info["type"]

            # Only check if severity meets threshold
            if self.SEVERITY_LEVELS[severity] < threshold_level:
                continue

            matches = pattern.finditer(text)
            for match in matches:
                detected.append({
                    "type": content_type,
                    "matched_text": match.group(),
                    "position": match.span(),
                    "severity": severity,
                    "start": match.start(),
                    "end": match.end()
                })

        # Sort by position
        detected.sort(key=lambda x: x["start"])

        return detected

    def _remove_unsafe_content(self, text: str, detected_content: List[Dict]) -> str:
        """
        Remove unsafe content from text.

        Args:
            text: Original text
            detected_content: List of detected unsafe content

        Returns:
            Text with unsafe content removed
        """
        if not detected_content:
            return text

        # Remove from end to beginning to preserve positions
        modified = text
        for content in reversed(detected_content):
            start, end = content["position"]
            modified = modified[:start] + self.replacement_text + modified[end:]

        return modified

    def validate(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> GuardrailResult:
        """
        Check response for unsafe content.

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

        # Check for unsafe content
        detected_content = self._check_content(response)

        if detected_content:
            # Unsafe content found
            content_types = list(set(c["type"] for c in detected_content))
            severities = list(set(c["severity"] for c in detected_content))
            highest_severity = max(severities, key=lambda s: self.SEVERITY_LEVELS[s])

            message = f"Unsafe content detected ({', '.join(content_types)}, severity: {highest_severity})"

            if self.action_on_detect == "block":
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    message=message,
                    metadata={
                        "detected_content": detected_content,
                        "content_types": content_types,
                        "highest_severity": highest_severity,
                        "count": len(detected_content)
                    },
                    guardrail_name=self.name
                )
            elif self.action_on_detect == "warn":
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    passed=False,
                    message=message + " (warning only)",
                    metadata={
                        "detected_content": detected_content,
                        "content_types": content_types,
                        "highest_severity": highest_severity,
                        "count": len(detected_content)
                    },
                    guardrail_name=self.name
                )
            else:  # modify
                modified_response = self._remove_unsafe_content(response, detected_content)
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    passed=False,
                    message=f"Unsafe content removed: {', '.join(content_types)}",
                    modified_content=modified_response,
                    metadata={
                        "detected_content": detected_content,
                        "content_types": content_types,
                        "highest_severity": highest_severity,
                        "count": len(detected_content)
                    },
                    guardrail_name=self.name
                )

        # No unsafe content detected
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            passed=True,
            message="No unsafe content detected",
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
