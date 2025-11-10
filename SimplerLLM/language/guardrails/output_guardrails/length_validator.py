"""
Length validator guardrail for output responses.
"""

from typing import Optional, Dict
from SimplerLLM.language.guardrails.base import (
    OutputGuardrail,
    GuardrailResult,
    GuardrailAction
)


class LengthValidatorGuardrail(OutputGuardrail):
    """
    Validates that LLM output meets length constraints.

    This guardrail checks that generated responses fall within specified
    minimum and maximum length constraints (characters, words, or sentences).

    Configuration:
        - min_length (int): Minimum length (default: None, no minimum)
        - max_length (int): Maximum length (default: None, no maximum)
        - unit (str): Unit of measurement - 'characters', 'words', 'sentences' (default: 'characters')
        - action_on_violation (str): Action to take - 'block', 'warn', or 'truncate' (default: 'block')
        - truncate_position (str): Where to truncate - 'end', 'middle' (default: 'end')
        - truncation_indicator (str): Text to add when truncating (default: '...')

    Example:
        >>> guardrail = LengthValidatorGuardrail(config={
        ...     "min_length": 10,
        ...     "max_length": 500,
        ...     "unit": "words",
        ...     "action_on_violation": "block"
        ... })
        >>> result = guardrail.validate(
        ...     response="This is a short response",
        ...     original_prompt="Explain something"
        ... )
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the length validator guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get configuration
        self.min_length = self.config.get("min_length", None)
        self.max_length = self.config.get("max_length", None)
        self.unit = self.config.get("unit", "characters")
        self.action_on_violation = self.config.get("action_on_violation", "block")
        self.truncate_position = self.config.get("truncate_position", "end")
        self.truncation_indicator = self.config.get("truncation_indicator", "...")

        # Validate configuration
        if self.unit not in ["characters", "words", "sentences"]:
            self.unit = "characters"

        if self.action_on_violation not in ["block", "warn", "truncate"]:
            self.action_on_violation = "block"

        if self.truncate_position not in ["end", "middle"]:
            self.truncate_position = "end"

    def _measure_length(self, text: str) -> int:
        """
        Measure text length based on configured unit.

        Args:
            text: Text to measure

        Returns:
            Length in configured units
        """
        if self.unit == "characters":
            return len(text)
        elif self.unit == "words":
            return len(text.split())
        elif self.unit == "sentences":
            # Simple sentence counting (split by . ! ?)
            import re
            sentences = re.split(r'[.!?]+', text)
            return len([s for s in sentences if s.strip()])
        return 0

    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to max length.

        Args:
            text: Text to truncate
            max_length: Maximum length in configured units

        Returns:
            Truncated text
        """
        if self.unit == "characters":
            if self.truncate_position == "end":
                return text[:max_length - len(self.truncation_indicator)] + self.truncation_indicator
            else:  # middle
                half = (max_length - len(self.truncation_indicator)) // 2
                return text[:half] + self.truncation_indicator + text[-half:]

        elif self.unit == "words":
            words = text.split()
            if self.truncate_position == "end":
                truncated_words = words[:max_length]
                return " ".join(truncated_words) + self.truncation_indicator
            else:  # middle
                half = max_length // 2
                return " ".join(words[:half]) + self.truncation_indicator + " ".join(words[-half:])

        elif self.unit == "sentences":
            import re
            # Split into sentences while preserving delimiters
            sentences = re.split(r'([.!?]+)', text)
            # Recombine sentences with their delimiters
            full_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    full_sentences.append(sentences[i] + sentences[i + 1])

            if self.truncate_position == "end":
                truncated = "".join(full_sentences[:max_length])
                return truncated + self.truncation_indicator
            else:  # middle
                half = max_length // 2
                return "".join(full_sentences[:half]) + self.truncation_indicator + "".join(full_sentences[-half:])

        return text

    def validate(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> GuardrailResult:
        """
        Validate response length.

        Args:
            response: Generated response to validate
            original_prompt: Original user prompt (for context)
            **kwargs: Additional context

        Returns:
            GuardrailResult with validation result
        """
        if not response:
            if self.min_length and self.min_length > 0:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK if self.action_on_violation == "block" else GuardrailAction.WARN,
                    passed=False,
                    message="Empty response received",
                    metadata={
                        "length": 0,
                        "unit": self.unit,
                        "violation": "too_short"
                    },
                    guardrail_name=self.name
                )
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                passed=True,
                message="Empty response allowed",
                guardrail_name=self.name
            )

        # Measure length
        length = self._measure_length(response)
        metadata = {
            "length": length,
            "unit": self.unit,
            "min_length": self.min_length,
            "max_length": self.max_length
        }

        # Check minimum length
        if self.min_length is not None and length < self.min_length:
            message = f"Response too short: {length} {self.unit} (minimum: {self.min_length})"

            if self.action_on_violation == "block":
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    message=message,
                    metadata={**metadata, "violation": "too_short"},
                    guardrail_name=self.name
                )
            else:  # warn or truncate (can't truncate if too short)
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    passed=False,
                    message=message + " (warning only)",
                    metadata={**metadata, "violation": "too_short"},
                    guardrail_name=self.name
                )

        # Check maximum length
        if self.max_length is not None and length > self.max_length:
            message = f"Response too long: {length} {self.unit} (maximum: {self.max_length})"

            if self.action_on_violation == "block":
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    message=message,
                    metadata={**metadata, "violation": "too_long"},
                    guardrail_name=self.name
                )
            elif self.action_on_violation == "warn":
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    passed=False,
                    message=message + " (warning only)",
                    metadata={**metadata, "violation": "too_long"},
                    guardrail_name=self.name
                )
            else:  # truncate
                truncated_response = self._truncate_text(response, self.max_length)
                new_length = self._measure_length(truncated_response)
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    passed=False,
                    message=f"Response truncated from {length} to {new_length} {self.unit}",
                    modified_content=truncated_response,
                    metadata={
                        **metadata,
                        "violation": "too_long",
                        "new_length": new_length,
                        "truncated": True
                    },
                    guardrail_name=self.name
                )

        # Length is within bounds
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            passed=True,
            message=f"Response length valid: {length} {self.unit}",
            metadata=metadata,
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
