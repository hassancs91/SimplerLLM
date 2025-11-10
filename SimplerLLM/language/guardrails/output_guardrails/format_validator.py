"""
Format validator guardrail for output responses.
"""

import json
import re
from typing import Optional, Dict
from SimplerLLM.language.guardrails.base import (
    OutputGuardrail,
    GuardrailResult,
    GuardrailAction
)


class FormatValidatorGuardrail(OutputGuardrail):
    """
    Validates that LLM output matches expected format.

    This guardrail checks that the generated response conforms to a
    specified format like JSON, XML, markdown, or custom patterns.

    Configuration:
        - format_type (str): Expected format - 'json', 'xml', 'markdown', 'plain', 'custom' (default: 'json')
        - strict (bool): Whether to block on validation failure (default: True)
        - custom_pattern (str): Regex pattern for custom format validation
        - extract_json (bool): Try to extract JSON from text if not valid (default: False)
        - required_fields (list): Required fields for JSON validation

    Example:
        >>> guardrail = FormatValidatorGuardrail(config={
        ...     "format_type": "json",
        ...     "strict": True,
        ...     "required_fields": ["name", "age"]
        ... })
        >>> result = guardrail.validate(
        ...     response='{"name": "John", "age": 30}',
        ...     original_prompt="Generate user data"
        ... )
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the format validator guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get configuration
        self.format_type = self.config.get("format_type", "json")
        self.strict = self.config.get("strict", True)
        self.custom_pattern = self.config.get("custom_pattern", None)
        self.extract_json = self.config.get("extract_json", False)
        self.required_fields = self.config.get("required_fields", [])

        # Compile custom pattern if provided
        if self.custom_pattern:
            try:
                self.compiled_pattern = re.compile(self.custom_pattern)
            except re.error:
                self.compiled_pattern = None

    def _validate_json(self, response: str) -> tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate JSON format.

        Returns:
            Tuple of (is_valid, error_message, parsed_data)
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)

            # Check required fields if specified
            if self.required_fields and isinstance(data, dict):
                missing_fields = [f for f in self.required_fields if f not in data]
                if missing_fields:
                    return False, f"Missing required fields: {', '.join(missing_fields)}", data

            return True, None, data

        except json.JSONDecodeError as e:
            # Try to extract JSON if enabled
            if self.extract_json:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        return True, "Extracted JSON from text", data
                    except json.JSONDecodeError:
                        pass

            return False, f"Invalid JSON: {str(e)}", None

    def _validate_xml(self, response: str) -> tuple[bool, Optional[str]]:
        """
        Validate XML format (basic check).

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic XML validation using regex
        if not response.strip().startswith('<'):
            return False, "Response doesn't start with XML tag"

        # Check for matching tags (basic)
        opening_tags = re.findall(r'<(\w+)', response)
        closing_tags = re.findall(r'</(\w+)>', response)

        if len(opening_tags) != len(closing_tags):
            return False, "Mismatched XML tags"

        return True, None

    def _validate_markdown(self, response: str) -> tuple[bool, Optional[str]]:
        """
        Validate markdown format (basic check).

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for markdown indicators
        markdown_indicators = [
            r'^#+\s',  # Headers
            r'\*\*.*\*\*',  # Bold
            r'\*.*\*',  # Italic
            r'\[.*\]\(.*\)',  # Links
            r'^-\s',  # Lists
            r'^```',  # Code blocks
        ]

        has_markdown = any(re.search(pattern, response, re.MULTILINE) for pattern in markdown_indicators)

        if not has_markdown:
            return False, "No markdown formatting detected"

        return True, None

    def _validate_custom(self, response: str) -> tuple[bool, Optional[str]]:
        """
        Validate using custom pattern.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.compiled_pattern:
            return True, "No custom pattern configured"

        if self.compiled_pattern.search(response):
            return True, None

        return False, "Response doesn't match custom pattern"

    def validate(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> GuardrailResult:
        """
        Validate response format.

        Args:
            response: Generated response to validate
            original_prompt: Original user prompt (for context)
            **kwargs: Additional context

        Returns:
            GuardrailResult with validation result
        """
        if not response:
            return GuardrailResult(
                action=GuardrailAction.WARN,
                passed=False,
                message="Empty response received",
                guardrail_name=self.name
            )

        # Validate based on format type
        is_valid = False
        error_message = None
        metadata = {"format_type": self.format_type}

        if self.format_type == "json":
            is_valid, error_message, parsed_data = self._validate_json(response)
            if parsed_data is not None:
                metadata["parsed_data"] = parsed_data
        elif self.format_type == "xml":
            is_valid, error_message = self._validate_xml(response)
        elif self.format_type == "markdown":
            is_valid, error_message = self._validate_markdown(response)
        elif self.format_type == "custom":
            is_valid, error_message = self._validate_custom(response)
        elif self.format_type == "plain":
            # Plain text always passes
            is_valid = True
        else:
            # Unknown format type
            is_valid = True
            metadata["warning"] = f"Unknown format type: {self.format_type}"

        # Determine action
        if is_valid:
            message = f"Valid {self.format_type} format"
            if error_message:  # Extracted JSON case
                message += f" ({error_message})"

            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                passed=True,
                message=message,
                metadata=metadata,
                guardrail_name=self.name
            )
        else:
            if self.strict:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    message=error_message or f"Invalid {self.format_type} format",
                    metadata=metadata,
                    guardrail_name=self.name
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    passed=False,
                    message=f"{error_message or 'Invalid format'} (non-strict mode)",
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
