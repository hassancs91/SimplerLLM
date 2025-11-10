"""
Topic filter guardrail for blocking or warning about prohibited topics.
"""

import re
from typing import Optional, List, Dict
from SimplerLLM.language.guardrails.base import (
    InputGuardrail,
    GuardrailResult,
    GuardrailAction
)


class TopicFilterGuardrail(InputGuardrail):
    """
    Filters prompts based on prohibited topics or keywords.

    This guardrail checks user prompts for prohibited topics, keywords,
    or patterns and can block, warn, or allow with modification.

    Configuration:
        - prohibited_topics (list): List of prohibited topic keywords
        - prohibited_patterns (list): List of regex patterns to block
        - action_on_match (str): Action to take - 'block', 'warn', or 'modify' (default: 'block')
        - case_sensitive (bool): Whether matching is case-sensitive (default: False)
        - match_whole_words (bool): Only match complete words (default: True)
        - custom_message (str): Custom message when content is blocked

    Example:
        >>> guardrail = TopicFilterGuardrail(config={
        ...     "prohibited_topics": ["violence", "illegal activities"],
        ...     "action_on_match": "block"
        ... })
        >>> result = guardrail.validate(
        ...     prompt="How to make illegal substances",
        ...     system_prompt="You are helpful"
        ... )
    """

    DEFAULT_PROHIBITED_TOPICS = [
        "illegal activities",
        "harmful instructions",
        "explicit violence",
        "hate speech",
    ]

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the topic filter guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get configuration
        self.prohibited_topics = self.config.get("prohibited_topics", self.DEFAULT_PROHIBITED_TOPICS)
        self.prohibited_patterns = self.config.get("prohibited_patterns", [])
        self.action_on_match = self.config.get("action_on_match", "block")
        self.case_sensitive = self.config.get("case_sensitive", False)
        self.match_whole_words = self.config.get("match_whole_words", True)
        self.custom_message = self.config.get("custom_message", None)

        # Validate action
        if self.action_on_match not in ["block", "warn", "modify"]:
            self.action_on_match = "block"

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns = []

        # Add patterns from prohibited topics
        for topic in self.prohibited_topics:
            if self.match_whole_words:
                pattern = r'\b' + re.escape(topic) + r'\b'
            else:
                pattern = re.escape(topic)

            flags = 0 if self.case_sensitive else re.IGNORECASE
            self.compiled_patterns.append(re.compile(pattern, flags))

        # Add custom patterns
        for pattern_str in self.prohibited_patterns:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                self.compiled_patterns.append(re.compile(pattern_str, flags))
            except re.error:
                # Skip invalid patterns
                pass

    def _check_for_matches(self, text: str) -> List[Dict]:
        """
        Check text for prohibited topics/patterns.

        Args:
            text: Text to check

        Returns:
            List of matches with details
        """
        matches = []

        for pattern in self.compiled_patterns:
            found = pattern.finditer(text)
            for match in found:
                matches.append({
                    "matched_text": match.group(),
                    "position": match.span(),
                    "pattern": pattern.pattern
                })

        return matches

    def validate(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> GuardrailResult:
        """
        Check prompt for prohibited topics.

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

        # Check for matches
        matches = self._check_for_matches(prompt)

        if matches:
            # Prohibited content found
            matched_texts = [m["matched_text"] for m in matches]
            message = self.custom_message or f"Prohibited topics detected: {', '.join(set(matched_texts))}"

            if self.action_on_match == "block":
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    message=message,
                    metadata={
                        "matches": matches,
                        "match_count": len(matches),
                        "unique_matches": list(set(matched_texts))
                    },
                    guardrail_name=self.name
                )
            elif self.action_on_match == "warn":
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    passed=False,
                    message=message + " (warning only)",
                    metadata={
                        "matches": matches,
                        "match_count": len(matches)
                    },
                    guardrail_name=self.name
                )
            else:  # modify
                # Remove matched content
                modified_prompt = prompt
                for match in sorted(matches, key=lambda m: m["position"][0], reverse=True):
                    start, end = match["position"]
                    modified_prompt = modified_prompt[:start] + "[REMOVED]" + modified_prompt[end:]

                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    passed=False,
                    message="Prohibited content removed from prompt",
                    modified_content=modified_prompt,
                    metadata={
                        "matches": matches,
                        "match_count": len(matches)
                    },
                    guardrail_name=self.name
                )

        # No matches - allow
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            passed=True,
            message="No prohibited topics detected",
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
