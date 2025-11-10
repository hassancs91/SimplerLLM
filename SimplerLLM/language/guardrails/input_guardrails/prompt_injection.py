"""
Prompt injection guardrail for adding safety and instruction rules to system prompts.
"""

from typing import Optional, List, Dict
from SimplerLLM.language.guardrails.base import (
    InputGuardrail,
    GuardrailResult,
    GuardrailAction
)


class PromptInjectionGuardrail(InputGuardrail):
    """
    Injects safety rules or instructions into the system prompt.

    This guardrail automatically adds safety rules, ethical guidelines,
    or specific instructions to the system prompt before LLM generation.

    Configuration:
        - safety_rules (str): Rules to inject (default: built-in safety rules)
        - position (str): Where to inject - 'prepend' or 'append' (default: 'prepend')
        - separator (str): Text to separate injected rules from original (default: '\\n\\n')
        - custom_instructions (list): Additional custom instructions to include

    Example:
        >>> guardrail = PromptInjectionGuardrail(config={
        ...     "safety_rules": "Always be helpful and harmless.",
        ...     "position": "prepend"
        ... })
        >>> result = guardrail.validate(
        ...     prompt="Hello",
        ...     system_prompt="You are an assistant"
        ... )
    """

    DEFAULT_SAFETY_RULES = """IMPORTANT SAFETY AND ETHICAL GUIDELINES:
- Do not generate harmful, illegal, or unethical content
- Respect user privacy and do not request sensitive personal information
- Be truthful, accurate, and acknowledge uncertainty when appropriate
- Decline requests that violate ethical guidelines or could cause harm
- Avoid generating content that could be used for malicious purposes"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the prompt injection guardrail.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get configuration
        self.safety_rules = self.config.get("safety_rules", self.DEFAULT_SAFETY_RULES)
        self.position = self.config.get("position", "prepend")  # prepend or append
        self.separator = self.config.get("separator", "\n\n")
        self.custom_instructions = self.config.get("custom_instructions", [])

        # Validate configuration
        if self.position not in ["prepend", "append"]:
            self.position = "prepend"

    def _inject_rules(self, system_prompt: str) -> str:
        """
        Inject rules into system prompt.

        Args:
            system_prompt: Original system prompt

        Returns:
            Modified system prompt with injected rules
        """
        # Combine safety rules with custom instructions
        all_rules = [self.safety_rules]
        if self.custom_instructions:
            all_rules.extend(self.custom_instructions)

        injected_content = self.separator.join(all_rules)

        # Inject based on position
        if self.position == "prepend":
            modified = f"{injected_content}{self.separator}{system_prompt}"
        else:  # append
            modified = f"{system_prompt}{self.separator}{injected_content}"

        return modified

    def validate(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> GuardrailResult:
        """
        Inject safety rules into system prompt.

        Args:
            prompt: User prompt (not modified)
            system_prompt: System prompt to modify
            messages: Optional conversation messages
            **kwargs: Additional context

        Returns:
            GuardrailResult with MODIFY action and modified system prompt
        """
        modified_system = self._inject_rules(system_prompt)

        return GuardrailResult(
            action=GuardrailAction.MODIFY,
            passed=True,
            message="Injected safety rules into system prompt",
            modified_content=modified_system,
            metadata={
                "system": True,
                "target": "system",
                "injection_position": self.position,
                "rules_count": 1 + len(self.custom_instructions)
            },
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
