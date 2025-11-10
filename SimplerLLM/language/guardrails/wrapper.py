"""
GuardrailsLLM wrapper class that adds guardrails to any LLM instance.

This module provides the main wrapper class that transparently adds
guardrail functionality to any SimplerLLM LLM instance.
"""

from typing import List, Optional, Dict, Any
from SimplerLLM.language.llm.base import LLM
from SimplerLLM.language.guardrails.base import (
    InputGuardrail,
    OutputGuardrail,
    GuardrailResult,
    GuardrailAction
)
from SimplerLLM.language.guardrails.exceptions import (
    GuardrailBlockedException,
    GuardrailValidationException
)
from SimplerLLM.language.llm_providers.llm_response_models import LLMFullResponse


class GuardrailsLLM(LLM):
    """
    Wrapper that adds guardrails to any LLM instance.

    This class wraps an existing LLM instance and applies input and output
    guardrails transparently. It supports all standard LLM methods and can
    be used as a drop-in replacement.

    Example:
        >>> from SimplerLLM.language.llm.base import LLM, LLMProvider
        >>> from SimplerLLM.language.guardrails import GuardrailsLLM, PromptInjectionGuardrail
        >>>
        >>> base_llm = LLM.create(provider=LLMProvider.OPENAI)
        >>> guardrailed_llm = GuardrailsLLM(
        ...     llm_instance=base_llm,
        ...     input_guardrails=[PromptInjectionGuardrail()]
        ... )
        >>> response = guardrailed_llm.generate_response(prompt="Hello!")
    """

    def __init__(
        self,
        llm_instance: LLM,
        input_guardrails: Optional[List[InputGuardrail]] = None,
        output_guardrails: Optional[List[OutputGuardrail]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GuardrailsLLM wrapper.

        Args:
            llm_instance: The LLM instance to wrap
            input_guardrails: List of input guardrails to apply before generation
            output_guardrails: List of output guardrails to apply after generation
            config: Configuration options:
                - fail_fast (bool): Stop on first BLOCK action (default: True)
                - include_metadata (bool): Include guardrail metadata in response (default: True)
                - auto_modify (bool): Automatically apply MODIFY actions (default: True)
                - skip_on_error (bool): Continue if a guardrail fails (default: False)
        """
        # Don't call super().__init__ as we're wrapping, not replacing
        self.llm = llm_instance
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.config = config or {}

        # Configuration options
        self.fail_fast = self.config.get("fail_fast", True)
        self.include_metadata = self.config.get("include_metadata", True)
        self.auto_modify = self.config.get("auto_modify", True)
        self.skip_on_error = self.config.get("skip_on_error", False)

        # Delegate attributes to wrapped LLM
        self.provider = llm_instance.provider
        self.model_name = llm_instance.model_name
        self.temperature = llm_instance.temperature
        self.top_p = llm_instance.top_p
        self.api_key = getattr(llm_instance, 'api_key', None)
        self.verbose = getattr(llm_instance, 'verbose', False)

    def _apply_input_guardrails(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> tuple[str, str, List[GuardrailResult]]:
        """
        Apply all input guardrails.

        Returns:
            Tuple of (modified_prompt, modified_system_prompt, results)
        """
        modified_prompt = prompt
        modified_system = system_prompt
        results = []

        for guardrail in self.input_guardrails:
            if not guardrail.enabled:
                continue

            try:
                result = guardrail.validate(
                    prompt=modified_prompt if modified_prompt else "",
                    system_prompt=modified_system,
                    messages=messages,
                    **kwargs
                )
                results.append(result)

                # Handle guardrail result
                if result.action == GuardrailAction.BLOCK:
                    if self.fail_fast:
                        raise GuardrailBlockedException(
                            f"Input blocked by {guardrail.name}: {result.message}",
                            guardrail_name=guardrail.name,
                            metadata=result.metadata
                        )
                elif result.action == GuardrailAction.MODIFY and self.auto_modify:
                    if result.modified_content:
                        # Check metadata to see what was modified
                        if result.metadata.get("system") or result.metadata.get("target") == "system":
                            modified_system = result.modified_content
                        else:
                            modified_prompt = result.modified_content

            except GuardrailBlockedException:
                raise
            except Exception as e:
                if self.skip_on_error:
                    results.append(GuardrailResult(
                        action=GuardrailAction.WARN,
                        passed=False,
                        message=f"Guardrail error: {str(e)}",
                        guardrail_name=guardrail.name
                    ))
                else:
                    raise GuardrailValidationException(
                        f"Error in {guardrail.name}: {str(e)}",
                        guardrail_name=guardrail.name,
                        original_exception=e
                    )

        return modified_prompt, modified_system, results

    async def _apply_input_guardrails_async(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ) -> tuple[str, str, List[GuardrailResult]]:
        """
        Apply all input guardrails asynchronously.

        Returns:
            Tuple of (modified_prompt, modified_system_prompt, results)
        """
        modified_prompt = prompt
        modified_system = system_prompt
        results = []

        for guardrail in self.input_guardrails:
            if not guardrail.enabled:
                continue

            try:
                result = await guardrail.validate_async(
                    prompt=modified_prompt if modified_prompt else "",
                    system_prompt=modified_system,
                    messages=messages,
                    **kwargs
                )
                results.append(result)

                # Handle guardrail result
                if result.action == GuardrailAction.BLOCK:
                    if self.fail_fast:
                        raise GuardrailBlockedException(
                            f"Input blocked by {guardrail.name}: {result.message}",
                            guardrail_name=guardrail.name,
                            metadata=result.metadata
                        )
                elif result.action == GuardrailAction.MODIFY and self.auto_modify:
                    if result.modified_content:
                        # Check metadata to see what was modified
                        if result.metadata.get("system") or result.metadata.get("target") == "system":
                            modified_system = result.modified_content
                        else:
                            modified_prompt = result.modified_content

            except GuardrailBlockedException:
                raise
            except Exception as e:
                if self.skip_on_error:
                    results.append(GuardrailResult(
                        action=GuardrailAction.WARN,
                        passed=False,
                        message=f"Guardrail error: {str(e)}",
                        guardrail_name=guardrail.name
                    ))
                else:
                    raise GuardrailValidationException(
                        f"Error in {guardrail.name}: {str(e)}",
                        guardrail_name=guardrail.name,
                        original_exception=e
                    )

        return modified_prompt, modified_system, results

    def _apply_output_guardrails(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> tuple[str, List[GuardrailResult]]:
        """
        Apply all output guardrails.

        Returns:
            Tuple of (modified_response, results)
        """
        modified_response = response
        results = []

        for guardrail in self.output_guardrails:
            if not guardrail.enabled:
                continue

            try:
                result = guardrail.validate(
                    response=modified_response,
                    original_prompt=original_prompt,
                    **kwargs
                )
                results.append(result)

                # Handle guardrail result
                if result.action == GuardrailAction.BLOCK:
                    if self.fail_fast:
                        raise GuardrailBlockedException(
                            f"Output blocked by {guardrail.name}: {result.message}",
                            guardrail_name=guardrail.name,
                            metadata=result.metadata
                        )
                elif result.action == GuardrailAction.MODIFY and self.auto_modify:
                    if result.modified_content:
                        modified_response = result.modified_content

            except GuardrailBlockedException:
                raise
            except Exception as e:
                if self.skip_on_error:
                    results.append(GuardrailResult(
                        action=GuardrailAction.WARN,
                        passed=False,
                        message=f"Guardrail error: {str(e)}",
                        guardrail_name=guardrail.name
                    ))
                else:
                    raise GuardrailValidationException(
                        f"Error in {guardrail.name}: {str(e)}",
                        guardrail_name=guardrail.name,
                        original_exception=e
                    )

        return modified_response, results

    async def _apply_output_guardrails_async(
        self,
        response: str,
        original_prompt: str = "",
        **kwargs
    ) -> tuple[str, List[GuardrailResult]]:
        """
        Apply all output guardrails asynchronously.

        Returns:
            Tuple of (modified_response, results)
        """
        modified_response = response
        results = []

        for guardrail in self.output_guardrails:
            if not guardrail.enabled:
                continue

            try:
                result = await guardrail.validate_async(
                    response=modified_response,
                    original_prompt=original_prompt,
                    **kwargs
                )
                results.append(result)

                # Handle guardrail result
                if result.action == GuardrailAction.BLOCK:
                    if self.fail_fast:
                        raise GuardrailBlockedException(
                            f"Output blocked by {guardrail.name}: {result.message}",
                            guardrail_name=guardrail.name,
                            metadata=result.metadata
                        )
                elif result.action == GuardrailAction.MODIFY and self.auto_modify:
                    if result.modified_content:
                        modified_response = result.modified_content

            except GuardrailBlockedException:
                raise
            except Exception as e:
                if self.skip_on_error:
                    results.append(GuardrailResult(
                        action=GuardrailAction.WARN,
                        passed=False,
                        message=f"Guardrail error: {str(e)}",
                        guardrail_name=guardrail.name
                    ))
                else:
                    raise GuardrailValidationException(
                        f"Error in {guardrail.name}: {str(e)}",
                        guardrail_name=guardrail.name,
                        original_exception=e
                    )

        return modified_response, results

    def _create_metadata(
        self,
        input_results: List[GuardrailResult],
        output_results: List[GuardrailResult],
        original_prompt: str,
        modified_prompt: str,
        original_system: str,
        modified_system: str,
        original_response: str,
        modified_response: str
    ) -> Dict[str, Any]:
        """Create guardrails metadata dictionary."""
        return {
            "input_guardrails": [r.to_dict() for r in input_results],
            "output_guardrails": [r.to_dict() for r in output_results],
            "prompt_modified": modified_prompt != original_prompt,
            "system_modified": modified_system != original_system,
            "response_modified": modified_response != original_response,
            "total_guardrails": len(self.input_guardrails) + len(self.output_guardrails),
            "guardrails_passed": all(r.passed for r in input_results + output_results)
        }

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
    ):
        """
        Generate response with guardrails applied.

        This method applies input guardrails, calls the wrapped LLM,
        applies output guardrails, and returns the result with metadata.

        Args:
            model_name: Model name to use
            prompt: User prompt
            messages: List of conversation messages
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            top_p: Top-p parameter
            full_response: Return full response object with metadata
            **kwargs: Additional parameters passed to wrapped LLM

        Returns:
            Generated response (string or LLMFullResponse)

        Raises:
            GuardrailBlockedException: If a guardrail blocks the request/response
            GuardrailValidationException: If a guardrail fails unexpectedly
        """
        original_prompt = prompt
        original_system = system_prompt

        # 1. Apply input guardrails
        modified_prompt, modified_system, input_results = self._apply_input_guardrails(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs
        )

        # 2. Call underlying LLM (always get full response internally)
        response = self.llm.generate_response(
            model_name=model_name,
            prompt=modified_prompt,
            messages=messages,
            system_prompt=modified_system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            full_response=True,  # Always get full response for metadata
            **kwargs
        )

        # Extract response text (handle both string and object responses)
        if isinstance(response, LLMFullResponse):
            response_text = response.generated_text
        elif isinstance(response, dict):
            response_text = response.get('generated_text', str(response))
        else:
            response_text = str(response)

        original_response = response_text

        # 3. Apply output guardrails
        modified_response, output_results = self._apply_output_guardrails(
            response=response_text,
            original_prompt=original_prompt or "",
            **kwargs
        )

        # 4. Prepare response with metadata
        if full_response or self.include_metadata:
            # Create or enhance response object
            if isinstance(response, LLMFullResponse):
                response_obj = response
                response_obj.generated_text = modified_response
            else:
                # Create a minimal response object
                response_obj = LLMFullResponse(
                    generated_text=modified_response,
                    model=model_name or self.model_name or "unknown",
                    process_time=0.0,
                    llm_provider_response=response
                )

            # Add guardrails metadata
            response_obj.guardrails_metadata = self._create_metadata(
                input_results=input_results,
                output_results=output_results,
                original_prompt=original_prompt or "",
                modified_prompt=modified_prompt or "",
                original_system=original_system,
                modified_system=modified_system,
                original_response=original_response,
                modified_response=modified_response
            )

            return response_obj

        return modified_response

    async def generate_response_async(
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
    ):
        """
        Async version of generate_response with guardrails.

        See generate_response for full documentation.
        """
        original_prompt = prompt
        original_system = system_prompt

        # 1. Apply input guardrails
        modified_prompt, modified_system, input_results = await self._apply_input_guardrails_async(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs
        )

        # 2. Call underlying LLM (always get full response internally)
        response = await self.llm.generate_response_async(
            model_name=model_name,
            prompt=modified_prompt,
            messages=messages,
            system_prompt=modified_system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            full_response=True,
            **kwargs
        )

        # Extract response text
        if isinstance(response, LLMFullResponse):
            response_text = response.generated_text
        elif isinstance(response, dict):
            response_text = response.get('generated_text', str(response))
        else:
            response_text = str(response)

        original_response = response_text

        # 3. Apply output guardrails
        modified_response, output_results = await self._apply_output_guardrails_async(
            response=response_text,
            original_prompt=original_prompt or "",
            **kwargs
        )

        # 4. Prepare response with metadata
        if full_response or self.include_metadata:
            # Create or enhance response object
            if isinstance(response, LLMFullResponse):
                response_obj = response
                response_obj.generated_text = modified_response
            else:
                # Create a minimal response object
                response_obj = LLMFullResponse(
                    generated_text=modified_response,
                    model=model_name or self.model_name or "unknown",
                    process_time=0.0,
                    llm_provider_response=response
                )

            # Add guardrails metadata
            response_obj.guardrails_metadata = self._create_metadata(
                input_results=input_results,
                output_results=output_results,
                original_prompt=original_prompt or "",
                modified_prompt=modified_prompt or "",
                original_system=original_system,
                modified_system=modified_system,
                original_response=original_response,
                modified_response=modified_response
            )

            return response_obj

        return modified_response

    def append_messages(self, system_prompt: str, messages: list):
        """Delegate to wrapped LLM."""
        return self.llm.append_messages(system_prompt, messages)

    def prepare_params(self, model_name, temperature, top_p):
        """Delegate to wrapped LLM."""
        return self.llm.prepare_params(model_name, temperature, top_p)

    def set_model(self, provider):
        """Delegate to wrapped LLM."""
        return self.llm.set_model(provider)
