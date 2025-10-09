"""
Mini Agent Flow System

Inspired by n8n/Make.com/Zapier - Linear workflow execution without conditions/loops.
"""

import time
import asyncio
from typing import Any, Dict, List, Optional, Union, Type
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_async
)
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import StepResult, FlowResult
from .tool_registry import ToolRegistry


class MiniAgent:
    """
    A mini-agent that executes a predefined flow of steps.

    Each flow consists of sequential steps that can be:
    - LLM steps: Use the LLM to generate text/responses
    - Tool steps: Execute tools from the tool registry

    Steps are executed linearly, with each step's output becoming the next step's input.
    """

    def __init__(
        self,
        name: str,
        llm_instance: LLM,
        system_prompt: str = "You are a helpful AI assistant.",
        max_steps: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize a Mini Agent.

        Args:
            name: Name of this agent
            llm_instance: An LLM instance created via LLM.create()
            system_prompt: System prompt for LLM steps
            max_steps: Maximum number of steps allowed (default: 3)
            verbose: Whether to print detailed execution logs
        """
        self.name = name
        self.llm_instance = llm_instance
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.verbose = verbose
        self.steps: List[Dict[str, Any]] = []

        if self.verbose:
            verbose_print(f"Initialized MiniAgent: {name}", "info")
            verbose_print(f"Max steps: {max_steps}", "debug")

    def add_step(
        self,
        step_type: str,
        tool_name: Optional[str] = None,
        prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        max_retries: int = 3,
    ):
        """
        Add a step to the flow.

        Args:
            step_type: Type of step - "llm" or "tool"
            tool_name: Name of the tool (required if step_type is "tool")
            prompt: Prompt for LLM (required if step_type is "llm")
            params: Additional parameters for the tool/LLM
            output_model: Optional Pydantic model for structured JSON output (only for "llm" steps)
            max_retries: Maximum retries for JSON validation (default: 3, only used with output_model)

        Raises:
            ValueError: If step configuration is invalid or max_steps is exceeded
        """
        if len(self.steps) >= self.max_steps:
            raise ValueError(
                f"Cannot add more than {self.max_steps} steps to this flow. "
                f"Current steps: {len(self.steps)}"
            )

        step_type = step_type.lower()
        if step_type not in ["llm", "tool"]:
            raise ValueError(f"Invalid step_type: {step_type}. Must be 'llm' or 'tool'")

        if step_type == "tool" and not tool_name:
            raise ValueError("tool_name is required for 'tool' step type")

        if step_type == "llm" and not prompt:
            raise ValueError("prompt is required for 'llm' step type")

        if output_model and step_type != "llm":
            raise ValueError("output_model can only be used with 'llm' step type")

        # Validate tool exists
        if step_type == "tool":
            ToolRegistry.get_tool(tool_name)  # This will raise ValueError if not found

        step = {
            "type": step_type,
            "tool_name": tool_name,
            "prompt": prompt,
            "params": params or {},
            "output_model": output_model,
            "max_retries": max_retries,
        }

        self.steps.append(step)

        if self.verbose:
            json_mode_str = " (JSON mode)" if output_model else ""
            verbose_print(
                f"Added step {len(self.steps)}: {step_type}" +
                (f" - {tool_name}" if tool_name else f" - LLM{json_mode_str}"),
                "debug"
            )

    def run(self, user_input: Any) -> FlowResult:
        """
        Execute the flow with the given input.

        Args:
            user_input: Initial input to the flow

        Returns:
            FlowResult containing all step results and final output
        """
        if not self.steps:
            raise ValueError("No steps defined. Add at least one step before running.")

        if self.verbose:
            verbose_print(f"\n{'='*60}", "info")
            verbose_print(f"Starting flow execution: {self.name}", "info")
            verbose_print(f"Initial input: {user_input}", "debug")
            verbose_print(f"{'='*60}\n", "info")

        flow_start_time = time.time()
        step_results = []
        current_data = user_input
        flow_success = True
        flow_error = None

        for idx, step_config in enumerate(self.steps):
            step_number = idx + 1

            if self.verbose:
                verbose_print(f"\n--- Step {step_number}/{len(self.steps)} ---", "info")
                verbose_print(f"Type: {step_config['type']}", "debug")

            try:
                step_result = self._execute_step(
                    step_number=step_number,
                    step_config=step_config,
                    input_data=current_data,
                )
                step_results.append(step_result)

                if step_result.error:
                    flow_success = False
                    flow_error = f"Step {step_number} failed: {step_result.error}"
                    if self.verbose:
                        verbose_print(f"Error in step {step_number}: {step_result.error}", "error")
                    break

                # Use this step's output as next step's input
                current_data = step_result.output_data

                if self.verbose:
                    verbose_print(f"Output: {step_result.output_data}", "debug")
                    verbose_print(f"Duration: {step_result.duration_seconds:.2f}s", "debug")

            except Exception as e:
                error_msg = str(e)
                flow_success = False
                flow_error = f"Step {step_number} failed: {error_msg}"

                # Create error step result
                step_result = StepResult(
                    step_number=step_number,
                    step_type=step_config["type"],
                    input_data=current_data,
                    output_data=None,
                    duration_seconds=0.0,
                    tool_used=step_config.get("tool_name"),
                    prompt_used=step_config.get("prompt"),
                    error=error_msg,
                )
                step_results.append(step_result)

                if self.verbose:
                    verbose_print(f"Exception in step {step_number}: {error_msg}", "error")
                break

        flow_end_time = time.time()
        total_duration = flow_end_time - flow_start_time

        flow_result = FlowResult(
            agent_name=self.name,
            total_steps=len(step_results),
            steps=step_results,
            total_duration_seconds=total_duration,
            final_output=current_data if flow_success else None,
            success=flow_success,
            error=flow_error,
        )

        if self.verbose:
            verbose_print(f"\n{'='*60}", "info")
            verbose_print(f"Flow completed: {self.name}", "info")
            verbose_print(f"Status: {'SUCCESS' if flow_success else 'FAILED'}", "info" if flow_success else "error")
            verbose_print(f"Total duration: {total_duration:.2f}s", "info")
            verbose_print(f"{'='*60}\n", "info")

        return flow_result

    def _execute_step(
        self, step_number: int, step_config: Dict[str, Any], input_data: Any
    ) -> StepResult:
        """
        Execute a single step in the flow.

        Args:
            step_number: The step number (1-indexed)
            step_config: Configuration for this step
            input_data: Input data for this step

        Returns:
            StepResult containing the execution result
        """
        step_start_time = time.time()
        step_type = step_config["type"]

        try:
            if step_type == "tool":
                output_data = self._execute_tool_step(step_config, input_data)
                tool_used = step_config["tool_name"]
                prompt_used = None
                output_model_class = None
            else:  # llm
                output_data = self._execute_llm_step(step_config, input_data)
                tool_used = None
                prompt_used = step_config["prompt"]
                output_model = step_config.get("output_model")
                output_model_class = output_model.__name__ if output_model else None

            step_end_time = time.time()
            duration = step_end_time - step_start_time

            return StepResult(
                step_number=step_number,
                step_type=step_type,
                input_data=input_data,
                output_data=output_data,
                duration_seconds=duration,
                tool_used=tool_used,
                prompt_used=prompt_used,
                output_model_class=output_model_class,
                error=None,
            )

        except Exception as e:
            step_end_time = time.time()
            duration = step_end_time - step_start_time

            return StepResult(
                step_number=step_number,
                step_type=step_type,
                input_data=input_data,
                output_data=None,
                duration_seconds=duration,
                tool_used=step_config.get("tool_name"),
                prompt_used=step_config.get("prompt"),
                output_model_class=None,
                error=str(e),
            )

    def _execute_tool_step(self, step_config: Dict[str, Any], input_data: Any) -> Any:
        """Execute a tool step."""
        tool_name = step_config["tool_name"]
        params = step_config.get("params", {})

        if self.verbose:
            verbose_print(f"Executing tool: {tool_name}", "debug")
            if params:
                verbose_print(f"Tool params: {params}", "debug")

        tool_func = ToolRegistry.get_tool(tool_name)

        # If the tool takes a single argument and no params are provided,
        # pass the input_data directly
        if not params:
            result = tool_func(input_data)
        else:
            # If params are provided, use them and optionally include input_data
            # Check if the params have an 'input' key, if not, assume first arg is input_data
            if 'input' in params:
                result = tool_func(**params)
            else:
                result = tool_func(input_data, **params)

        return result

    def _execute_llm_step(self, step_config: Dict[str, Any], input_data: Any) -> Union[str, BaseModel]:
        """Execute an LLM step, optionally with structured JSON output."""
        prompt_template = step_config["prompt"]
        params = step_config.get("params", {})
        output_model = step_config.get("output_model")
        max_retries = step_config.get("max_retries", 3)

        # Replace {previous_output} or {input} placeholder with actual input
        if "{previous_output}" in prompt_template:
            final_prompt = prompt_template.replace("{previous_output}", str(input_data))
        elif "{input}" in prompt_template:
            final_prompt = prompt_template.replace("{input}", str(input_data))
        else:
            # If no placeholder, append input to prompt
            final_prompt = f"{prompt_template}\n\nInput: {input_data}"

        if self.verbose:
            verbose_print(f"LLM Prompt: {final_prompt[:200]}...", "debug")
            if output_model:
                verbose_print(f"JSON Mode: {output_model.__name__}", "debug")

        # Get max_tokens from params or use default
        max_tokens = params.get("max_tokens", 500)
        temperature = params.get("temperature", self.llm_instance.temperature)
        top_p = params.get("top_p", self.llm_instance.top_p)

        # Check if we need structured JSON output
        if output_model:
            # Use generate_pydantic_json_model for validated JSON output
            if self.verbose:
                verbose_print(f"Generating structured JSON with {max_retries} retries", "debug")

            response = generate_pydantic_json_model(
                model_class=output_model,
                prompt=final_prompt,
                llm_instance=self.llm_instance,
                max_retries=max_retries,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=self.system_prompt,
                full_response=False,
            )

            # Check if response is an error string
            if isinstance(response, str):
                raise ValueError(f"JSON generation failed: {response}")

            return response
        else:
            # Regular text generation
            response = self.llm_instance.generate_response(
                prompt=final_prompt,
                system_prompt=self.system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            return response

    def clear_steps(self):
        """Clear all steps from the flow."""
        self.steps = []
        if self.verbose:
            verbose_print(f"Cleared all steps from {self.name}", "debug")

    def get_step_count(self) -> int:
        """Get the number of steps in the flow."""
        return len(self.steps)

    # ========================================================================
    # ASYNC METHODS
    # ========================================================================

    async def run_async(self, user_input: Any) -> FlowResult:
        """
        Execute the flow asynchronously with the given input.

        Args:
            user_input: Initial input to the flow

        Returns:
            FlowResult containing all step results and final output
        """
        if not self.steps:
            raise ValueError("No steps defined. Add at least one step before running.")

        if self.verbose:
            verbose_print(f"\n{'='*60}", "info")
            verbose_print(f"Starting async flow execution: {self.name}", "info")
            verbose_print(f"Initial input: {user_input}", "debug")
            verbose_print(f"{'='*60}\n", "info")

        flow_start_time = time.time()
        step_results = []
        current_data = user_input
        flow_success = True
        flow_error = None

        for idx, step_config in enumerate(self.steps):
            step_number = idx + 1

            if self.verbose:
                verbose_print(f"\n--- Step {step_number}/{len(self.steps)} ---", "info")
                verbose_print(f"Type: {step_config['type']}", "debug")

            try:
                step_result = await self._execute_step_async(
                    step_number=step_number,
                    step_config=step_config,
                    input_data=current_data,
                )
                step_results.append(step_result)

                if step_result.error:
                    flow_success = False
                    flow_error = f"Step {step_number} failed: {step_result.error}"
                    if self.verbose:
                        verbose_print(f"Error in step {step_number}: {step_result.error}", "error")
                    break

                # Use this step's output as next step's input
                current_data = step_result.output_data

                if self.verbose:
                    verbose_print(f"Output: {step_result.output_data}", "debug")
                    verbose_print(f"Duration: {step_result.duration_seconds:.2f}s", "debug")

            except Exception as e:
                error_msg = str(e)
                flow_success = False
                flow_error = f"Step {step_number} failed: {error_msg}"

                # Create error step result
                step_result = StepResult(
                    step_number=step_number,
                    step_type=step_config["type"],
                    input_data=current_data,
                    output_data=None,
                    duration_seconds=0.0,
                    tool_used=step_config.get("tool_name"),
                    prompt_used=step_config.get("prompt"),
                    output_model_class=None,
                    error=error_msg,
                )
                step_results.append(step_result)

                if self.verbose:
                    verbose_print(f"Exception in step {step_number}: {error_msg}", "error")
                break

        flow_end_time = time.time()
        total_duration = flow_end_time - flow_start_time

        flow_result = FlowResult(
            agent_name=self.name,
            total_steps=len(step_results),
            steps=step_results,
            total_duration_seconds=total_duration,
            final_output=current_data if flow_success else None,
            success=flow_success,
            error=flow_error,
        )

        if self.verbose:
            verbose_print(f"\n{'='*60}", "info")
            verbose_print(f"Async flow completed: {self.name}", "info")
            verbose_print(f"Status: {'SUCCESS' if flow_success else 'FAILED'}", "info" if flow_success else "error")
            verbose_print(f"Total duration: {total_duration:.2f}s", "info")
            verbose_print(f"{'='*60}\n", "info")

        return flow_result

    async def _execute_step_async(
        self, step_number: int, step_config: Dict[str, Any], input_data: Any
    ) -> StepResult:
        """
        Execute a single step in the flow asynchronously.

        Args:
            step_number: The step number (1-indexed)
            step_config: Configuration for this step
            input_data: Input data for this step

        Returns:
            StepResult containing the execution result
        """
        step_start_time = time.time()
        step_type = step_config["type"]

        try:
            if step_type == "tool":
                output_data = await self._execute_tool_step_async(step_config, input_data)
                tool_used = step_config["tool_name"]
                prompt_used = None
                output_model_class = None
            else:  # llm
                output_data = await self._execute_llm_step_async(step_config, input_data)
                tool_used = None
                prompt_used = step_config["prompt"]
                output_model = step_config.get("output_model")
                output_model_class = output_model.__name__ if output_model else None

            step_end_time = time.time()
            duration = step_end_time - step_start_time

            return StepResult(
                step_number=step_number,
                step_type=step_type,
                input_data=input_data,
                output_data=output_data,
                duration_seconds=duration,
                tool_used=tool_used,
                prompt_used=prompt_used,
                output_model_class=output_model_class,
                error=None,
            )

        except Exception as e:
            step_end_time = time.time()
            duration = step_end_time - step_start_time

            return StepResult(
                step_number=step_number,
                step_type=step_type,
                input_data=input_data,
                output_data=None,
                duration_seconds=duration,
                tool_used=step_config.get("tool_name"),
                prompt_used=step_config.get("prompt"),
                output_model_class=None,
                error=str(e),
            )

    async def _execute_tool_step_async(self, step_config: Dict[str, Any], input_data: Any) -> Any:
        """Execute a tool step asynchronously (most tools are sync, so we run them in executor)."""
        tool_name = step_config["tool_name"]
        params = step_config.get("params", {})

        if self.verbose:
            verbose_print(f"Executing tool: {tool_name}", "debug")
            if params:
                verbose_print(f"Tool params: {params}", "debug")

        tool_func = ToolRegistry.get_tool(tool_name)

        # Most tools are synchronous, so we run them in an executor
        # to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        if not params:
            result = await loop.run_in_executor(None, tool_func, input_data)
        else:
            # If params are provided, use them and optionally include input_data
            if 'input' in params:
                result = await loop.run_in_executor(None, lambda: tool_func(**params))
            else:
                result = await loop.run_in_executor(None, lambda: tool_func(input_data, **params))

        return result

    async def _execute_llm_step_async(self, step_config: Dict[str, Any], input_data: Any) -> Union[str, BaseModel]:
        """Execute an LLM step asynchronously, optionally with structured JSON output."""
        prompt_template = step_config["prompt"]
        params = step_config.get("params", {})
        output_model = step_config.get("output_model")
        max_retries = step_config.get("max_retries", 3)

        # Replace {previous_output} or {input} placeholder with actual input
        if "{previous_output}" in prompt_template:
            final_prompt = prompt_template.replace("{previous_output}", str(input_data))
        elif "{input}" in prompt_template:
            final_prompt = prompt_template.replace("{input}", str(input_data))
        else:
            # If no placeholder, append input to prompt
            final_prompt = f"{prompt_template}\n\nInput: {input_data}"

        if self.verbose:
            verbose_print(f"LLM Prompt: {final_prompt[:200]}...", "debug")
            if output_model:
                verbose_print(f"JSON Mode (async): {output_model.__name__}", "debug")

        # Get max_tokens from params or use default
        max_tokens = params.get("max_tokens", 500)
        temperature = params.get("temperature", self.llm_instance.temperature)
        top_p = params.get("top_p", self.llm_instance.top_p)

        # Check if we need structured JSON output
        if output_model:
            # Use generate_pydantic_json_model_async for validated JSON output
            if self.verbose:
                verbose_print(f"Generating structured JSON with {max_retries} retries (async)", "debug")

            response = await generate_pydantic_json_model_async(
                model_class=output_model,
                prompt=final_prompt,
                llm_instance=self.llm_instance,
                max_retries=max_retries,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=self.system_prompt,
                full_response=False,
            )

            # Check if response is an error string
            if isinstance(response, str):
                raise ValueError(f"JSON generation failed: {response}")

            return response
        else:
            # Regular text generation (async)
            response = await self.llm_instance.generate_response_async(
                prompt=final_prompt,
                system_prompt=self.system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            return response
