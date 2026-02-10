"""
Hugging Face Transformers local model provider for SimplerLLM.

This module provides local inference using Hugging Face Transformers models.
Requires optional dependencies: pip install simplerllm[transformers]

Environment Variables:
    HF_DEVICE: Device to use ('cuda', 'cpu', 'mps', or 'auto'). Default: 'auto'
    HF_TORCH_DTYPE: Data type ('float16', 'bfloat16', 'float32', 'auto'). Default: 'auto'
    HF_LOAD_IN_4BIT: Enable 4-bit quantization ('true'/'false'). Default: 'false'
    HF_LOAD_IN_8BIT: Enable 8-bit quantization ('true'/'false'). Default: 'false'
    HF_TRUST_REMOTE_CODE: Trust remote code for custom models. Default: 'false'
    HF_TIMEOUT: Generation timeout in seconds. Default: 300
"""

from typing import Dict, Optional, List, Any, Union
import os
import time
import asyncio
from dotenv import load_dotenv
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv(override=True)

# Configuration from environment
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", 300))
HF_DEVICE = os.getenv("HF_DEVICE", "auto")
HF_TORCH_DTYPE = os.getenv("HF_TORCH_DTYPE", "auto")
HF_LOAD_IN_4BIT = os.getenv("HF_LOAD_IN_4BIT", "false").lower() == "true"
HF_LOAD_IN_8BIT = os.getenv("HF_LOAD_IN_8BIT", "false").lower() == "true"
HF_TRUST_REMOTE_CODE = os.getenv("HF_TRUST_REMOTE_CODE", "false").lower() == "true"

# Global cache for loaded models to avoid reloading
_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}


def _check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        raise ImportError(
            "Hugging Face Transformers support requires additional dependencies. "
            "Install with: pip install simplerllm[transformers]"
        )


def _get_device() -> str:
    """Determine the best available device."""
    import torch

    device = HF_DEVICE
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _get_torch_dtype():
    """Get the torch dtype from environment variable."""
    import torch

    dtype_str = HF_TORCH_DTYPE
    if dtype_str == "auto":
        # Use bfloat16 on CUDA if available, else float16, else float32
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float32)


def _load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer with lazy loading and caching.

    Args:
        model_name: HuggingFace model ID or local path

    Returns:
        Tuple of (model, tokenizer)
    """
    _check_dependencies()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Check cache first
    if model_name in _model_cache and model_name in _tokenizer_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name]

    device = _get_device()
    torch_dtype = _get_torch_dtype()

    # Prepare model loading kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": HF_TRUST_REMOTE_CODE,
    }

    # Handle quantization
    if HF_LOAD_IN_4BIT or HF_LOAD_IN_8BIT:
        try:
            from transformers import BitsAndBytesConfig

            if HF_LOAD_IN_4BIT:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            elif HF_LOAD_IN_8BIT:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        except ImportError:
            raise ImportError(
                "Quantization requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
    else:
        # Only set device_map if not using quantization (bitsandbytes handles this)
        model_kwargs["device_map"] = device if device != "cpu" else None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    # Move to device if not using device_map
    if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
        model = model.to(device)

    # Cache for reuse
    _model_cache[model_name] = model
    _tokenizer_cache[model_name] = tokenizer

    return model, tokenizer


def _format_messages_with_template(tokenizer, messages: List[dict]) -> str:
    """
    Format messages using the tokenizer's chat template if available.

    Args:
        tokenizer: The HuggingFace tokenizer
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Formatted prompt string
    """
    # Try using the built-in chat template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass  # Fall back to basic formatting

    # Basic fallback format
    formatted_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            formatted_parts.append(f"System: {content}")
        elif role == "user":
            formatted_parts.append(f"User: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")

    formatted_parts.append("Assistant:")
    return "\n\n".join(formatted_parts)


def unload_model(model_name: str = None):
    """
    Unload model(s) from memory to free GPU/RAM.

    Args:
        model_name: Specific model to unload, or None to unload all
    """
    import gc

    if model_name:
        if model_name in _model_cache:
            del _model_cache[model_name]
        if model_name in _tokenizer_cache:
            del _tokenizer_cache[model_name]
    else:
        _model_cache.clear()
        _tokenizer_cache.clear()

    gc.collect()

    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def generate_response(
    model_name: str,
    messages: List[dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    json_mode: bool = False,
    **kwargs,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using a local Hugging Face model.

    Args:
        model_name: HuggingFace model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
                   or local path to a fine-tuned model
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Controls randomness (0.0-2.0). Default 0.7
        max_tokens: Maximum new tokens to generate. Default 300
        top_p: Nucleus sampling parameter. Default 1.0
        full_response: If True, returns LLMFullResponse with metadata
        json_mode: If True, adds JSON instruction to prompt (no strict enforcement)
        **kwargs: Additional generation kwargs (ignored for compatibility)

    Returns:
        Generated text string, or LLMFullResponse if full_response=True

    Raises:
        ImportError: If transformers/torch not installed
        Exception: If model loading or generation fails
    """
    _check_dependencies()
    import torch

    start_time = time.time()

    # Load model and tokenizer (uses cache if already loaded)
    model, tokenizer = _load_model_and_tokenizer(model_name)

    # Format messages into prompt
    prompt = _format_messages_with_template(tokenizer, messages)

    # Add JSON instruction if json_mode is enabled
    if json_mode:
        prompt = prompt.rstrip()
        if not prompt.endswith(":"):
            prompt += "\n\nRespond with valid JSON only:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_length = inputs["input_ids"].shape[1]

    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Prepare generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

    # Decode only the new tokens (exclude input)
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Calculate token counts
    output_length = len(generated_ids)

    process_time = time.time() - start_time

    if full_response:
        return LLMFullResponse(
            generated_text=generated_text,
            model=model_name,
            process_time=process_time,
            input_token_count=input_length,
            output_token_count=output_length,
            llm_provider_response={
                "device": str(device),
                "dtype": str(next(model.parameters()).dtype),
            },
        )

    return generated_text


async def generate_response_async(
    model_name: str,
    messages: List[dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    json_mode: bool = False,
    **kwargs,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using a local Hugging Face model.

    This wraps the synchronous generation in a thread to avoid blocking
    the event loop during model inference.

    Args:
        model_name: HuggingFace model ID or local path
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Controls randomness (0.0-2.0). Default 0.7
        max_tokens: Maximum new tokens to generate. Default 300
        top_p: Nucleus sampling parameter. Default 1.0
        full_response: If True, returns LLMFullResponse with metadata
        json_mode: If True, adds JSON instruction to prompt
        **kwargs: Additional generation kwargs (ignored for compatibility)

    Returns:
        Generated text string, or LLMFullResponse if full_response=True
    """
    # Run synchronous generation in a thread pool to avoid blocking
    return await asyncio.to_thread(
        generate_response,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        full_response=full_response,
        json_mode=json_mode,
        **kwargs,
    )


def get_cached_models() -> List[str]:
    """Return list of currently cached model names."""
    return list(_model_cache.keys())
