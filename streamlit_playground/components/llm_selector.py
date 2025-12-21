"""
LLM Selector Component
Reusable provider and model selection dropdowns
"""

import streamlit as st
from SimplerLLM.language import LLMProvider


# Provider to models mapping
PROVIDER_MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "Anthropic": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"],
    "Gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
}

# Provider name to LLMProvider enum mapping
PROVIDER_ENUM_MAP = {
    "OpenAI": LLMProvider.OPENAI,
    "Anthropic": LLMProvider.ANTHROPIC,
    "Gemini": LLMProvider.GEMINI,
}


def get_provider_enum(provider_name: str) -> LLMProvider:
    """Convert provider name to LLMProvider enum."""
    return PROVIDER_ENUM_MAP.get(provider_name, LLMProvider.OPENAI)


def render_llm_selector(
    key_prefix: str,
    allowed_providers: list = None,
    show_model: bool = True,
    label: str = "LLM Provider",
    default_provider: str = "OpenAI",
    default_model: str = None,
    columns: bool = True,
) -> tuple:
    """
    Render LLM provider and model selector.

    Args:
        key_prefix: Unique key prefix for Streamlit widgets
        allowed_providers: List of allowed provider names (default: all)
        show_model: Whether to show model selector
        label: Label for provider selector
        default_provider: Default provider to select
        default_model: Default model to select
        columns: Whether to display in columns

    Returns:
        Tuple of (provider_name, model_name)
    """
    if allowed_providers is None:
        allowed_providers = list(PROVIDER_MODELS.keys())

    # Filter to only allowed providers
    available_providers = [p for p in allowed_providers if p in PROVIDER_MODELS]

    if not available_providers:
        st.error("No valid providers available")
        return None, None

    # Set default provider
    default_idx = 0
    if default_provider in available_providers:
        default_idx = available_providers.index(default_provider)

    if columns and show_model:
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox(
                label,
                options=available_providers,
                index=default_idx,
                key=f"{key_prefix}_provider"
            )
        with col2:
            models = PROVIDER_MODELS.get(provider, [])
            model_idx = 0
            if default_model and default_model in models:
                model_idx = models.index(default_model)
            model = st.selectbox(
                "Model",
                options=models,
                index=model_idx,
                key=f"{key_prefix}_model"
            )
    else:
        provider = st.selectbox(
            label,
            options=available_providers,
            index=default_idx,
            key=f"{key_prefix}_provider"
        )
        if show_model:
            models = PROVIDER_MODELS.get(provider, [])
            model_idx = 0
            if default_model and default_model in models:
                model_idx = models.index(default_model)
            model = st.selectbox(
                "Model",
                options=models,
                index=model_idx,
                key=f"{key_prefix}_model"
            )
        else:
            model = PROVIDER_MODELS.get(provider, [""])[0]

    return provider, model


def render_multi_provider_selector(
    key_prefix: str,
    label: str = "Select Providers",
    default_providers: list = None,
    min_selection: int = 1,
) -> list:
    """
    Render multi-select for providers.

    Returns:
        List of selected (provider_name, model_name) tuples
    """
    if default_providers is None:
        default_providers = ["OpenAI"]

    selected_providers = st.multiselect(
        label,
        options=list(PROVIDER_MODELS.keys()),
        default=default_providers,
        key=f"{key_prefix}_providers"
    )

    if len(selected_providers) < min_selection:
        st.warning(f"Please select at least {min_selection} provider(s)")
        return []

    # For each selected provider, use the default model
    result = []
    for provider in selected_providers:
        model = PROVIDER_MODELS.get(provider, [""])[0]
        result.append((provider, model))

    return result
