"""
API Key Manager Component
Handles API key input and environment variable fallback
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Provider to environment variable mapping
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}


def get_api_key(provider: str) -> str:
    """
    Get API key for a provider.
    Priority: Session state > Environment variable
    """
    provider_lower = provider.lower()
    session_key = f"{provider_lower}_api_key"

    # Check session state first
    if session_key in st.session_state and st.session_state[session_key]:
        return st.session_state[session_key]

    # Fallback to environment variable
    env_key = PROVIDER_ENV_KEYS.get(provider_lower, f"{provider.upper()}_API_KEY")
    return os.getenv(env_key, "")


def render_api_key_manager():
    """Render the API key manager in the sidebar."""

    with st.sidebar.expander("API Keys Configuration", expanded=False):
        st.caption("Keys are stored in session only (not saved to disk)")

        # Initialize session state if needed
        for provider in ["openai", "anthropic", "gemini", "perplexity"]:
            session_key = f"{provider}_api_key"
            if session_key not in st.session_state:
                st.session_state[session_key] = ""

        # OpenAI
        openai_env = os.getenv("OPENAI_API_KEY", "")
        openai_value = st.session_state.get("openai_api_key") or openai_env
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=openai_value,
            type="password",
            key="openai_input",
            help="Get your key at platform.openai.com"
        )

        # Anthropic
        anthropic_env = os.getenv("ANTHROPIC_API_KEY", "")
        anthropic_value = st.session_state.get("anthropic_api_key") or anthropic_env
        st.session_state.anthropic_api_key = st.text_input(
            "Anthropic API Key",
            value=anthropic_value,
            type="password",
            key="anthropic_input",
            help="Get your key at console.anthropic.com"
        )

        # Gemini
        gemini_env = os.getenv("GOOGLE_API_KEY", "")
        gemini_value = st.session_state.get("gemini_api_key") or gemini_env
        st.session_state.gemini_api_key = st.text_input(
            "Google Gemini API Key",
            value=gemini_value,
            type="password",
            key="gemini_input",
            help="Get your key at makersuite.google.com"
        )

        # Perplexity
        perplexity_env = os.getenv("PERPLEXITY_API_KEY", "")
        perplexity_value = st.session_state.get("perplexity_api_key") or perplexity_env
        st.session_state.perplexity_api_key = st.text_input(
            "Perplexity API Key",
            value=perplexity_value,
            type="password",
            key="perplexity_input",
            help="Required for web search features"
        )

        # Status indicators
        st.markdown("---")
        st.caption("Status:")

        providers_status = {
            "OpenAI": bool(get_api_key("openai")),
            "Anthropic": bool(get_api_key("anthropic")),
            "Gemini": bool(get_api_key("gemini")),
            "Perplexity": bool(get_api_key("perplexity")),
        }

        for provider, configured in providers_status.items():
            if configured:
                st.markdown(f"- {provider}: :green[Configured]")
            else:
                st.markdown(f"- {provider}: :red[Not set]")


def validate_api_keys(required_providers: list) -> tuple:
    """
    Check if required API keys are available.
    Returns: (is_valid, missing_providers)
    """
    missing = []
    for provider in required_providers:
        key = get_api_key(provider)
        if not key:
            missing.append(provider)
    return len(missing) == 0, missing
