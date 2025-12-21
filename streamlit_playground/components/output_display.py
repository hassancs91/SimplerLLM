"""
Output Display Component
Reusable result visualization and error handling
"""

import json
import streamlit as st
import pandas as pd
from typing import Any


def display_error(error: Exception, context: str = ""):
    """Display user-friendly error message with troubleshooting tips."""
    error_type = type(error).__name__
    error_msg = str(error)

    # Map common errors to user-friendly messages
    if "API key" in error_msg or "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
        st.error("API Key Error")
        st.info("Please check your API key in the sidebar settings. Make sure it's valid and has the required permissions.")
    elif "rate limit" in error_msg.lower() or "rate_limit" in error_msg.lower():
        st.error("Rate Limit Reached")
        st.info("You've hit the API rate limit. Please wait a moment and try again.")
    elif "timeout" in error_msg.lower():
        st.error("Request Timeout")
        st.info("The request took too long. Try with a simpler prompt or fewer options.")
    elif "model" in error_msg.lower() and "not found" in error_msg.lower():
        st.error("Model Not Available")
        st.info("The selected model is not available. Please try a different model.")
    else:
        st.error(f"Error: {error_msg}")

    # Technical details in expander
    with st.expander("Technical Details", expanded=False):
        st.code(f"{error_type}: {error_msg}")
        if context:
            st.text(f"Context: {context}")


def display_success(message: str, execution_time: float = None):
    """Display success message with optional execution time."""
    if execution_time:
        st.success(f"{message} (Completed in {execution_time:.2f}s)")
    else:
        st.success(message)


def display_metrics(metrics: dict, columns: int = 3):
    """Display metrics in columns."""
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            st.metric(label=label, value=value)


def display_json(data: Any, title: str = "JSON Output"):
    """Display JSON data with copy and download options."""
    with st.expander(title, expanded=False):
        # Convert to JSON string
        if hasattr(data, "dict"):
            json_str = json.dumps(data.dict(), indent=2)
        elif hasattr(data, "model_dump"):
            json_str = json.dumps(data.model_dump(), indent=2)
        elif isinstance(data, dict):
            json_str = json.dumps(data, indent=2)
        else:
            json_str = json.dumps(data, indent=2, default=str)

        st.code(json_str, language="json")

        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="result.json",
            mime="application/json"
        )


def display_dataframe_with_export(df: pd.DataFrame, title: str = "Results"):
    """Display a dataframe with export options."""
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )
    with col2:
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="results.json",
            mime="application/json"
        )


def display_score_badge(score: float, max_score: float = 10, label: str = "Score"):
    """Display a score with color-coded badge."""
    percentage = score / max_score

    if percentage >= 0.8:
        color = "green"
    elif percentage >= 0.6:
        color = "orange"
    else:
        color = "red"

    st.markdown(f"**{label}:** :{color}[{score:.1f}/{max_score}]")


def display_pass_fail_badge(passed: bool, label: str = "Result"):
    """Display a PASS/FAIL badge."""
    if passed:
        st.markdown(f"### {label}: :green[PASS]")
    else:
        st.markdown(f"### {label}: :red[FAIL]")


def display_confidence_bar(confidence: float, label: str = "Confidence"):
    """Display a confidence score as a progress bar."""
    st.markdown(f"**{label}:** {confidence:.0%}")
    st.progress(confidence)
