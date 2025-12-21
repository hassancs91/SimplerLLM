"""
Validator Tool
Multi-model consensus validation
"""

import streamlit as st
from components.api_key_manager import get_api_key, validate_api_keys
from components.llm_selector import render_llm_selector, get_provider_enum, PROVIDER_MODELS
from components.output_display import display_error, display_pass_fail_badge, display_confidence_bar


def render_validator():
    """Render the Validator tool interface."""

    # Description
    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Validator** uses multiple LLM models to validate content accuracy through consensus.

        **Process:**
        1. Submit content to multiple validators in parallel
        2. Each validator scores the content (0-1)
        3. Aggregate scores using your chosen method
        4. Determine PASS/FAIL based on threshold

        **Aggregation Methods:**
        - **Average**: Simple mean of all scores
        - **Weighted**: Weight by model capability
        - **Median**: Middle value (resistant to outliers)
        - **Consensus**: All must agree within tolerance

        **Use Cases:**
        - Fact-checking AI-generated content
        - Validating technical accuracy
        - Quality assurance for content
        - Cross-checking information
        """)

    # Input Section
    st.subheader("Input")

    # Content to validate
    content = st.text_area(
        "Content to Validate",
        placeholder="e.g., Python is a programming language created by Guido van Rossum in 1991. The latest stable version is Python 4.0...",
        height=150,
        key="validator_content"
    )

    # Validation prompt
    validation_prompt = st.text_area(
        "Validation Instructions",
        value="Validate the factual accuracy of this content. Check for any incorrect statements, outdated information, or misleading claims.",
        height=80,
        key="validator_prompt"
    )

    # Validators selection
    st.markdown("**Select Validators**")

    available_providers = list(PROVIDER_MODELS.keys())
    selected_validators = st.multiselect(
        "Choose providers to validate with",
        options=available_providers,
        default=["OpenAI", "Anthropic"],
        key="validator_providers",
        help="Select 2+ providers for consensus validation"
    )

    if len(selected_validators) < 2:
        st.warning("Select at least 2 validators for consensus")

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        aggregation_method = st.selectbox(
            "Aggregation Method",
            options=["average", "weighted", "median", "consensus"],
            index=0,
            key="validator_aggregation"
        )

    with col2:
        threshold = st.slider(
            "Pass Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="validator_threshold",
            help="Score required to pass validation"
        )

    # Example content
    with st.expander("Example Content to Validate", expanded=False):
        examples = [
            {
                "title": "Python Facts (with error)",
                "content": "Python is a programming language created by Guido van Rossum in 1991. It's known for its simple syntax. The latest stable version is Python 4.0, released in 2024.",
            },
            {
                "title": "Historical Facts",
                "content": "The Eiffel Tower was completed in 1889 for the World's Fair in Paris. It stands at approximately 330 meters tall and was the world's tallest structure until 1930.",
            },
        ]

        for example in examples:
            if st.button(example["title"], key=f"validator_ex_{hash(example['title'])}"):
                st.session_state.validator_content = example["content"]
                st.rerun()

    st.markdown("---")

    # Run button
    if st.button("Validate Content", type="primary", key="validator_run"):
        if not content.strip():
            st.warning("Please enter content to validate")
            return

        if len(selected_validators) < 1:
            st.warning("Please select at least one validator")
            return

        # Validate API keys
        providers_lower = [p.lower() for p in selected_validators]
        is_valid, missing = validate_api_keys(providers_lower)
        if not is_valid:
            st.error(f"Missing API keys for: {', '.join(missing)}")
            st.info("Please configure your API keys in the sidebar.")
            return

        # Run validation
        with st.spinner("Validating with multiple models..."):
            try:
                from SimplerLLM.language import LLM, LLMProvider, LLMValidator, AggregationMethod

                # Map aggregation method
                agg_method_map = {
                    "average": AggregationMethod.AVERAGE,
                    "weighted": AggregationMethod.WEIGHTED,
                    "median": AggregationMethod.MEDIAN,
                    "consensus": AggregationMethod.CONSENSUS,
                }

                # Create validator LLMs
                validators = []
                for provider_name in selected_validators:
                    provider_enum = get_provider_enum(provider_name)
                    api_key = get_api_key(provider_name)
                    model = PROVIDER_MODELS[provider_name][0]  # Use default model
                    llm = LLM.create(provider_enum, model_name=model, api_key=api_key)
                    validators.append(llm)

                # Create LLMValidator
                validator = LLMValidator(
                    validators=validators,
                    aggregation_method=agg_method_map[aggregation_method],
                    threshold=threshold,
                    parallel=True,
                    verbose=False,
                )

                # Run validation
                result = validator.validate(
                    content=content,
                    validation_prompt=validation_prompt,
                )

                # Display results
                st.markdown("---")
                st.subheader("Validation Results")

                # Overall result
                display_pass_fail_badge(result.passed, "Overall Result")

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{result.overall_score:.2f}")
                with col2:
                    st.metric("Threshold", f"{threshold:.2f}")
                with col3:
                    consensus_text = "Yes" if result.consensus else "No"
                    st.metric("Consensus", consensus_text)

                # Progress bar for score
                display_confidence_bar(result.overall_score, "Score")

                # Individual validator results
                st.subheader("Individual Validators")

                for i, validator_score in enumerate(result.validator_scores):
                    provider_name = selected_validators[i] if i < len(selected_validators) else f"Validator {i+1}"

                    with st.expander(f"{provider_name}: {validator_score.score:.2f}", expanded=True):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            display_confidence_bar(validator_score.score, "Score")

                        with col2:
                            if validator_score.execution_time:
                                st.caption(f"⏱️ {validator_score.execution_time:.2f}s")

                        if validator_score.explanation:
                            st.markdown("**Explanation:**")
                            st.markdown(validator_score.explanation)

            except Exception as e:
                display_error(e, "Validator")
