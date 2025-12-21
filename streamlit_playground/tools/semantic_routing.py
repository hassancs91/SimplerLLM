"""
Semantic Routing Tool
Query classification and intelligent routing
"""

import json
import streamlit as st
from components.api_key_manager import get_api_key, validate_api_keys
from components.llm_selector import render_llm_selector, get_provider_enum
from components.output_display import display_error, display_confidence_bar


def render_semantic_routing():
    """Render the Semantic Routing tool interface."""

    # Description
    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Semantic Routing** classifies incoming queries into predefined categories
        and routes them to appropriate handlers.

        **Methods:**
        - **Pattern-only**: Fast regex matching (no LLM calls)
        - **LLM-only**: Uses AI for complex classification
        - **Hybrid**: Tries patterns first, falls back to LLM for complex cases

        **Use Cases:**
        - Customer support ticket routing
        - Intent detection in chatbots
        - Content moderation
        """)

    # Input Section
    st.subheader("Input")

    # Query input
    query = st.text_input(
        "Query to classify",
        placeholder="e.g., How much does the enterprise plan cost?",
        key="routing_query"
    )

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Classification Method",
            options=["hybrid", "pattern", "llm"],
            index=0,
            key="routing_method",
            help="Hybrid: tries patterns first, then LLM for complex cases"
        )

    with col2:
        enable_cache = st.checkbox(
            "Enable Cache",
            value=True,
            key="routing_cache",
            help="Cache results to avoid re-classifying identical queries"
        )

    # LLM Provider (only if method uses LLM)
    if method in ["hybrid", "llm"]:
        st.markdown("**LLM Settings**")
        provider, model = render_llm_selector(
            key_prefix="routing",
            label="LLM Provider"
        )
    else:
        provider, model = "OpenAI", "gpt-4o"

    # Custom Patterns
    with st.expander("Custom Patterns (Optional)", expanded=False):
        st.caption("Define regex patterns for each category (JSON format)")

        default_patterns = {
            "sales": ["price", "cost", "buy", "upgrade", "plans?", "demo", "subscribe"],
            "technical": ["error", "bug", "broken", "crash", "not working", "how to", "help"],
            "billing": ["invoice", "refund", "payment", "charge", "receipt", "cancel"],
        }

        patterns_json = st.text_area(
            "Patterns (JSON)",
            value=json.dumps(default_patterns, indent=2),
            height=150,
            key="routing_patterns"
        )

    # Example prompts
    with st.expander("Example Queries", expanded=False):
        examples = [
            "How much does the enterprise plan cost?",
            "My app keeps crashing when I click save",
            "I need a copy of my invoice from last month",
            "I was charged twice but the feature still doesn't work",
            "Can you help me understand your pricing before I decide?",
        ]

        for example in examples:
            if st.button(example, key=f"routing_ex_{hash(example)}"):
                st.session_state.routing_query = example
                st.rerun()

    st.markdown("---")

    # Run button
    if st.button("Classify Query", type="primary", key="routing_run"):
        if not query.strip():
            st.warning("Please enter a query to classify")
            return

        # Validate API keys if using LLM
        if method in ["hybrid", "llm"]:
            is_valid, missing = validate_api_keys([provider.lower()])
            if not is_valid:
                st.error(f"Missing API key for: {', '.join(missing)}")
                st.info("Please configure your API keys in the sidebar.")
                return

        # Parse patterns
        try:
            custom_patterns = json.loads(patterns_json)
        except json.JSONDecodeError:
            st.error("Invalid JSON in custom patterns")
            return

        # Run classification
        with st.spinner("Classifying query..."):
            try:
                from SimplerLLM.language import LLM, LLMProvider
                from SimplerLLM.language.llm_provider_router import QueryClassifier

                # Create LLM if needed
                llm = None
                if method in ["hybrid", "llm"]:
                    provider_enum = get_provider_enum(provider)
                    api_key = get_api_key(provider)
                    llm = LLM.create(provider_enum, model_name=model, api_key=api_key)

                # Create classifier
                classifier = QueryClassifier(
                    classifier_llm=llm,
                    method=method,
                    enable_cache=enable_cache,
                    custom_patterns=custom_patterns,
                    verbose=False,
                )

                # Classify
                result = classifier.classify(query)

                # Display results
                st.markdown("---")
                st.subheader("Classification Result")

                # Route destination
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"### Route to: :blue[{result.query_type.upper()}]")

                with col2:
                    st.metric("Method Used", result.matched_by.capitalize())

                # Confidence
                display_confidence_bar(result.confidence, "Confidence")

                # Reasoning
                if result.reasoning:
                    with st.expander("Reasoning", expanded=True):
                        st.write(result.reasoning)

                # Cache stats
                if enable_cache:
                    stats = classifier.get_cache_stats()
                    st.caption(f"Cache: {stats['total_entries']} entries, {stats['total_hits']} hits")

            except Exception as e:
                display_error(e, "Semantic Routing")

    # Batch mode
    st.markdown("---")
    with st.expander("Batch Classification", expanded=False):
        st.caption("Classify multiple queries at once (one per line)")

        batch_queries = st.text_area(
            "Queries (one per line)",
            placeholder="How much does the plan cost?\nMy app is crashing\nI need an invoice",
            height=100,
            key="routing_batch"
        )

        if st.button("Classify All", key="routing_batch_run"):
            queries = [q.strip() for q in batch_queries.strip().split("\n") if q.strip()]

            if not queries:
                st.warning("Please enter at least one query")
                return

            if method in ["hybrid", "llm"]:
                is_valid, missing = validate_api_keys([provider.lower()])
                if not is_valid:
                    st.error(f"Missing API key for: {', '.join(missing)}")
                    return

            try:
                custom_patterns = json.loads(patterns_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON in custom patterns")
                return

            with st.spinner(f"Classifying {len(queries)} queries..."):
                try:
                    from SimplerLLM.language import LLM, LLMProvider
                    from SimplerLLM.language.llm_provider_router import QueryClassifier

                    llm = None
                    if method in ["hybrid", "llm"]:
                        provider_enum = get_provider_enum(provider)
                        api_key = get_api_key(provider)
                        llm = LLM.create(provider_enum, model_name=model, api_key=api_key)

                    classifier = QueryClassifier(
                        classifier_llm=llm,
                        method=method,
                        enable_cache=enable_cache,
                        custom_patterns=custom_patterns,
                        verbose=False,
                    )

                    results = []
                    for q in queries:
                        result = classifier.classify(q)
                        results.append({
                            "Query": q[:50] + "..." if len(q) > 50 else q,
                            "Route": result.query_type.upper(),
                            "Confidence": f"{result.confidence:.0%}",
                            "Method": result.matched_by,
                        })

                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                except Exception as e:
                    display_error(e, "Batch Classification")
