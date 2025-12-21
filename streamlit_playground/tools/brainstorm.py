"""
Brainstorm Tool
Recursive idea generation with quality scoring
"""

import streamlit as st
import pandas as pd
from components.api_key_manager import get_api_key, validate_api_keys
from components.llm_selector import render_llm_selector, get_provider_enum
from components.output_display import display_error, display_metrics, display_dataframe_with_export


def render_brainstorm():
    """Render the Brainstorm tool interface."""

    # Description
    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Brainstorm** generates ideas recursively with quality scoring.

        **Process:**
        1. Generate initial ideas on your topic
        2. Evaluate each idea with a quality score (1-10)
        3. Recursively expand the best ideas into sub-ideas
        4. Build a complete idea tree with multiple depth levels

        **Modes:**
        - **Tree**: Expand ALL ideas (comprehensive exploration)
        - **Linear**: Only expand the BEST idea at each level (focused depth)
        - **Hybrid**: Expand top-N ideas at each level (balanced)

        **Use Cases:**
        - Content ideation (YouTube videos, blog posts)
        - Marketing strategies
        - Product feature brainstorming
        - Problem-solving
        """)

    # Input Section
    st.subheader("Input")

    # Main prompt
    prompt = st.text_area(
        "Brainstorming Topic",
        placeholder="e.g., YouTube video ideas for a tech channel focused on AI tools",
        height=100,
        key="brainstorm_prompt"
    )

    # Optional context
    context = st.text_area(
        "Additional Context (Optional)",
        placeholder="e.g., Target audience: developers and tech enthusiasts",
        height=60,
        key="brainstorm_context"
    )

    # Parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        max_depth = st.slider(
            "Max Depth",
            min_value=1,
            max_value=3,
            value=2,
            key="brainstorm_depth",
            help="How many levels deep to expand ideas"
        )

    with col2:
        ideas_per_level = st.slider(
            "Ideas Per Level",
            min_value=2,
            max_value=5,
            value=3,
            key="brainstorm_ideas",
            help="Number of ideas to generate at each level"
        )

    with col3:
        mode = st.selectbox(
            "Expansion Mode",
            options=["tree", "linear", "hybrid"],
            index=0,
            key="brainstorm_mode",
            help="How to expand ideas at each level"
        )

    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        min_quality = st.slider(
            "Minimum Quality Threshold",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key="brainstorm_quality",
            help="Only expand ideas scoring above this threshold"
        )

    # LLM Provider
    st.markdown("**LLM Settings**")
    provider, model = render_llm_selector(
        key_prefix="brainstorm",
        label="LLM Provider"
    )

    # Example prompts
    with st.expander("Example Topics", expanded=False):
        examples = [
            "YouTube video ideas for a tech channel focused on AI tools",
            "Marketing strategies for a new SaaS product",
            "Blog post topics about machine learning for beginners",
            "Mobile app feature ideas for a productivity tool",
        ]

        for example in examples:
            if st.button(example[:50] + "...", key=f"brainstorm_ex_{hash(example)}"):
                st.session_state.brainstorm_prompt = example
                st.rerun()

    st.markdown("---")

    # Run button
    if st.button("Generate Ideas", type="primary", key="brainstorm_run"):
        if not prompt.strip():
            st.warning("Please enter a brainstorming topic")
            return

        # Validate API keys
        is_valid, missing = validate_api_keys([provider.lower()])
        if not is_valid:
            st.error(f"Missing API key for: {', '.join(missing)}")
            st.info("Please configure your API keys in the sidebar.")
            return

        # Run brainstorm
        with st.spinner("Generating ideas... This may take a moment."):
            try:
                from SimplerLLM.language import LLM, LLMProvider
                from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm

                # Create LLM
                provider_enum = get_provider_enum(provider)
                api_key = get_api_key(provider)
                llm = LLM.create(provider_enum, model_name=model, api_key=api_key)

                # Create brainstormer
                brainstorm = RecursiveBrainstorm(
                    llm=llm,
                    max_depth=max_depth,
                    ideas_per_level=ideas_per_level,
                    mode=mode,
                    min_quality_threshold=min_quality if 'min_quality' in dir() else 5.0,
                    verbose=False,
                )

                # Run brainstorm
                result = brainstorm.brainstorm(
                    prompt=prompt,
                    context=context if context.strip() else None
                )

                # Display results
                st.markdown("---")
                st.subheader("Results")

                # Metrics
                display_metrics({
                    "Total Ideas": result.total_ideas,
                    "Execution Time": f"{result.execution_time:.1f}s",
                    "Max Depth Reached": result.max_depth_reached,
                })

                # Best idea highlight
                st.subheader("Best Idea")
                best = result.overall_best_idea
                st.success(f"**{best.text}**")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Quality Score", f"{best.quality_score:.1f}/10")
                with col2:
                    st.metric("Depth Level", best.depth)

                if best.reasoning:
                    st.caption(f"*{best.reasoning}*")

                # Idea tree by level
                st.subheader("All Ideas by Level")

                for level in result.levels:
                    with st.expander(f"Level {level.depth} ({level.total_ideas} ideas)", expanded=(level.depth == 0)):
                        # Sort by quality score
                        sorted_ideas = sorted(level.ideas, key=lambda x: x.quality_score, reverse=True)

                        for idea in sorted_ideas:
                            score_color = "green" if idea.quality_score >= 7 else "orange" if idea.quality_score >= 5 else "red"

                            st.markdown(f":{score_color}[**{idea.quality_score:.1f}**] {idea.text}")

                            if idea.parent_id:
                                st.caption(f"↳ Expanded from parent idea")

                # Export as DataFrame
                st.subheader("Export")

                # Create DataFrame
                ideas_data = []
                for idea in result.all_ideas:
                    ideas_data.append({
                        "Idea": idea.text,
                        "Score": idea.quality_score,
                        "Depth": idea.depth,
                        "Reasoning": idea.reasoning or "",
                    })

                df = pd.DataFrame(ideas_data)
                df = df.sort_values("Score", ascending=False)

                display_dataframe_with_export(df, "Ideas Table")

            except Exception as e:
                display_error(e, "Brainstorm")
