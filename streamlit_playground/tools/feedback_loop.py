"""
Feedback Loop Tool
Iterative content improvement through AI self-critique
"""

import streamlit as st
import plotly.graph_objects as go
from components.api_key_manager import get_api_key, validate_api_keys
from components.llm_selector import render_llm_selector, get_provider_enum
from components.output_display import display_error, display_score_badge


def render_feedback_loop():
    """Render the Feedback Loop tool interface."""

    # Description
    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Feedback Loop** iteratively improves content through AI self-critique.

        **Process:**
        1. Generate an initial answer to your prompt
        2. LLM critiques its own answer (identifies strengths, weaknesses, suggestions)
        3. LLM applies feedback to improve the answer
        4. Repeat until quality threshold is reached or max iterations

        **Use Cases:**
        - Improving explanations for clarity
        - Refining product descriptions
        - Polishing creative writing
        - Making technical content more accessible
        """)

    # Input Section
    st.subheader("Input")

    # Main prompt
    prompt = st.text_area(
        "Prompt",
        placeholder="e.g., Explain what an API is to a beginner with no programming experience",
        height=100,
        key="feedback_prompt"
    )

    # Optional initial answer
    with st.expander("Initial Answer (Optional)", expanded=False):
        st.caption("Provide an existing answer to improve, or leave empty to generate one")
        initial_answer = st.text_area(
            "Starting Answer",
            placeholder="Leave empty to let the AI generate the initial answer",
            height=100,
            key="feedback_initial"
        )

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=5,
            value=3,
            key="feedback_max_iter",
            help="Maximum number of improvement cycles"
        )

    with col2:
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=1.0,
            max_value=10.0,
            value=8.0,
            step=0.5,
            key="feedback_threshold",
            help="Stop early if this score is reached"
        )

    # LLM Provider
    st.markdown("**LLM Settings**")
    provider, model = render_llm_selector(
        key_prefix="feedback",
        label="LLM Provider"
    )

    # Example prompts
    with st.expander("Example Prompts", expanded=False):
        examples = [
            "Explain what an API is to a beginner with no programming experience",
            "Write a product description for an AI writing assistant",
            "Explain recursion in programming with a simple example",
            "Describe the benefits of cloud computing for small businesses",
        ]

        for example in examples:
            if st.button(example[:50] + "...", key=f"feedback_ex_{hash(example)}"):
                st.session_state.feedback_prompt = example
                st.rerun()

    st.markdown("---")

    # Run button
    if st.button("Start Improvement Loop", type="primary", key="feedback_run"):
        if not prompt.strip():
            st.warning("Please enter a prompt")
            return

        # Validate API keys
        is_valid, missing = validate_api_keys([provider.lower()])
        if not is_valid:
            st.error(f"Missing API key for: {', '.join(missing)}")
            st.info("Please configure your API keys in the sidebar.")
            return

        # Run feedback loop
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            from SimplerLLM.language import LLM, LLMProvider, LLMFeedbackLoop

            # Create LLM
            provider_enum = get_provider_enum(provider)
            api_key = get_api_key(provider)
            llm = LLM.create(provider_enum, model_name=model, api_key=api_key)

            # Create feedback loop
            feedback_loop = LLMFeedbackLoop(
                llm=llm,
                max_iterations=max_iterations,
                quality_threshold=quality_threshold,
                verbose=False,
            )

            # Run the loop
            status_text.text("Starting improvement loop...")

            if initial_answer.strip():
                result = feedback_loop.improve(
                    prompt=prompt,
                    initial_answer=initial_answer.strip()
                )
            else:
                result = feedback_loop.generate_and_improve(prompt=prompt)

            progress_bar.progress(100)
            status_text.empty()

            # Display results
            st.markdown("---")
            st.subheader("Results")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Iterations", result.iterations_completed)
            with col2:
                st.metric("Final Score", f"{result.final_score:.1f}/10")
            with col3:
                st.metric("Execution Time", f"{result.execution_time:.1f}s")

            # Stop reason
            st.info(f"**Stop Reason:** {result.stop_reason}")

            # Score trajectory chart
            if result.trajectory and len(result.trajectory) > 1:
                st.subheader("Improvement Trajectory")

                scores = [t.score for t in result.trajectory]
                iterations = list(range(1, len(scores) + 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=iterations,
                    y=scores,
                    mode='lines+markers',
                    name='Score',
                    line=dict(color='#4F46E5', width=3),
                    marker=dict(size=10)
                ))
                fig.add_hline(
                    y=quality_threshold,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Quality Threshold"
                )
                fig.update_layout(
                    xaxis_title="Iteration",
                    yaxis_title="Score",
                    yaxis_range=[0, 10.5],
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Final answer
            st.subheader("Final Answer")
            st.markdown(result.final_answer)

            # Side-by-side comparison
            if result.trajectory and len(result.trajectory) > 0:
                with st.expander("Compare Initial vs Final", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Initial Answer**")
                        initial = result.trajectory[0].answer if result.trajectory else "N/A"
                        st.markdown(initial[:500] + "..." if len(initial) > 500 else initial)

                    with col2:
                        st.markdown("**Final Answer**")
                        st.markdown(result.final_answer[:500] + "..." if len(result.final_answer) > 500 else result.final_answer)

            # Iteration details
            if result.trajectory:
                with st.expander("Iteration Details", expanded=False):
                    for i, iteration in enumerate(result.trajectory, 1):
                        st.markdown(f"### Iteration {i}")

                        display_score_badge(iteration.score, 10, "Score")

                        if hasattr(iteration, 'critique') and iteration.critique:
                            critique = iteration.critique

                            if hasattr(critique, 'strengths') and critique.strengths:
                                st.markdown("**Strengths:**")
                                for s in critique.strengths[:3]:
                                    st.markdown(f"- {s}")

                            if hasattr(critique, 'weaknesses') and critique.weaknesses:
                                st.markdown("**Weaknesses:**")
                                for w in critique.weaknesses[:3]:
                                    st.markdown(f"- {w}")

                            if hasattr(critique, 'suggestions') and critique.suggestions:
                                st.markdown("**Suggestions:**")
                                for s in critique.suggestions[:3]:
                                    st.markdown(f"- {s}")

                        st.markdown("---")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            display_error(e, "Feedback Loop")
