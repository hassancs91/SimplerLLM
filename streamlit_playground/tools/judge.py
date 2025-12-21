"""
Judge Tool
Multi-model competition with synthesis
"""

import streamlit as st
from components.api_key_manager import get_api_key, validate_api_keys
from components.llm_selector import render_llm_selector, get_provider_enum, PROVIDER_MODELS
from components.output_display import display_error, display_score_badge


def render_judge():
    """Render the Judge tool interface."""

    # Description
    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Judge** runs multiple LLMs in competition and has a judge pick or synthesize the best answer.

        **Modes:**
        - **Select Best**: Judge picks the single best answer
        - **Synthesize**: Combine the best parts of all answers
        - **Compare**: Detailed comparison with scores

        **Process:**
        1. Send your prompt to multiple providers in parallel
        2. A judge LLM evaluates all responses
        3. Return the best answer or a synthesis

        **Use Cases:**
        - Getting the best possible answer to a question
        - Comparing model capabilities
        - Combining strengths of different models
        """)

    # Input Section
    st.subheader("Input")

    # Prompt
    prompt = st.text_area(
        "Prompt",
        placeholder="e.g., Explain what an API is using a real-world analogy that anyone can understand",
        height=100,
        key="judge_prompt"
    )

    # Mode selection
    mode = st.selectbox(
        "Mode",
        options=["select_best", "synthesize", "compare"],
        index=1,
        format_func=lambda x: {
            "select_best": "Select Best - Pick the single best answer",
            "synthesize": "Synthesize - Combine best parts of all answers",
            "compare": "Compare - Detailed comparison with scores",
        }[x],
        key="judge_mode"
    )

    # Competing providers
    st.markdown("**Competing Providers**")
    available_providers = list(PROVIDER_MODELS.keys())

    competing_providers = st.multiselect(
        "Select providers to compete",
        options=available_providers,
        default=["OpenAI", "Anthropic"],
        key="judge_providers",
        help="These models will compete to answer your prompt"
    )

    if len(competing_providers) < 2:
        st.warning("Select at least 2 providers to compete")

    # Judge provider
    st.markdown("**Judge**")
    judge_provider, judge_model = render_llm_selector(
        key_prefix="judge_judge",
        label="Judge Provider",
        default_provider="OpenAI"
    )

    # Evaluation criteria
    with st.expander("Evaluation Criteria", expanded=False):
        criteria_options = [
            "clarity",
            "accuracy",
            "completeness",
            "beginner-friendly",
            "creativity",
            "conciseness",
        ]

        selected_criteria = st.multiselect(
            "Select criteria for evaluation",
            options=criteria_options,
            default=["clarity", "accuracy", "beginner-friendly"],
            key="judge_criteria"
        )

    # Example prompts
    with st.expander("Example Prompts", expanded=False):
        examples = [
            "Explain what an API is using a real-world analogy that anyone can understand",
            "Explain recursion in programming with a simple example",
            "What are the key differences between SQL and NoSQL databases?",
            "Explain how machine learning works to a 10-year-old",
        ]

        for example in examples:
            if st.button(example[:50] + "...", key=f"judge_ex_{hash(example)}"):
                st.session_state.judge_prompt = example
                st.rerun()

    st.markdown("---")

    # Run button
    if st.button("Run Competition", type="primary", key="judge_run"):
        if not prompt.strip():
            st.warning("Please enter a prompt")
            return

        if len(competing_providers) < 2:
            st.warning("Please select at least 2 providers to compete")
            return

        # Validate API keys
        all_providers = competing_providers + [judge_provider]
        providers_lower = list(set([p.lower() for p in all_providers]))
        is_valid, missing = validate_api_keys(providers_lower)
        if not is_valid:
            st.error(f"Missing API keys for: {', '.join(missing)}")
            st.info("Please configure your API keys in the sidebar.")
            return

        # Run competition
        with st.spinner("Running competition... Getting answers from all providers..."):
            try:
                from SimplerLLM.language import LLM, LLMProvider, LLMJudge

                # Create competing LLMs
                providers = []
                for provider_name in competing_providers:
                    provider_enum = get_provider_enum(provider_name)
                    api_key = get_api_key(provider_name)
                    model = PROVIDER_MODELS[provider_name][0]
                    llm = LLM.create(provider_enum, model_name=model, api_key=api_key)
                    providers.append(llm)

                # Create judge LLM
                judge_enum = get_provider_enum(judge_provider)
                judge_api_key = get_api_key(judge_provider)
                judge_llm = LLM.create(judge_enum, model_name=judge_model, api_key=judge_api_key)

                # Create LLMJudge
                judge = LLMJudge(
                    providers=providers,
                    judge_llm=judge_llm,
                    parallel=True,
                    default_criteria=selected_criteria if selected_criteria else ["clarity", "accuracy"],
                    verbose=False,
                )

                # Run competition
                result = judge.generate(prompt=prompt, mode=mode)

                # Display results
                st.markdown("---")
                st.subheader("Results")

                # Final answer
                st.markdown("### Final Answer")
                st.markdown(result.final_answer)

                # Mode-specific display
                if mode == "select_best":
                    display_select_best_results(result, competing_providers)
                elif mode == "synthesize":
                    display_synthesize_results(result, competing_providers)
                elif mode == "compare":
                    display_compare_results(result, competing_providers)

                # Execution time
                st.caption(f"Total execution time: {result.total_execution_time:.2f}s")

            except Exception as e:
                display_error(e, "Judge")


def display_select_best_results(result, provider_names):
    """Display results for select_best mode."""
    st.markdown("### Provider Scores")

    # Find winner
    winner = None
    if result.evaluations:
        for evaluation in result.evaluations:
            if evaluation.rank == 1:
                winner = evaluation.provider_name
                break

    if winner:
        st.success(f"**Winner: {winner}**")

    # Individual evaluations
    for i, evaluation in enumerate(result.evaluations):
        provider_name = evaluation.provider_name or (provider_names[i] if i < len(provider_names) else f"Provider {i+1}")

        is_winner = evaluation.rank == 1
        expander_title = f"{'🏆 ' if is_winner else ''}{provider_name}: {evaluation.overall_score:.1f}/10 (Rank #{evaluation.rank})"

        with st.expander(expander_title, expanded=is_winner):
            display_score_badge(evaluation.overall_score, 10, "Score")

            if evaluation.strengths:
                st.markdown("**Strengths:**")
                for s in evaluation.strengths[:3]:
                    st.markdown(f"✅ {s}")

            if evaluation.weaknesses:
                st.markdown("**Weaknesses:**")
                for w in evaluation.weaknesses[:3]:
                    st.markdown(f"⚠️ {w}")


def display_synthesize_results(result, provider_names):
    """Display results for synthesize mode."""
    st.markdown("### Provider Contributions")

    st.info("The final answer above combines the best elements from all providers.")

    # Show individual evaluations
    for i, evaluation in enumerate(result.evaluations):
        provider_name = evaluation.provider_name or (provider_names[i] if i < len(provider_names) else f"Provider {i+1}")

        with st.expander(f"{provider_name}: {evaluation.overall_score:.1f}/10", expanded=False):
            display_score_badge(evaluation.overall_score, 10, "Score")

            if evaluation.strengths:
                st.markdown("**Contributions to synthesis:**")
                for s in evaluation.strengths[:3]:
                    st.markdown(f"✅ {s}")


def display_compare_results(result, provider_names):
    """Display results for compare mode."""
    st.markdown("### Detailed Comparison")

    # Create comparison table
    cols = st.columns(len(result.evaluations))

    for i, evaluation in enumerate(result.evaluations):
        provider_name = evaluation.provider_name or (provider_names[i] if i < len(provider_names) else f"Provider {i+1}")

        with cols[i]:
            st.markdown(f"### {provider_name}")
            st.metric("Score", f"{evaluation.overall_score:.1f}/10")
            st.metric("Rank", f"#{evaluation.rank}")

            if evaluation.strengths:
                st.markdown("**Strengths:**")
                for s in evaluation.strengths[:2]:
                    st.markdown(f"✅ {s}")

            if evaluation.weaknesses:
                st.markdown("**Weaknesses:**")
                for w in evaluation.weaknesses[:2]:
                    st.markdown(f"⚠️ {w}")
