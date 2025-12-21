"""
SimplerLLM Toolkit Playground
A beginner-friendly UI for experimenting with LLM tools
"""

import streamlit as st
from components.api_key_manager import render_api_key_manager
from tools.brainstorm import render_brainstorm
from tools.feedback_loop import render_feedback_loop
from tools.semantic_routing import render_semantic_routing
from tools.web_search_json import render_web_search_json
from tools.validator import render_validator
from tools.judge import render_judge


# Page config
st.set_page_config(
    page_title="SimplerLLM Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .tool-header {
        padding: 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .example-box {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 SimplerLLM Toolkit")
st.sidebar.markdown("---")

# API Key Manager
render_api_key_manager()

st.sidebar.markdown("---")

# Tool Navigation
TOOLS = {
    "🧠 Brainstorm": {
        "key": "brainstorm",
        "description": "Recursive idea generation with quality scoring",
        "render": render_brainstorm,
    },
    "🔄 Feedback Loop": {
        "key": "feedback_loop",
        "description": "Iterative content improvement through AI self-critique",
        "render": render_feedback_loop,
    },
    "🔀 Semantic Routing": {
        "key": "semantic_routing",
        "description": "Query classification and intelligent routing",
        "render": render_semantic_routing,
    },
    "🔍 Web Search + JSON": {
        "key": "web_search_json",
        "description": "Real-time web search with structured output",
        "render": render_web_search_json,
    },
    "✅ Validator": {
        "key": "validator",
        "description": "Multi-model consensus validation",
        "render": render_validator,
    },
    "⚖️ Judge": {
        "key": "judge",
        "description": "Multi-model competition with synthesis",
        "render": render_judge,
    },
}

selected_tool = st.sidebar.radio(
    "Select Tool",
    options=list(TOOLS.keys()),
    format_func=lambda x: x,
    key="tool_selector"
)

# Footer in sidebar
st.sidebar.markdown("---")
st.sidebar.caption("Prompting 2.0 Course")
st.sidebar.caption("Powered by SimplerLLM")

# Main content area
tool_config = TOOLS[selected_tool]

# Tool header
st.title(selected_tool)
st.markdown(f"*{tool_config['description']}*")
st.markdown("---")

# Render selected tool
tool_config["render"]()
