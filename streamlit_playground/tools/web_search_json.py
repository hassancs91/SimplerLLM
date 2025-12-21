"""
Web Search + JSON Tool
Real-time web search with structured Pydantic output
"""

import json
import streamlit as st
from pydantic import BaseModel, Field, create_model
from typing import List, Optional
from components.api_key_manager import get_api_key, validate_api_keys
from components.llm_selector import render_llm_selector, get_provider_enum
from components.output_display import display_error, display_json


# Pre-built schemas
class NewsItem(BaseModel):
    title: str = Field(description="The headline of the news article")
    source: str = Field(description="The publication or website name")
    date: str = Field(description="Publication date in YYYY-MM-DD format")
    summary: str = Field(description="A 1-2 sentence summary of the news")
    category: str = Field(description="Category: breakthrough, product, research, or business")


class AINewsResponse(BaseModel):
    news: List[NewsItem] = Field(description="List of AI news items")
    search_date: str = Field(description="Today's date when search was performed")


class ProductItem(BaseModel):
    name: str = Field(description="Product name")
    price: str = Field(description="Price in USD")
    rating: float = Field(description="Rating out of 5")
    pros: List[str] = Field(description="List of advantages")
    cons: List[str] = Field(description="List of disadvantages")


class ProductComparisonResponse(BaseModel):
    products: List[ProductItem] = Field(description="List of compared products")
    comparison_date: str = Field(description="Date of comparison")


class WeatherDay(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    high_temp: int = Field(description="High temperature in Fahrenheit")
    low_temp: int = Field(description="Low temperature in Fahrenheit")
    condition: str = Field(description="Weather condition (sunny, cloudy, rainy, etc.)")


class WeatherResponse(BaseModel):
    location: str = Field(description="City and country")
    forecast: List[WeatherDay] = Field(description="Weather forecast for upcoming days")


SCHEMA_OPTIONS = {
    "AI News": {
        "model": AINewsResponse,
        "example_prompt": "Search for the top 5 most important AI news stories from the past week. Include breakthroughs, products, research, and business news.",
    },
    "Product Comparison": {
        "model": ProductComparisonResponse,
        "example_prompt": "Compare the top 3 noise-canceling headphones under $300. Include price, rating, pros and cons.",
    },
    "Weather Forecast": {
        "model": WeatherResponse,
        "example_prompt": "Get the 5-day weather forecast for New York City.",
    },
}


def render_web_search_json():
    """Render the Web Search + JSON tool interface."""

    # Description
    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Web Search + JSON** combines real-time web search with structured output.

        **Process:**
        1. Search the web for current information
        2. Parse results into a defined Pydantic schema
        3. Return validated, structured JSON

        **Features:**
        - Real-time web search capability
        - Type-safe structured data via Pydantic models
        - Automatic JSON validation with retry logic

        **Use Cases:**
        - News aggregation
        - Product comparisons
        - Research data collection
        - Market analysis
        """)

    # Input Section
    st.subheader("Input")

    # Schema selection
    schema_choice = st.selectbox(
        "Select Output Schema",
        options=list(SCHEMA_OPTIONS.keys()),
        key="websearch_schema"
    )

    selected_schema = SCHEMA_OPTIONS[schema_choice]

    # Show schema preview
    with st.expander("Schema Preview", expanded=False):
        model_class = selected_schema["model"]
        schema_json = model_class.model_json_schema()
        st.json(schema_json)

    # Prompt
    prompt = st.text_area(
        "Search Prompt",
        value=selected_schema["example_prompt"],
        height=100,
        key="websearch_prompt",
        help="Describe what information you want to search for"
    )

    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        key="websearch_temp",
        help="Lower = more factual, Higher = more creative"
    )

    # LLM Provider
    st.markdown("**LLM Settings**")
    provider, model = render_llm_selector(
        key_prefix="websearch",
        label="LLM Provider"
    )

    st.markdown("---")

    # Run button
    if st.button("Search & Parse", type="primary", key="websearch_run"):
        if not prompt.strip():
            st.warning("Please enter a search prompt")
            return

        # Validate API keys
        is_valid, missing = validate_api_keys([provider.lower()])
        if not is_valid:
            st.error(f"Missing API key for: {', '.join(missing)}")
            st.info("Please configure your API keys in the sidebar.")
            return

        # Run web search
        with st.spinner("Searching the web and parsing results..."):
            try:
                from SimplerLLM.language import LLM, LLMProvider
                from SimplerLLM.language.llm_addons import generate_pydantic_json_model

                # Create LLM
                provider_enum = get_provider_enum(provider)
                api_key = get_api_key(provider)
                llm = LLM.create(provider_enum, model_name=model, api_key=api_key)

                # Get the schema model
                model_class = selected_schema["model"]

                # Run search and parse
                result = generate_pydantic_json_model(
                    model_class=model_class,
                    prompt=prompt,
                    llm_instance=llm,
                    web_search=True,
                    temperature=temperature,
                )

                # Check for errors
                if isinstance(result, str):
                    st.error(f"Error: {result}")
                    return

                # Display results
                st.markdown("---")
                st.subheader("Results")

                # Display based on schema type
                if schema_choice == "AI News":
                    display_news_results(result)
                elif schema_choice == "Product Comparison":
                    display_product_results(result)
                elif schema_choice == "Weather Forecast":
                    display_weather_results(result)
                else:
                    # Generic display
                    display_json(result, "Parsed Results")

                # Raw JSON
                display_json(result, "Raw JSON Output")

            except Exception as e:
                display_error(e, "Web Search + JSON")


def display_news_results(result):
    """Display AI News results in a nice format."""
    st.caption(f"Search Date: {result.search_date}")
    st.caption(f"Found {len(result.news)} stories")

    category_emojis = {
        "breakthrough": "🔬",
        "product": "📦",
        "research": "📚",
        "business": "💼",
    }

    for i, item in enumerate(result.news, 1):
        emoji = category_emojis.get(item.category.lower(), "📰")

        with st.container():
            st.markdown(f"### {emoji} {item.title}")
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.caption(f"📰 {item.source}")
            with col2:
                st.caption(f"📅 {item.date}")
            with col3:
                st.caption(f"🏷️ {item.category}")
            st.markdown(item.summary)
            st.markdown("---")


def display_product_results(result):
    """Display Product Comparison results."""
    st.caption(f"Comparison Date: {result.comparison_date}")

    cols = st.columns(len(result.products))

    for i, product in enumerate(result.products):
        with cols[i]:
            st.markdown(f"### {product.name}")
            st.metric("Price", product.price)
            st.metric("Rating", f"{product.rating}/5")

            st.markdown("**Pros:**")
            for pro in product.pros[:3]:
                st.markdown(f"✅ {pro}")

            st.markdown("**Cons:**")
            for con in product.cons[:3]:
                st.markdown(f"❌ {con}")


def display_weather_results(result):
    """Display Weather Forecast results."""
    st.markdown(f"### Weather for {result.location}")

    cols = st.columns(len(result.forecast))

    condition_emojis = {
        "sunny": "☀️",
        "cloudy": "☁️",
        "rainy": "🌧️",
        "partly cloudy": "⛅",
        "stormy": "⛈️",
        "snowy": "❄️",
    }

    for i, day in enumerate(result.forecast):
        with cols[i]:
            emoji = "🌤️"
            for condition, e in condition_emojis.items():
                if condition in day.condition.lower():
                    emoji = e
                    break

            st.markdown(f"**{day.date}**")
            st.markdown(f"# {emoji}")
            st.markdown(f"**{day.high_temp}°F** / {day.low_temp}°F")
            st.caption(day.condition)
