# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
pip install -e .
```

### Running Tests
```bash
# Run specific test files
python tests/test_agent.py
python tests/test_router_youtube_titles.py

# Run vector database examples
python tests/example_vector_db_usage.py
python tests/example_unified_vector_db.py
```

### Package Building
```bash
# Build package for distribution
python setup.py sdist bdist_wheel
```

## Architecture Overview

SimplerLLM is a Python library designed to simplify interactions with Large Language Models. The codebase follows a modular architecture:

### Core Components

**LLM Layer (`SimplerLLM/language/`)**
- `llm/base.py`: Core LLM abstraction with factory pattern using LLMProvider enum
- `llm/wrappers/`: Provider-specific implementations (OpenAI, Anthropic, Gemini, Ollama, DeepSeek, OpenRouter)
- `llm_providers/`: Direct provider implementations
- `llm_router/`: Model routing and selection logic
- `embeddings.py`: Embedding model abstractions (OpenAI, Voyage AI)

**Vector Storage (`SimplerLLM/vectors/`)**
- `vector_db.py`: Unified vector database interface using VectorProvider enum
- `local_vector_db.py`: Local vector storage implementation
- `qdrant_vector_db.py`: Qdrant vector database integration
- `simpler_vector.py`: Simplified vector operations

**Tools (`SimplerLLM/tools/`)**
- `generic_loader.py`: Universal content loader (PDF, DOCX, web pages)
- `text_chunker.py`: Text chunking strategies (semantic, sentence, paragraph)
- `serp.py`: Search engine integration (Serper, Value Serp)
- `rapid_api.py`: RapidAPI client wrapper
- `youtube.py`: YouTube data extraction
- `json_helpers.py`: JSON parsing and Pydantic model utilities

**Prompts (`SimplerLLM/prompts/`)**
- `prompt_builder.py`: Template-based prompt construction
- `messages_template.py`: Message formatting utilities
- `hub/`: Prompt management and agentic prompts

### Key Design Patterns

1. **Factory Pattern**: LLM and VectorDB classes use static `create()` methods with provider enums
2. **Wrapper Pattern**: Each LLM provider has a wrapper that standardizes the interface
3. **Unified Interface**: All providers implement consistent `generate_response()` methods
4. **Pydantic Integration**: JSON responses are validated using Pydantic models

### Environment Variables Required

```bash
OPENAI_API_KEY          # OpenAI API access
GEMINI_API_KEY          # Google Gemini API access
ANTHROPIC_API_KEY       # Anthropic Claude API access
OPENROUTER_API_KEY      # OpenRouter API access
OPENROUTER_SITE_URL     # Optional: Your site URL for OpenRouter tracking
OPENROUTER_SITE_NAME    # Optional: Your site name for OpenRouter tracking
COHERE_API_KEY          # Cohere API access
VOYAGE_API_KEY          # Voyage AI embeddings API access
RAPIDAPI_API_KEY        # RapidAPI services
VALUE_SERP_API_KEY      # Value Serp search API
SERPER_API_KEY          # Serper search API
STABILITY_API_KEY       # Stability AI for image generation
```

### Common Usage Patterns

- LLM instances are created using `LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")`
- OpenRouter provides unified access to 100+ models: `LLM.create(provider=LLMProvider.OPENROUTER, model_name="openai/gpt-4o")`
- Embedding models support multiple providers: `EmbeddingsLLM.create(provider=EmbeddingsProvider.VOYAGE, model_name="voyage-3-large")`
- Cohere provides both chat and embeddings: `LLM.create(provider=LLMProvider.COHERE, model_name="command-r-plus")` and `EmbeddingsLLM.create(provider=EmbeddingsProvider.COHERE, model_name="embed-english-v3.0")`
- Vector databases are created using `VectorDB.create(provider=VectorProvider.LOCAL)`
- JSON responses are generated using `generate_pydantic_json_model()` for consistent structured output
- Content loading supports multiple formats through `load_content()` function

### Testing Structure

Tests are located in `/tests/` directory and include:
- Agent functionality testing
- Vector database usage examples
- YouTube title generation workflows
- Router testing for model selection