# ⚪ SimplerLLM (Beta)

⚡ Your Easy Pass to Advanced AI ⚡


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Join the Discord chat!](https://img.shields.io/badge/Join-Discord-7289DA.svg)](https://discord.gg/HUrtZXyp3j)




## 🤔 What is SimplerLLM?

SimplerLLM is an open-source Python library designed to simplify interactions with Large Language Models (LLMs) for researchers and beginners. It offers a unified interface for different LLM providers and a suite of tools to enhance language model capabilities and make it Super easy for anyone to develop AI-powered tools and apps.

## Easy Installation

With pip:

```bash
pip install simplerllm
```

## Features

- **Unified LLM Interface**: Define an LLM instance in one line for providers like OpenAI and Google Gemini. Future versions will support more APIs and LLM providers.
- **Generic Text Loader**: Load text from various sources like DOCX, PDF, TXT files, YouTube scripts, or blog posts.
- **RapidAPI Connector**: Connect with AI services on RapidAPI.
- **SERP Integration**: Perform searches using DuckDuckGo, with more search engines coming soon.
- **Prompt Template Builder**: Easily create and manage prompt templates.
  And Much More Coming Soon!

### Setting Up Environment Variables

To use this library, you need to set several API keys in your environment. Start by creating a .env file in the root directory of your project and adding your API keys there.

🔴 This file should be kept private and not committed to version control to protect your keys.

Here is an example of what your .env file should look like:

```
OPENAI_API_KEY="your_openai_api_key_here"
GEMENI_API_KEY="your_gemeni_api_key_here"
CLAUDE_API_KEY="your_claude_api_key_here"
RAPIDAPI_API_KEY="your_rapidapi_key_here" # for accessing APIs on RapidAPI
VALUE_SERP_API_KEY="your_value_serp_api_key_here" #for Google search
SERPER_API_KEY="your_serper_api_key_here" #for Google search
STABILITY_API_KEY="your_stability_api_key_here" #for image generation

```

### Creating an LLM Instance

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# For OpenAI
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")

# For Google Gemini
#llm_instance = LLM.create(provider=LLMProvider.GEMINI,model_name="gemini-pro")

# For Anthropic Claude 
#llm_instance = LLM.create(LLMProvider.ANTHROPIC, model_name="claude-3-opus-20240229")

response = llm_instance.generate_response(prompt="generate a 5 words sentence")

```

### Using Tools

#### SERP

```python
from SimplerLLM.tools.serp import search_with_serper_api

search_results = search_with_serper_api("your search query", num_results=3)

# use the search results the way you want!

```

#### Generic Text Loader

```python
from SimplerLLM.tools.generic_loader import load_content

text_file = load_content("file.txt")

print(text_file.content)

```

#### Calling any RapidAPI API

```python
from  SimplerLLM.tools.rapid_api import RapidAPIClient

api_url = "https://domain-authority1.p.rapidapi.com/seo/get-domain-info"
api_params = {
    'domain': 'learnwithhasan.com',
}

api_client = RapidAPIClient()  # API key read from environment variable
response = api_client.call_api(api_url, method='GET', params=api_params)


```

#### Prompt Template Builder

```python
from SimplerLLM.prompts.prompt_builder import create_multi_value_prompts,create_prompt_template

basic_prompt = "Generate 5 titles for a blog about {topic} and {style}"

prompt_template = pr.create_prompt_template(basic_prompt)

prompt_template.assign_parms(topic = "marketing",style = "catchy")

print(prompt_template.content)


## working with multiple value prompts
multi_value_prompt_template = """Hello {name}, your next meeting is on {date}.
 and bring a {object} wit you"""

params_list = [
     {"name": "Alice", "date": "January 10th", "object" : "dog"},
     {"name": "Bob", "date": "January 12th", "object" : "bag"},
     {"name": "Charlie", "date": "January 15th", "object" : "pen"}
]


multi_value_prompt = create_multi_value_prompts(multi_value_prompt_template)
generated_prompts = multi_value_prompt.generate_prompts(params_list)

print(generated_prompts[0])

```

## Chunking Functions

We have introduced new functions to help you split texts into manageable chunks based on different criteria. These functions are part of the chunker tool.

### chunk_by_max_chunk_size

This function splits text into chunks with a maximum size, optionally preserving sentence structure.

### chunk_by_sentences

This function splits the text into chunks based on sentences.

### chunk_by_paragraphs

This function splits text into chunks based on paragraphs.

### chunk_by_semantics

This functions splits text into chunks based on semantics.

Example

```python
from SimplerLLM.tools import text_chunker as chunker

blog_url = "https://www.semrush.com/blog/digital-marketing/"
blog_post = loader.load_content(blog_url)

text = blog_post.content

chunks = chunker.chunk_by_max_chunk_size(text, 100, True)



```

### Next Updates

- Adding More Tools
- Interacting With Local LLMs
- Prompt Optimization
- Response Evaluation
- GPT Trainer
- Document Chunker
- Advanced Document Loader
- Integration With More Providers
- Simple RAG With SimplerVectors
- Integration with Vector Databases
- Agent Builder
- LLM Server 
