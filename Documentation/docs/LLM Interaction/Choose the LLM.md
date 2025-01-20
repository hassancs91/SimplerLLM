---
sidebar_position: 2
--- 

# Choose the LLM Provider

This section of the documentation covers the methods and functionalities provided by SimplerLLM to interact with the different Large Language Models (LLMs). 

The library simplifies the process of integrating AI capabilities into your projects, making it straightforward to send prompts, receive responses, and handle data from different LLM providers.

## 1. Creating an LLM Instance and Configuring API Keys

To interact with an LLM, you first need to create an instance of the model. This process involves specifying which LLM provider you're using and which model you want to interact with.

Here's an example usage of the OpenAI API leveraging the *gpt-3.5-turbo* model.

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create an instance of an LLM
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")
```

However, we'll need to setup our API keys to use the LLM. We have 2 ways:

### Add them to a `.env` file (Better Approach):

Set up a a `.env` file at the root of your project. This file would include your API keys and any other global variable you'll use in your scripts.

As an example, here's how the `.env` file would look like if we're planning to use the OpenAI LLM:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

The script will automatically load these enviromental variables and use them when you want to generate anything. 

### Pass them as a function parameter:

If you don't want to create a `.env` file you can simply pass your API key as a parameter when creating an LLM instance which would look like this:

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create an instance of an LLM
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo", api_key="your_api_key")
```

## 2. Generating Responses

Once you have an LLM instance, you can generate responses by providing prompts. Here's how to use the `generate_response` function:

```python
response = llm_instance.generate_response(
    prompt="Explain the theory of relativity",
    messages=None,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.6,
    max_tokens=500,
    top_p=0.8,
    full_response=False
)
print(response)
```

### Function Parameters
- `prompt`: The question or statement to send to the LLM.
- `messages`: (optional) List of message objects for structured conversations. For example, when creating a chatbot you'll need to save the chat history so everytime you add the new response of the api to the messages and send it with the new api call.
- `system_prompt` (optional): Guides the model's tone and approach in generating responses. 
- `max_tokens` (optional): The maximum number of tokens to generate.
- `temperature` (optional): Adjusts the diversity and predictability of the model's responses.
- `top_p` (optional): The close to 0 the more the randomness of the response, where it limits the selection of the top probable words.
- `full_response` (optional): If set to True, returns the complete response object from the API, including additional metadata.

If you don't set any of the optional parameters, they'll take the default values set in the library which are:
- `messages`= None
- `system_prompt`= "You are a helpful AI Assistant"
- `max_tokens`= 300
- `temperature`= 0.7
- `top_p`= 1.0
- `full_response`= False

## 3. Prompt Caching

Prompt caching is useful for optimizing performance and reducing API calls which decreases your costs. However, each LLM company has its way of implementing the cache into their API endpoints, so to use caching each LLM Provider has its own syntax:

### OpenAI
Caching is automatically built into their API infrastructure. This means you don’t need to set up or manage caching yourself when using OpenAI models, so keep your function calls the same as shown in the section above. 

### Anthropic
Caching involves adding 2 new parameters to the `generate_response` function which are:
- `prompt_caching`: (optional, boolean) Determines whether you want to use prompt caching or not.
- `cached_input`: (optional) The specified text you want to cache.

If you don't set the above parameters yourself they take the following default values:
- `prompt_caching` = False
- `cached_input` = ""

### Google Gemini
The caching process with Gemini Models is managed differently that the other 2 LLM providers, where you'll have to use an additional function `create_cache` function that generates a unique cache ID. This function is used to create a cache for a specific input, and it returns this specific cache ID, which should be passed to the `generate_response` function. 

Make sure when you want to use caching with Gemini to use a stable model when creating the LLM instance, because it doesnt work with unstable models. [Visit this page to check which models are stable](https://ai.google.dev/gemini-api/docs/models/gemini)

Here's an example of using caching with Gemini:

```python
llm_instance = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-1.5-flash-001") #This is a stable model
cache_id = llm_instance.create_cache(cached_input = "THE INPUT YOU WANT TO CACHE")
response = llm_instance.generate_response(
    prompt="Explain the theory of relativity",
    system_prompt="You are a helpful AI Assistant",
    temperature=0.6,
    max_tokens=500,
    top_p=0.8,
    full_response=False,
    prompt_caching=True,
    cache_id=cache_id 
)
print(response)
```

## 4. Handling Retries

If you want to handle how many time's the function retries after a failed api call, and the delay between each retry you'll need to adjust the enviromental variables that handle this.

The `MAX_RETRIES` and `RETRY_DELAY` are by default set to 3 and 2 respectively, so if you plan to change them you'll need to go to your `.env` file:

```
MAX_RETRIES=5
RETRY_DELAY=3
```

Now, if an API request fails, the library will retry the request a maximum of 5 times if it keeps failing with a delay of 3 seconds between each retry.

## 4. API call for each LLM provider

Each LLM provider requires a slight change in the parameters when creating the llm instance, but the most important thing is adding the API key correctly in the `.env` file. So, when you plan on using multiple providers make sure you [set all the keys needed in a correct format](https://docs.simplerllm.com/#2-set-up-your-environment-env-file) 

Anyways, here's a brief overview and example usage for each provider:

### OpenAI

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")

response = llm_instance.generate_response(prompt="generate a 5 words sentence")
print(response)
```

### Google Gemini

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm_instance = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-1.5-flash")

response = llm_instance.generate_response(prompt="generate a 5 words sentence")
print(response)
```

### Anthropic Claude

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm_instance = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20240620")

response = llm_instance.generate_response(prompt="generate a 5 words sentence")
print(response)
```

### For Ollama (Local Model) - The Phi Model

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm_instance = LLM.create(provider=LLMProvider.OLLAMA, model_name="phi")

response = llm_instance.generate_response(prompt="generate a 5 words sentence")
print(response)
```
