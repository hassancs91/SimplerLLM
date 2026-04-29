# Prompts

Build prompt templates, manage conversation messages, and fetch prompts from the Prompt Hub.

## Prompt Templates

### Basic Template

Create a reusable template with dynamic placeholders:

```python
from SimplerLLM.prompts.prompt_builder import create_prompt_template

template = create_prompt_template("Explain {topic} in {style} terms.")
prompt = template.assign_parms(topic="machine learning", style="simple")
print(prompt)  # "Explain machine learning in simple terms."
```

Update the template at any time:

```python
template.update_template("Summarize {topic} for a {audience} audience.")
prompt = template.assign_parms(topic="AI safety", audience="technical")
```

### Multiple Values

Generate multiple prompts from one template:

```python
from SimplerLLM.prompts.prompt_builder import create_multi_value_prompts

template = create_multi_value_prompts("Write a {length} {type} about {subject}.")

prompts = template.generate_prompts([
    {"length": "short", "type": "blog post", "subject": "AI"},
    {"length": "long", "type": "article", "subject": "machine learning"},
])

for prompt in prompts:
    print(prompt)
```

## Messages Template

Build validated conversation message lists for chat-based LLMs:

```python
from SimplerLLM.prompts.messages_template import MessagesTemplate

msgs = MessagesTemplate()
msgs.add_user_message("What is Python?")
msgs.add_assistant_message("Python is a programming language.")
msgs.add_user_message("What is it used for?")

messages = msgs.get_messages()
# [
#   {"role": "user", "content": "What is Python?"},
#   {"role": "assistant", "content": "Python is a programming language."},
#   {"role": "user", "content": "What is it used for?"}
# ]
```

Validate message structure before sending:

```python
is_valid, message = msgs.validate_alternation()
print(is_valid)   # True
print(message)    # "MessageTemplate is valid."
```

> **Note:** Messages must start and end with a user message, and roles must alternate between user and assistant.

| Method | Description |
|--------|-------------|
| `add_user_message(content)` | Append a user message |
| `add_assistant_message(content)` | Append an assistant message |
| `validate_alternation()` | Returns `(bool, str)` — validity and message |
| `get_messages()` | Returns validated message list |
| `get_last_message()` | Returns last message or `None` |
| `prepend_messages(messages_list)` | Prepend messages to the start |

## Prompt Hub

Fetch and manage prompts from the SimplerLLM Prompt Hub API.

### Setup

Set your API key in `.env`:

```env
SIMPLERLLM_API_KEY=your_api_key
```

### Fetch a Prompt

```python
from SimplerLLM.prompts.hub import fetch_prompt_from_hub

prompt = fetch_prompt_from_hub("your_prompt_id")

prompt.set_variables(target_audience="developers", tone="friendly")
formatted = prompt.get_formatted_prompt()
print(formatted)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_id` | `str` | — | Unique prompt identifier |
| `api_key` | `str` | `None` | API key (falls back to `SIMPLERLLM_API_KEY` env var) |

### List Prompts

```python
from SimplerLLM.prompts.hub import list_prompts_from_hub

prompts = list_prompts_from_hub()

for p in prompts:
    print(f"{p.name} (v{p.version}) - {p.description}")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `include_shared` | `bool` | `True` | Include shared prompts in results |

### Fetch Specific Version

```python
from SimplerLLM.prompts.hub import fetch_prompt_version_from_hub

prompt = fetch_prompt_version_from_hub("your_prompt_id", version=2)
print(prompt.template)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_id` | `str` | — | Unique prompt identifier |
| `version` | `int` | — | Version number to fetch |
| `api_key` | `str` | `None` | API key (falls back to env var) |

### ManagedPrompt

The object returned by `fetch_prompt_from_hub()`:

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Prompt ID |
| `name` | `str` | Prompt name |
| `description` | `str` | Prompt description |
| `template` | `str` | Raw template string |
| `variables` | `list` | List of prompt variables |
| `tags` | `list` | Tags |
| `version` | `int` | Version number |

| Method | Description |
|--------|-------------|
| `set_variable(name, value)` | Set a single variable |
| `set_variables(**kwargs)` | Set multiple variables at once |
| `get_formatted_prompt()` | Returns template with variables substituted |
