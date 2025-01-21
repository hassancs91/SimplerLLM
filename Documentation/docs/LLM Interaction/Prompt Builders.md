---
sidebar_position: 4
---

# Prompt Template Builder

Easily create and manage prompt templates with SimplerLLM. This feature allows you to define templates with dynamic placeholders and populate them with single or multiple sets of parameters.

The `Prompt Template Builder` provides tools to define, customize, and reuse prompt templates. Here's how you can use it:

## Single Value Prompt Template

For basic templates with single sets of parameters:

```python
from SimplerLLM.prompts.prompt_builder import create_prompt_template

# Define your prompt template
basic_prompt = "Generate 5 titles for a blog about {topic} and {style}"

# Create a prompt template
prompt_template = create_prompt_template(basic_prompt)

# Assign values to the parameters
prompt_template.assign_parms(topic="marketing", style="catchy")

# Access the populated prompt
print(prompt_template)
```

This will output the following: **Generate 5 titles for a blog about marketing and catchy**

## Multi-Value Prompt Template

For working with multiple sets of parameters, use the `create_multi_value_prompts` function:

```python
from SimplerLLM.prompts.prompt_builder import create_multi_value_prompts

# Define your multi-value prompt template
multi_value_prompt_template = """Hello {name}, your next meeting is on {date}.
and bring a {object} with you"""

# Define multiple parameter sets
params_list = [
    {"name": "Alice", "date": "January 10th", "object": "dog"},
    {"name": "Bob", "date": "January 12th", "object": "bag"},
    {"name": "Charlie", "date": "January 15th", "object": "pen"}
]

# Create and generate multi-value prompts
multi_value_prompt = create_multi_value_prompts(multi_value_prompt_template)
generated_prompts = multi_value_prompt.generate_prompts(params_list)

# Access the generated prompts
print("This will output the first prompt:", generated_prompts[0])
print("This will output the second prompt:", generated_prompts[1])
print("This will output the third prompt:", generated_prompts[2])
```

This will output the first prompt: **Hello Alice, your next meeting is on January 10th.**
                                   **and bring a dog with you**
This will output the second prompt: **Hello Bob, your next meeting is on January 12th.**
                                   **and bring a bag with you**
This will output the third prompt: **Hello Charlie, your next meeting is on January 15th.**
                                   **and bring a pen with you**

That's how you can benefit from SimplerLLM to make managing and creating prompts Simpler!