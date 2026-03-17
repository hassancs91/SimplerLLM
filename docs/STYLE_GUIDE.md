# Documentation Style Guide

Guidelines for writing SimplerLLM documentation.

## Principles

1. **Simple and direct** - No fluff, marketing speak, or filler words
2. **Show, don't explain** - Code examples over lengthy descriptions
3. **Practical focus** - What users need to do, not theory
4. **Scannable** - Tables, headers, and short paragraphs

## Structure

Each guide should follow this pattern:

```
# Title

One-line description of what this covers.

## Section

Brief intro (1-2 sentences max), then code or table.
```

## Formatting Rules

### Headers
- `#` for page title
- `##` for main sections
- `###` for subsections (use sparingly)

### Code Blocks
- Always specify language: ` ```python `, ` ```bash `, ` ```env `
- Keep examples minimal but complete
- Show the import, the setup, and the result

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
response = llm.generate_response(prompt="Hello")
```

### Tables
Use tables for:
- Parameter lists
- Provider/option mappings
- Environment variables

```markdown
| Parameter | Description |
|-----------|-------------|
| `prompt` | The input text |
| `max_tokens` | Maximum response length |
```

### Notes
Use blockquotes for tips, warnings, or clarifications:

```markdown
> **Note:** This requires an API key.
```

## What to Avoid

- Long introductions before getting to the point
- Explaining what users already know
- Repeating information across sections
- Emojis
- "In this guide, we will..." style intros
- Overly complex examples with unnecessary options

## Example: Good vs Bad

**Bad:**
```markdown
## Introduction to Text Generation

Text generation is a fundamental capability of large language models.
In this section, we will explore how SimplerLLM makes it easy to
generate text using various providers. Let's dive in!
```

**Good:**
```markdown
## Basic Usage

```python
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
response = llm.generate_response(prompt="What is Python?")
```
```

## File Naming

- Use lowercase with hyphens: `quick-start.md`, `environment-setup.md`
- Keep names short and descriptive

## Checklist Before Publishing

- [ ] Title is clear and matches the content
- [ ] First line explains what the page covers
- [ ] Code examples are tested and work
- [ ] Tables are used for lists of 3+ items
- [ ] No unnecessary words or sections
