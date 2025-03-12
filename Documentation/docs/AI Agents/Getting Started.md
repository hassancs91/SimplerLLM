---
sidebar_position: 1
---

# Getting Started with Agents

SimplerLLM now includes a powerful Agent Builder that allows you to create AI agents that can use tools, maintain memory, and generate structured responses.

## What are Agents?

Agents are AI systems that can:
1. Understand user requests
2. Decide what actions to take
3. Use tools to gather information or perform tasks
4. Generate helpful responses

The SimplerLLM Agent Builder provides a flexible framework for creating agents that leverage the existing LLM and tools functionality.

## Basic Usage

Here's a simple example of creating and using an agent:

```python
from SimplerLLM.language import OpenAILLM, LLMProvider
from SimplerLLM.agents import Agent
from SimplerLLM.tools.serp import search_with_duck_duck_go

# Create an LLM instance
llm = OpenAILLM(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",
    temperature=0.7
)

# Create an agent
agent = Agent(llm=llm)

# Add a search tool
agent.add_tool(
    name="web_search",
    func=search_with_duck_duck_go,
    description="Search the web for information",
    parameters={
        "query": "The search query string",
        "max_results": "Maximum number of results to return (default: 5)"
    }
)

# Run the agent
response = agent.run("What were the major tech announcements this week?")
print(response)
```

## Key Components

### Agent Class

The `Agent` class is the core of the Agent Builder. It handles:
- Maintaining conversation memory
- Managing tools
- Generating structured responses using LLMs
- Executing tools based on agent decisions

```python
agent = Agent(
    llm,                    # LLM instance
    memory=None,            # Optional AgentMemory instance
    tools=None,             # Optional dictionary of tools
    system_prompt=None,     # Optional system prompt
    verbose=False           # Whether to print verbose output
)
```

### AgentMemory

The `AgentMemory` class stores conversation history and other agent state:

```python
memory = AgentMemory(max_tokens=4000)  # Approximate token limit

# Add messages
memory.add_user_message("Hello, agent!")
memory.add_assistant_message("Hello! How can I help you?")
memory.add_system_message("You are a helpful assistant.")

# Get messages
messages = memory.get_messages()  # List of message dictionaries
history = memory.get_chat_history()  # Formatted string of chat history
```

### Adding Tools

You can add any function as a tool for the agent to use:

```python
agent.add_tool(
    name="tool_name",           # Name the agent will use to call the tool
    func=your_function,         # The function to call
    description="Description",  # Description of what the tool does
    parameters={                # Dictionary of parameter descriptions
        "param1": "Description of parameter 1",
        "param2": "Description of parameter 2"
    }
)
```

## Advanced Usage

### Using ReliableLLM

You can use `ReliableLLM` for fallback support:

```python
from SimplerLLM.language import OpenAILLM, GeminiLLM, ReliableLLM, LLMProvider

# Create primary and secondary LLM instances
primary_llm = OpenAILLM(provider=LLMProvider.OPENAI, model_name="gpt-4o")
secondary_llm = GeminiLLM(provider=LLMProvider.GEMINI, model_name="gemini-pro")

# Create a ReliableLLM instance
reliable_llm = ReliableLLM(primary_llm, secondary_llm)

# Create an agent with the reliable LLM
agent = Agent(llm=reliable_llm)
```

### Custom System Prompts

You can customize the agent's behavior with system prompts:

```python
system_prompt = """You are an AI assistant specialized in data analysis.
When responding to queries:
1. Determine if you need to analyze data
2. Use Python code execution when calculations are needed
3. Provide clear explanations of your analysis
"""

agent = Agent(llm=llm, system_prompt=system_prompt)
```

### Multiple Tools

Agents can use multiple tools to solve complex problems:

```python
# Add a web search tool
agent.add_tool(
    name="web_search",
    func=search_with_duck_duck_go,
    description="Search the web for information",
    parameters={"query": "The search query string"}
)

# Add a Python execution tool
agent.add_tool(
    name="execute_python",
    func=execute_python_code,
    description="Execute Python code and return the result",
    parameters={"input_code": "Python code to execute"}
)

# Add a file saving tool
agent.add_tool(
    name="save_file",
    func=save_text_to_file,
    description="Save text to a file",
    parameters={
        "text": "Text content to save",
        "filename": "Name of the file to save to"
    }
)
```

## How It Works

The Agent Builder uses the following process:

1. The user provides input to the agent
2. The agent adds the input to its memory
3. The agent creates a prompt that includes:
   - Available tools
   - Conversation history
   - Instructions for generating a structured response
4. The agent uses the LLM to generate a structured response using Pydantic models
5. If the response includes a tool call, the agent executes the tool and continues
6. If the response includes a final answer, the agent returns it to the user

This process allows the agent to have multi-step reasoning and tool use capabilities.

## Examples

Check out the examples directory for more detailed examples:
- `agent_example.py` - Basic agent usage
- `advanced_agent_example.py` - Advanced agent with multiple tools and ReliableLLM

## Next Steps

- Explore the different tools available in SimplerLLM
- Create custom tools for your specific use cases
- Experiment with different system prompts to customize agent behavior
