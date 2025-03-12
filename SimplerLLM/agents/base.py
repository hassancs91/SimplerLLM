from typing import Dict, Any, Optional, List, Union, Callable, Type
from pydantic import BaseModel

from SimplerLLM.language.llm import LLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from .memory import AgentMemory
from .models import AgentAction, ToolCall

class Agent:
    """
    Base Agent class that can use tools and generate responses using LLMs.
    
    This agent can:
    1. Maintain conversation memory
    2. Use tools provided by the user
    3. Generate structured responses using Pydantic models
    """
    
    def __init__(
        self,
        llm: LLM,
        memory: Optional[AgentMemory] = None,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the agent.
        
        Args:
            llm (LLM): The LLM instance to use for generating responses.
            memory (AgentMemory, optional): Memory system for the agent.
            tools (Dict[str, Dict[str, Any]], optional): Dictionary of tools.
            system_prompt (str, optional): System prompt for the agent.
            verbose (bool, optional): Whether to print verbose output.
        """
        self.llm = llm
        self.memory = memory or AgentMemory()
        self.tools = tools or {}
        self.verbose = verbose
        
        # Set default system prompt if none provided
        default_system_prompt = """You are a helpful AI assistant with access to tools.
Follow these steps for each user request:
1. Think carefully about what the user is asking
2. Determine if you need to use a tool or can answer directly
3. If using a tool, specify the tool name and parameters exactly as defined
4. If answering directly, provide a helpful response

Your output must be a valid JSON object matching the specified format."""

        if system_prompt:
            self.memory.add_system_message(system_prompt)
        else:
            self.memory.add_system_message(default_system_prompt)
            
    def add_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a tool to the agent's toolkit.
        
        Args:
            name (str): Name of the tool.
            func (Callable): Function to call when the tool is used.
            description (str): Description of what the tool does.
            parameters (Dict[str, str], optional): Dictionary of parameter names to descriptions.
        """
        self.tools[name] = {
            "func": func,
            "description": description,
            "parameters": parameters or {}
        }
        
    def run(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Main execution loop for the agent.
        
        Args:
            user_input (str): The user's input.
            max_iterations (int, optional): Maximum number of iterations to run.
                This prevents infinite loops if the agent keeps calling tools.
                
        Returns:
            str: The agent's final response.
        """
        # Add user input to memory
        self.memory.add_user_message(user_input)
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # Create the agent prompt
            prompt = self._create_prompt()
            
            if self.verbose:
                print(f"Iteration {iteration} - Generating response...")
                
            # Generate agent response using Pydantic model
            agent_response = generate_pydantic_json_model(
                model_class=AgentAction,
                prompt=prompt,
                llm_instance=self.llm,
                temperature=0.7,
                max_tokens=1500,
                system_prompt=self._get_system_prompt()
            )
            
            # Process the response
            if isinstance(agent_response, AgentAction):
                # Add agent's thought to memory
                if self.verbose:
                    print(f"Agent thought: {agent_response.thought}")
                self.memory.add_assistant_message(f"Thought: {agent_response.thought}")
                
                # Check if agent wants to use a tool
                if agent_response.action:
                    tool_name = agent_response.action.tool_name
                    parameters = agent_response.action.parameters
                    
                    if self.verbose:
                        print(f"Agent wants to use tool: {tool_name} with parameters: {parameters}")
                    
                    # Execute the tool if it exists
                    if tool_name in self.tools:
                        try:
                            tool_result = self.tools[tool_name]["func"](**parameters)
                            observation = f"Tool {tool_name} returned: {tool_result}"
                        except Exception as e:
                            observation = f"Error executing tool {tool_name}: {str(e)}"
                        
                        if self.verbose:
                            print(f"Tool observation: {observation}")
                            
                        # Add observation to memory
                        self.memory.add_assistant_message(f"Observation: {observation}")
                        
                        # Continue to next iteration
                        continue
                    else:
                        error_msg = f"Tool {tool_name} not found. Available tools: {', '.join(self.tools.keys())}"
                        self.memory.add_assistant_message(f"Error: {error_msg}")
                        if self.verbose:
                            print(error_msg)
                        return error_msg
                
                # If no tool call, return the final response
                if agent_response.response:
                    final_response = agent_response.response
                    self.memory.add_assistant_message(final_response)
                    if self.verbose:
                        print(f"Final response: {final_response}")
                    return final_response
            
            # Handle errors in response generation
            error_msg = "Failed to generate a valid response"
            self.memory.add_assistant_message(error_msg)
            if self.verbose:
                print(error_msg)
            return error_msg
        
        # If we've reached the maximum number of iterations
        max_iter_msg = f"Reached maximum number of iterations ({max_iterations})"
        self.memory.add_assistant_message(max_iter_msg)
        if self.verbose:
            print(max_iter_msg)
        return max_iter_msg
    
    async def run_async(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Asynchronous version of the main execution loop.
        
        This is a placeholder for future implementation.
        
        Args:
            user_input (str): The user's input.
            max_iterations (int, optional): Maximum number of iterations to run.
                
        Returns:
            str: The agent's final response.
        """
        # This is a placeholder for future async implementation
        # For now, just call the synchronous version
        return self.run(user_input, max_iterations)
    
    def _create_prompt(self) -> str:
        """
        Create the prompt for the agent.
        
        Returns:
            str: The formatted prompt.
        """
        # Tool descriptions
        tool_descriptions = []
        for name, tool in self.tools.items():
            params_str = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            tool_descriptions.append(f"Tool: {name}\nDescription: {tool['description']}\nParameters: {params_str}")
        
        tools_section = "Available Tools:\n" + "\n\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        
        # Get conversation history
        history = self.memory.get_chat_history()
        
        # Create the full prompt
        prompt = f"""
{tools_section}

Conversation History:
{history}

Think step by step about how to respond to the user's request. 
If you need to use a tool, specify which tool and what parameters to use.
If you can answer directly, provide a response.
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        
        Returns:
            str: The system prompt.
        """
        # Find the system message in memory
        for msg in self.memory.get_messages():
            if msg["role"] == "system":
                return msg["content"]
        
        # Default if no system message found
        return """You are a helpful AI assistant with access to tools.
Follow these steps for each user request:
1. Think carefully about what the user is asking
2. Determine if you need to use a tool or can answer directly
3. If using a tool, specify the tool name and parameters exactly as defined
4. If answering directly, provide a helpful response

"""
