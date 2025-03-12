"""
Example of a specialized Research Agent that extends the base Agent class.

This example demonstrates how to create a custom agent that automatically
searches for information and synthesizes the results.
"""

from SimplerLLM.language import OpenAILLM, LLMProvider
from SimplerLLM.agents import Agent, AgentMemory
from SimplerLLM.tools.serp import search_with_duck_duck_go
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    """Model for generating search queries."""
    queries: List[str] = Field(..., description="List of search queries to perform")

class ResearchAgent(Agent):
    """
    A specialized agent for research tasks that automatically searches for information.
    """
    
    def __init__(self, llm, **kwargs):
        # Set a research-focused system prompt
        research_prompt = """You are a research assistant specialized in finding and synthesizing information.
For each query:
1. Identify the key research questions
2. Automatically search for relevant information
3. Synthesize the information into a clear, comprehensive response
4. Cite your sources

Always search for information before attempting to answer from your own knowledge."""

        # Initialize with research prompt
        kwargs["system_prompt"] = kwargs.get("system_prompt", research_prompt)
        super().__init__(llm, **kwargs)
        
        # Add search tool by default
        self.add_tool(
            name="web_search",
            func=search_with_duck_duck_go,
            description="Search the web for information",
            parameters={
                "query": "The search query string",
                "max_results": "Maximum number of results to return (default: 5)"
            }
        )
        
    def run(self, user_input: str, max_iterations: int = 15) -> str:
        """
        Override run to automatically perform searches for research queries.
        """
        # Add user input to memory
        self.memory.add_user_message(user_input)
        
        if self.verbose:
            print(f"Generating search queries for: {user_input}")
            
        # Generate search queries based on the user input using the LLM
        search_queries = self._generate_search_queries(user_input)
        
        if self.verbose:
            print(f"Generated queries: {search_queries}")
        
        # Perform searches and add results to memory
        for query in search_queries:
            if self.verbose:
                print(f"Searching for: {query}")
                
            try:
                search_results = search_with_duck_duck_go(query=query, max_results=3)
                
                # Format the search results
                result_text = "\n\n".join([
                    f"Title: {result.Title}\nURL: {result.URL}\nDescription: {result.Description}"
                    for result in search_results
                ])
                
                # Add search results to memory
                self.memory.add_assistant_message(
                    f"Search results for '{query}':\n{result_text}"
                )
                
                if self.verbose:
                    print(f"Added search results for '{query}' to memory")
                    
            except Exception as e:
                error_msg = f"Error searching for '{query}': {str(e)}"
                if self.verbose:
                    print(error_msg)
                self.memory.add_assistant_message(error_msg)
        
        # Now run the standard agent loop with the search results in memory
        if self.verbose:
            print("Generating final response based on search results...")
            
        return super().run(
            "Based on the search results above, please provide a comprehensive answer to my question.",
            max_iterations
        )
        
    def _generate_search_queries(self, user_input: str) -> List[str]:
        """
        Generate search queries based on the user input using the LLM.
        """
        from SimplerLLM.language.llm_addons import generate_pydantic_json_model
        
        prompt = f"""
Based on the following user question, generate 2-3 effective search queries that would help find relevant information.
The queries should be specific and focused on different aspects of the question.

User question: {user_input}

Generate search queries that would help find comprehensive information to answer this question.
"""
        
        try:
            # Use the LLM to generate search queries
            search_query_model = generate_pydantic_json_model(
                model_class=SearchQuery,
                prompt=prompt,
                llm_instance=self.llm,
                temperature=0.7,
                max_tokens=500,
                system_prompt="You are a helpful assistant that generates effective search queries."
            )
            
            if isinstance(search_query_model, SearchQuery):
                return search_query_model.queries
            else:
                # Fallback if model generation fails
                if self.verbose:
                    print("Failed to generate search queries, using default query")
                return [user_input]
                
        except Exception as e:
            # Fallback to just using the user input as a query
            if self.verbose:
                print(f"Error generating search queries: {str(e)}")
            return [user_input]


def main():
    # Create an LLM instance
    llm = OpenAILLM(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",  # You can change this to any supported model
        temperature=0.7
    )
    
    # Create a research agent with verbose output
    agent = ResearchAgent(llm=llm, verbose=True)
    
    # Example research questions
    research_questions = [
        "What are the environmental impacts of electric vehicles compared to gas vehicles?",
        "What are the latest advancements in quantum computing?",
        "How does intermittent fasting affect metabolism?"
    ]
    
    # Choose one question to demonstrate
    question_index = 0  # Change this to try different questions
    user_query = research_questions[question_index]
    
    print(f"\n\nResearch Question: {user_query}\n")
    
    # Run the research agent
    response = agent.run(user_query)
    
    print(f"\n\nFinal Research Report:\n{response}\n")

if __name__ == "__main__":
    main()
