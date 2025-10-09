"""
Example: YouTube Video Summarizer using MiniAgent Flow

This example demonstrates a simple 2-step flow:
1. Use the YouTube transcript tool to get the video transcript
2. Use LLM to summarize the transcript
"""

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.flow import MiniAgent

# Initialize LLM instance
llm_instance = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o"
)

# Create a MiniAgent for YouTube summarization
youtube_agent = MiniAgent(
    name="YouTube Summarizer",
    llm_instance=llm_instance,
    system_prompt="You are a helpful assistant that creates concise and informative summaries.",
    max_steps=3,
    verbose=True  # Show detailed execution logs
)

# Add steps to the flow
# Step 1: Get the YouTube transcript
youtube_agent.add_step(
    step_type="tool",
    tool_name="youtube_transcript"
)

# Step 2: Summarize the transcript
youtube_agent.add_step(
    step_type="llm",
    prompt="Please provide a concise summary of this video transcript in 3-5 bullet points:\n\n{previous_output}",
    params={"max_tokens": 500}
)

# Run the flow
if __name__ == "__main__":
    # Example YouTube URL
    video_url = "https://www.youtube.com/watch?v=S9FlxFv9dxg&pp=ugUEEgJlbg%3D%3D"

    print(f"\nSummarizing video: {video_url}\n")

    # Execute the flow
    result = youtube_agent.run(video_url)

    # Print results
    print("\n" + "="*60)
    print("FLOW EXECUTION RESULT")
    print("="*60)
    print(f"Agent: {result.agent_name}")
    print(f"Success: {result.success}")
    print(f"Total Duration: {result.total_duration_seconds:.2f}s")
    print(f"Total Steps: {result.total_steps}")

    if result.success:
        print("\n" + "-"*60)
        print("FINAL SUMMARY:")
        print("-"*60)
        print(result.final_output)
    else:
        print(f"\nError: {result.error}")

    # Print detailed step information (optional)
    print("\n" + "-"*60)
    print("DETAILED STEP INFORMATION:")
    print("-"*60)
    for step in result.steps:
        print(f"\nStep {step.step_number}: {step.step_type.upper()}")
        if step.tool_used:
            print(f"  Tool: {step.tool_used}")
        if step.prompt_used:
            print(f"  Prompt: {step.prompt_used[:100]}...")
        print(f"  Duration: {step.duration_seconds:.2f}s")
        if step.error:
            print(f"  Error: {step.error}")
        else:
            print(f"  Output: {str(step.output_data)[:200]}...")
