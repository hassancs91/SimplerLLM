"""
Example: Structured JSON Output with MiniAgent Flow

This example demonstrates using Pydantic models to get structured, validated JSON output
from LLM steps in a flow. This is useful when you need machine-readable data instead of
plain text.

Use cases:
- Extract structured metadata from content
- Parse information into specific formats
- Ensure consistent output structure for downstream processing
"""

from pydantic import BaseModel, Field
from typing import List
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.flow import MiniAgent


# Define Pydantic models for structured outputs

class VideoMetadata(BaseModel):
    """Structured metadata extracted from a video."""
    title: str = Field(description="Main topic or title of the video")
    key_points: List[str] = Field(description="List of 3-5 main points discussed")
    duration_estimate: str = Field(description="Estimated duration category: short, medium, or long")
    target_audience: str = Field(description="Who this video is aimed at")
    main_takeaway: str = Field(description="Single sentence summarizing the main takeaway")


class ArticleSummary(BaseModel):
    """Structured summary of an article or text."""
    headline: str = Field(description="Catchy headline summarizing the content")
    summary: str = Field(description="2-3 sentence summary")
    topics: List[str] = Field(description="List of main topics covered")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    word_count_estimate: int = Field(description="Estimated word count of original text")


class TaskBreakdown(BaseModel):
    """Structured breakdown of a complex task."""
    task_name: str = Field(description="Name of the task")
    subtasks: List[str] = Field(description="List of 3-7 subtasks needed to complete the task")
    estimated_time: str = Field(description="Estimated time to complete: quick, moderate, or lengthy")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")
    prerequisites: List[str] = Field(description="Required knowledge or tools needed")


# Example 1: YouTube Video Metadata Extraction
def example_youtube_metadata_extraction():
    """Extract structured metadata from a YouTube video."""
    print("\n" + "="*80)
    print("EXAMPLE 1: YouTube Video Metadata Extraction")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    agent = MiniAgent(
        name="Video Metadata Extractor",
        llm_instance=llm,
        system_prompt="You are an expert at analyzing video content and extracting key information.",
        max_steps=3,
        verbose=True
    )

    # Step 1: Get transcript
    agent.add_step(
        step_type="tool",
        tool_name="youtube_transcript"
    )

    # Step 2: Extract structured metadata using Pydantic model
    agent.add_step(
        step_type="llm",
        prompt="""Analyze this video transcript and extract structured metadata.

Transcript:
{previous_output}

Extract the following information and format it as JSON.""",
        output_model=VideoMetadata,  # This ensures validated JSON output!
        max_retries=3,
        params={"max_tokens": 1000}
    )

    # Example video URL
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    result = agent.run(video_url)

    if result.success:
        print("\n" + "-"*80)
        print("EXTRACTED METADATA (Structured JSON):")
        print("-"*80)

        # result.final_output is now a VideoMetadata object!
        metadata = result.final_output

        print(f"Title: {metadata.title}")
        print(f"Duration: {metadata.duration_estimate}")
        print(f"Target Audience: {metadata.target_audience}")
        print(f"\nKey Points:")
        for i, point in enumerate(metadata.key_points, 1):
            print(f"  {i}. {point}")
        print(f"\nMain Takeaway: {metadata.main_takeaway}")

        # You can also convert to dict or JSON
        print("\n" + "-"*80)
        print("As JSON:")
        print("-"*80)
        print(metadata.model_dump_json(indent=2))
    else:
        print(f"\nError: {result.error}")


# Example 2: Text Summarization with Structured Output
def example_text_summarization():
    """Summarize text and return structured data."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Text Summarization with Structured Output")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    agent = MiniAgent(
        name="Article Summarizer",
        llm_instance=llm,
        system_prompt="You are an expert at analyzing and summarizing articles.",
        max_steps=2,
        verbose=True
    )

    # Single step: Analyze text and output structured summary
    agent.add_step(
        step_type="llm",
        prompt="""Analyze the following text and provide a structured summary:

{input}

Extract key information and format as JSON.""",
        output_model=ArticleSummary,
        max_retries=3,
        params={"max_tokens": 800}
    )

    # Sample text
    sample_text = """
    Artificial Intelligence has revolutionized many industries in recent years.
    From healthcare to finance, AI systems are being deployed to improve efficiency
    and decision-making. Machine learning algorithms can now diagnose diseases,
    predict market trends, and even create art. However, concerns about AI ethics,
    bias, and job displacement remain significant challenges that society must address.
    As AI continues to advance, finding the right balance between innovation and
    responsible development will be crucial for ensuring these technologies benefit
    all of humanity.
    """

    result = agent.run(sample_text)

    if result.success:
        print("\n" + "-"*80)
        print("STRUCTURED SUMMARY:")
        print("-"*80)

        summary = result.final_output

        print(f"Headline: {summary.headline}")
        print(f"Sentiment: {summary.sentiment}")
        print(f"Estimated Word Count: {summary.word_count_estimate}")
        print(f"\nSummary: {summary.summary}")
        print(f"\nTopics Covered:")
        for topic in summary.topics:
            print(f"  - {topic}")
    else:
        print(f"\nError: {result.error}")


# Example 3: Task Breakdown Flow
def example_task_breakdown():
    """Break down a complex task into structured subtasks."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Complex Task Breakdown")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    agent = MiniAgent(
        name="Task Analyzer",
        llm_instance=llm,
        system_prompt="You are an expert project manager who breaks down complex tasks.",
        max_steps=2,
        verbose=True
    )

    agent.add_step(
        step_type="llm",
        prompt="""Break down this task into structured subtasks:

Task: {input}

Provide a detailed breakdown with subtasks, time estimate, difficulty, and prerequisites.""",
        output_model=TaskBreakdown,
        max_retries=3,
        params={"max_tokens": 1000}
    )

    task = "Build a web application for managing personal finances"

    result = agent.run(task)

    if result.success:
        print("\n" + "-"*80)
        print("TASK BREAKDOWN:")
        print("-"*80)

        breakdown = result.final_output

        print(f"Task: {breakdown.task_name}")
        print(f"Difficulty: {breakdown.difficulty}")
        print(f"Estimated Time: {breakdown.estimated_time}")

        print(f"\nPrerequisites:")
        for prereq in breakdown.prerequisites:
            print(f"  - {prereq}")

        print(f"\nSubtasks:")
        for i, subtask in enumerate(breakdown.subtasks, 1):
            print(f"  {i}. {subtask}")

        # Demonstrate JSON serialization for API responses
        print("\n" + "-"*80)
        print("JSON for API Response:")
        print("-"*80)
        print(breakdown.model_dump_json(indent=2))
    else:
        print(f"\nError: {result.error}")


# Example 4: Multi-step flow with JSON output
def example_multi_step_json_flow():
    """
    Advanced example: Multi-step flow where intermediate step produces JSON
    and final step uses it.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Step Flow with JSON Output")
    print("="*80 + "\n")

    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514"
    )

    agent = MiniAgent(
        name="Content Analyzer & Recommender",
        llm_instance=llm,
        system_prompt="You are a helpful content analysis assistant.",
        max_steps=3,
        verbose=True
    )

    # Step 1: Analyze text and output structured data
    agent.add_step(
        step_type="llm",
        prompt="Analyze this text: {input}",
        output_model=ArticleSummary,
        params={"max_tokens": 800}
    )

    # Step 2: Use the structured output to generate recommendations
    agent.add_step(
        step_type="llm",
        prompt="""Based on this analysis, provide 3 reading recommendations:

Analysis:
{previous_output}

Give specific article or book recommendations that relate to these topics.""",
        params={"max_tokens": 500}
    )

    text = "Machine learning is transforming how we approach data analysis and prediction."

    result = agent.run(text)

    if result.success:
        print("\n" + "-"*80)
        print("STEP 1 OUTPUT (Structured JSON):")
        print("-"*80)
        print(result.steps[0].output_data)

        print("\n" + "-"*80)
        print("STEP 2 OUTPUT (Text Recommendations):")
        print("-"*80)
        print(result.final_output)
    else:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    # Run examples
    # Note: Comment out examples you don't want to run

    # Example 1: YouTube metadata extraction
    # example_youtube_metadata_extraction()

    # Example 2: Text summarization with structured output
    example_text_summarization()

    # Example 3: Task breakdown
    # example_task_breakdown()

    # Example 4: Multi-step flow with JSON
    # example_multi_step_json_flow()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("\nKey Benefits of JSON Output Mode:")
    print("  ✓ Type-safe, validated output")
    print("  ✓ Easy to parse and process programmatically")
    print("  ✓ Automatic retry on validation failures")
    print("  ✓ Perfect for APIs and data pipelines")
    print("  ✓ Works seamlessly with existing SimplerLLM tools")
