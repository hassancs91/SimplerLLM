from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from pymongo import MongoClient
from pydantic import BaseModel, Field
import time
import datetime
import json

# Define Pydantic model for title evaluation
class TitleEvaluation(BaseModel):
    relevancy_score: float = Field(..., description="Score from 0-10 indicating how relevant the title is to the given topic", ge=0, le=10)
    clickability_score: float = Field(..., description="Score from 0-10 indicating how likely users are to click on this title", ge=0, le=10)
    overall_score: float = Field(..., description="Overall score from 0-10 combining relevancy and clickability", ge=0, le=10)
    feedback: str = Field(..., description="Brief explanation of the scores and suggestions for improvement")

# Initialize LLM
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# MongoDB connection
connection_string = "mongodb://root:SMgM242EzXWK8RV5eqjWGl2Yg61NMHoQkaVPIIE2zyXjhWYDVwbv0OjBWxe67n0k@62.171.145.173:5432/?directConnection=true"
client = MongoClient(connection_string)
db = client["test"]
collection = db["youtube_title_framworks"]

# User-defined topic (change this to generate titles for different topics)
TOPIC = "[top 5 ai automation I use daily and saves me countless hours]"  # <-- Change this to your desired topic

# Fetch all frameworks from MongoDB
documents = list(collection.find())
print(f"Found {len(documents)} frameworks in the database")

# Create a timestamp for the output files
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_output_file = f"generated_titles_{timestamp}.json"

# Create a list to store all generated titles and metadata
all_titles = []

# Process each framework
for i, doc in enumerate(documents):
    framework = doc.get("Framework", "")
    title_example = doc.get("Title", "")
    hook_score = doc.get("Hook Score", 0)  # Get the hook score
    
    print(f"Processing framework {i+1}/{len(documents)}: {framework}")
    
    # Create prompt for the LLM
    prompt = f"""
    Create a catchy and engaging YouTube title about {TOPIC} using the following title framework:
    
    Framework: {framework}
    
    Example of this framework in use: {title_example}
    
    Generate a title that follows this framework pattern but is about {TOPIC}.
    Only return the title, nothing else.
    """
    
    # Generate title using the LLM
    try:
        generated_title = llm_instance.generate_response(prompt=prompt).strip()
        
        # Evaluate the generated title
        evaluation_prompt = f"""
        Evaluate the following YouTube title for its effectiveness:
        
        Topic: {TOPIC}
        Title: {generated_title}
        
        Provide a relevancy score (0-10) indicating how well the title matches the topic.
        Provide a clickability score (0-10) indicating how likely users are to click on this title.
        Calculate an overall score (0-10) based on both relevancy and clickability.
        Include brief feedback explaining the scores and any suggestions for improvement.
        """
        
        # Get structured evaluation using Pydantic model
        evaluation = generate_pydantic_json_model(
            model_class=TitleEvaluation,
            prompt=evaluation_prompt,
            llm_instance=llm_instance,
            temperature=0.7,
            system_prompt="You are an expert YouTube title evaluator. Provide honest and helpful feedback."
        )
        
        # Create a dictionary for this title
        title_data = {
            "framework": framework,
            "example": title_example,
            "generated_title": generated_title,
            "hook_score": hook_score,
            "topic": TOPIC,
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluation": {
                "relevancy_score": evaluation.relevancy_score,
                "clickability_score": evaluation.clickability_score,
                "overall_score": evaluation.overall_score,
                "feedback": evaluation.feedback
            }
        }
        
        # Add to our list
        all_titles.append(title_data)
        
        # Print progress
        print(f"  Generated: {generated_title} (Hook Score: {hook_score}, Overall Score: {evaluation.overall_score})")
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    except Exception as e:
        error_message = f"Error generating or evaluating title for framework '{framework}': {str(e)}"
        print(error_message)
        
        # Add error entry
        title_data = {
            "framework": framework,
            "example": title_example,
            "error": str(e),
            "hook_score": hook_score,
            "topic": TOPIC,
            "timestamp": datetime.datetime.now().isoformat()
        }
        all_titles.append(title_data)

# Close MongoDB connection
client.close()

# Create a metadata object
metadata = {
    "topic": TOPIC,
    "generation_time": datetime.datetime.now().isoformat(),
    "total_frameworks": len(documents),
    "successful_generations": sum(1 for t in all_titles if "generated_title" in t),
    "average_relevancy_score": sum(t.get("evaluation", {}).get("relevancy_score", 0) for t in all_titles if "evaluation" in t) / 
                              max(1, sum(1 for t in all_titles if "evaluation" in t)),
    "average_overall_score": sum(t.get("evaluation", {}).get("overall_score", 0) for t in all_titles if "evaluation" in t) / 
                            max(1, sum(1 for t in all_titles if "evaluation" in t))
}

# Create the final JSON structure
output_data = {
    "metadata": metadata,
    "titles": all_titles
}

# Write to JSON file
with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\nAll titles have been generated and saved to {json_output_file}")
