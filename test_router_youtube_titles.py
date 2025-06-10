from SimplerLLM.language.llm_router import LLMRouter
from SimplerLLM.language.llm import LLM, LLMProvider
from pymongo import MongoClient
import math

# Initialize LLM
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# MongoDB connection
connection_string = "mongodb://root:SMgM242EzXWK8RV5eqjWGl2Yg61NMHoQkaVPIIE2zyXjhWYDVwbv0OjBWxe67n0k@62.171.145.173:5432/?directConnection=true"
client = MongoClient(connection_string)
db = client["test"]
collection = db["youtube_title_framworks"]

# Fetch all documents from MongoDB
documents = list(collection.find())
print(f"Total documents: {len(documents)}")

# Define batch size
BATCH_SIZE = 20

# Calculate number of batches
num_batches = math.ceil(len(documents) / BATCH_SIZE)
print(f"Processing in {num_batches} batches of {BATCH_SIZE} records each")

# Store top choices from each batch
all_top_choices = []

# User input
user_input = "I have a YouTube video idea, I want you to pick the best Title Frameworks that garantees the best CTR"

# Process each batch
for batch_num in range(num_batches):
    print(f"\nProcessing batch {batch_num + 1}/{num_batches}")
    
    # Get the current batch of documents
    start_idx = batch_num * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(documents))
    batch_documents = documents[start_idx:end_idx]
    
    # Create a new router for this batch
    batch_router = LLMRouter(llm_instance=llm_instance, confidence_threshold=0.75)
    
    # Format documents for the router
    choices_data = []
    for i, doc in enumerate(batch_documents):
        framework = doc.get("Framework", "")
        title = doc.get("Title", "")
        
        # Add a unique identifier to make each framework unique
        unique_framework = f"{framework} (ID: {start_idx + i + 1})"
        
        # Add to choices_data with appropriate format
        choices_data.append((
            unique_framework,  # This is the content
            {"title example": title}  # This is the metadata
        ))
    
    # Add choices to the batch router
    batch_router.add_choices(choices_data)
    
    # Get top 3 responses from this batch
    top_batch_responses = batch_router.route_top_k(user_input, k=3)
    
    # Store the top choices from this batch
    for response_item in top_batch_responses:
        choice_content, metadata = batch_router.get_choice(response_item.selected_index)
        all_top_choices.append((
            choice_content,
            metadata,
            response_item.confidence_score,
            response_item.reasoning
        ))
    
    print(f"Selected {len(top_batch_responses)} top choices from batch {batch_num + 1}")

# Close MongoDB connection
client.close()

print(f"\nTotal top choices from all batches: {len(all_top_choices)}")

# Create final router with all top choices
final_router = LLMRouter(llm_instance=llm_instance, confidence_threshold=0.75)

# Add all top choices to the final router
final_choices_data = []
for i, (content, metadata, _, _) in enumerate(all_top_choices):
    final_choices_data.append((content, metadata))

final_router.add_choices(final_choices_data)

# Get final top 10 choices
final_top_responses = final_router.route_top_k(user_input, k=10)

# Display final results
print("\n=== FINAL TOP 10 CHOICES ===")
for i, response_item in enumerate(final_top_responses):
    confidence = response_item.confidence_score
    reasoning = response_item.reasoning
    
    actual_choice_content, _ = final_router.get_choice(response_item.selected_index)
    
    # Find the original confidence and reasoning from the batch processing
    original_confidence = "N/A"
    original_reasoning = "N/A"
    for content, _, conf, reason in all_top_choices:
        if content == actual_choice_content:
            original_confidence = conf
            original_reasoning = reason
            break
    
    print(f"\nChoice #{i+1}")
    print(f"Content: {actual_choice_content}")
    print(f"Final confidence: {confidence}")
    print(f"Final reasoning: {reasoning}")
    print(f"Original batch confidence: {original_confidence}")
    print(f"Original batch reasoning: {original_reasoning}")
