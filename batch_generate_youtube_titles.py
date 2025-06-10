from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_async
from pymongo import MongoClient
from pydantic import BaseModel, Field
import asyncio
import time
import datetime
import json
import os
from typing import List, Dict, Any, Optional

try:
    from tqdm import tqdm  # For progress bars
except ImportError:
    print("tqdm not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

# Define Pydantic model for title evaluation
class TitleEvaluation(BaseModel):
    relevancy_score: float = Field(..., description="Score from 0-10 indicating how relevant the title is to the given topic", ge=0, le=10)
    clickability_score: float = Field(..., description="Score from 0-10 indicating how likely users are to click on this title", ge=0, le=10)
    overall_score: float = Field(..., description="Overall score from 0-10 combining relevancy and clickability", ge=0, le=10)
    feedback: str = Field(..., description="Brief explanation of the scores and suggestions for improvement")

# Configuration class
class Config:
    # User-defined topic
    TOPIC = "[how I built a Microsaas in 48 hours - step by step]"
    
    # Performance settings
    BATCH_SIZE = 10  # Number of frameworks to process concurrently
    MAX_RETRIES = 3  # Maximum number of retries for failed requests
    RATE_LIMIT = 10  # Maximum requests per minute (adjust based on your API limits)
    
    # MongoDB settings
    MONGO_CONNECTION_STRING = "mongodb://root:SMgM242EzXWK8RV5eqjWGl2Yg61NMHoQkaVPIIE2zyXjhWYDVwbv0OjBWxe67n0k@62.171.145.173:5432/?directConnection=true"
    MONGO_DB = "test"
    MONGO_COLLECTION = "youtube_title_framworks"
    
    # Output settings
    OUTPUT_DIR = "generated_titles"
    CHECKPOINT_INTERVAL = 10  # Save progress every N frameworks

# Rate limiter class
class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        
    async def acquire(self):
        now = time.time()
        # Remove timestamps older than the period
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) >= self.max_calls:
            # Wait until the oldest call is outside the period
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        # Add the current timestamp
        self.calls.append(time.time())

# Main class for batch title generation
class BatchTitleGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
        self.rate_limiter = RateLimiter(config.RATE_LIMIT)
        self.all_titles = []
        self.processed_count = 0
        self.start_time = None
        
        # Create output directory if it doesn't exist
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(config.OUTPUT_DIR, f"generated_titles_{timestamp}.json")
        self.checkpoint_file = os.path.join(config.OUTPUT_DIR, f"checkpoint_{timestamp}.json")
    
    async def generate_title(self, framework: str, title_example: str, hook_score: int, index: int) -> Dict[str, Any]:
        """Generate a title for a single framework"""
        await self.rate_limiter.acquire()
        
        try:
            # Create prompt for the LLM
            prompt = f"""
            Create a catchy and engaging YouTube title about {self.config.TOPIC} using the following title framework:
            
            Framework: {framework}
            
            Example of this framework in use: {title_example}
            
            Generate a title that follows this framework pattern but is about {self.config.TOPIC}.
            Only return the title, nothing else.
            """
            
            # Generate title using the LLM
            generated_title = await self.llm_instance.generate_response_async(prompt=prompt)
            generated_title = generated_title.strip()
            
            # Evaluate the generated title
            await self.rate_limiter.acquire()
            
            evaluation_prompt = f"""
            Evaluate the following YouTube title for its effectiveness:
            
            Topic: {self.config.TOPIC}
            Title: {generated_title}
            
            Provide a relevancy score (0-10) indicating how well the title matches the topic.
            Provide a clickability score (0-10) indicating how likely users are to click on this title.
            Calculate an overall score (0-10) based on both relevancy and clickability.
            Include brief feedback explaining the scores and any suggestions for improvement.
            """
            
            # Get structured evaluation using Pydantic model
            evaluation = await generate_pydantic_json_model_async(
                model_class=TitleEvaluation,
                prompt=evaluation_prompt,
                llm_instance=self.llm_instance,
                temperature=0.7,
                system_prompt="You are an expert YouTube title evaluator. Provide honest and helpful feedback."
            )
            
            # Create a dictionary for this title
            title_data = {
                "framework": framework,
                "example": title_example,
                "generated_title": generated_title,
                "hook_score": hook_score,
                "topic": self.config.TOPIC,
                "timestamp": datetime.datetime.now().isoformat(),
                "evaluation": {
                    "relevancy_score": evaluation.relevancy_score,
                    "clickability_score": evaluation.clickability_score,
                    "overall_score": evaluation.overall_score,
                    "feedback": evaluation.feedback
                },
                "original_index": index
            }
            
            return title_data
            
        except Exception as e:
            error_message = f"Error generating or evaluating title for framework '{framework}': {str(e)}"
            print(f"\nError: {error_message}")
            
            # Add error entry
            return {
                "framework": framework,
                "example": title_example,
                "error": str(e),
                "hook_score": hook_score,
                "topic": self.config.TOPIC,
                "timestamp": datetime.datetime.now().isoformat(),
                "original_index": index
            }
    
    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of frameworks concurrently"""
        tasks = []
        for i, doc in enumerate(batch):
            framework = doc.get("Framework", "")
            title_example = doc.get("Title", "")
            hook_score = doc.get("Hook Score", 0)
            original_index = self.processed_count + i
            
            task = self.generate_title(framework, title_example, hook_score, original_index)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results
    
    def save_checkpoint(self):
        """Save current progress to a checkpoint file"""
        checkpoint_data = {
            "processed_count": self.processed_count,
            "all_titles": self.all_titles,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "config": {
                "topic": self.config.TOPIC,
                "batch_size": self.config.BATCH_SIZE,
                "rate_limit": self.config.RATE_LIMIT
            }
        }
        
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """Load progress from a checkpoint file"""
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            self.processed_count = checkpoint_data.get("processed_count", 0)
            self.all_titles = checkpoint_data.get("all_titles", [])
            
            if checkpoint_data.get("start_time"):
                self.start_time = datetime.datetime.fromisoformat(checkpoint_data["start_time"])
            
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False
    
    async def run(self, resume_from: Optional[str] = None):
        """Run the batch title generation process"""
        # Connect to MongoDB
        client = MongoClient(self.config.MONGO_CONNECTION_STRING)
        db = client[self.config.MONGO_DB]
        collection = db[self.config.MONGO_COLLECTION]
        
        # Fetch all frameworks from MongoDB
        documents = list(collection.find())
        total_frameworks = len(documents)
        print(f"Found {total_frameworks} frameworks in the database")
        
        # Load checkpoint if resuming
        if resume_from and self.load_checkpoint(resume_from):
            print(f"Resuming from checkpoint with {self.processed_count} frameworks already processed")
            documents = documents[self.processed_count:]
        else:
            self.start_time = datetime.datetime.now()
        
        # Process in batches
        with tqdm(total=total_frameworks, initial=self.processed_count) as pbar:
            while documents:
                # Get the next batch
                batch = documents[:self.config.BATCH_SIZE]
                documents = documents[self.config.BATCH_SIZE:]
                
                # Process the batch
                batch_results = await self.process_batch(batch)
                
                # Update progress
                self.all_titles.extend(batch_results)
                self.processed_count += len(batch)
                pbar.update(len(batch))
                
                # Save checkpoint periodically
                if self.processed_count % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint()
                
                # Calculate and display statistics
                elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
                frameworks_per_second = self.processed_count / max(1, elapsed)
                remaining = (total_frameworks - self.processed_count) / max(0.1, frameworks_per_second)
                
                # Update progress bar description
                pbar.set_description(f"Processing: {frameworks_per_second:.2f} frameworks/sec, ETA: {remaining/60:.1f} min")
        
        # Close MongoDB connection
        client.close()
        
        # Calculate metadata
        metadata = {
            "topic": self.config.TOPIC,
            "generation_time": datetime.datetime.now().isoformat(),
            "total_frameworks": total_frameworks,
            "successful_generations": sum(1 for t in self.all_titles if "generated_title" in t),
            "average_relevancy_score": sum(t.get("evaluation", {}).get("relevancy_score", 0) for t in self.all_titles if "evaluation" in t) / 
                                      max(1, sum(1 for t in self.all_titles if "evaluation" in t)),
            "average_overall_score": sum(t.get("evaluation", {}).get("overall_score", 0) for t in self.all_titles if "evaluation" in t) / 
                                    max(1, sum(1 for t in self.all_titles if "evaluation" in t)),
            "processing_time_seconds": (datetime.datetime.now() - self.start_time).total_seconds(),
            "frameworks_per_second": total_frameworks / max(1, (datetime.datetime.now() - self.start_time).total_seconds())
        }
        
        # Sort titles by original index to maintain order
        sorted_titles = sorted(self.all_titles, key=lambda x: x.get("original_index", 0))
        
        # Create the final JSON structure
        output_data = {
            "metadata": metadata,
            "titles": sorted_titles
        }
        
        # Write to JSON file
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nAll titles have been generated and saved to {self.output_file}")
        print(f"Total processing time: {metadata['processing_time_seconds']:.2f} seconds")
        print(f"Average processing speed: {metadata['frameworks_per_second']:.2f} frameworks per second")

# Main entry point
async def main():
    # Create configuration
    config = Config()
    
    # Create generator
    generator = BatchTitleGenerator(config)
    
    # Run the generator
    await generator.run()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
