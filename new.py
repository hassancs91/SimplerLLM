from SimplerLLM.language.llm import LLM, LLMProvider
from dotenv import load_dotenv
import os
import time

load_dotenv()

def test_vector_storage_and_retrieval():
    llm = LLM(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")

    prompts = [
        "What is artificial intelligence and how does it differ from human intelligence?",
        "Explain the process of machine learning and its key components.",
        "Describe the architecture of deep neural networks and their layers.",
        "What are the applications of natural language processing in everyday technology?",
        "How does computer vision work and what are its real-world applications?",
        "Explain the concept of reinforcement learning and its use in robotics.",
        "What are the ethical concerns surrounding AI development and deployment?",
        "How does transfer learning accelerate AI model development?",
        "Describe the differences between supervised, unsupervised, and semi-supervised learning.",
        "What is the role of big data in advancing AI capabilities?",
        "Explain the concept of explainable AI and why it's important.",
        "How do genetic algorithms work in optimization problems?",
        "What are the challenges in developing artificial general intelligence (AGI)?",
        "Describe the impact of AI on healthcare diagnostics and treatment.",
        "How does AI contribute to autonomous vehicle technology?"
    ]

    print("Storing responses as vectors...")
    start_time = time.time()
    try:
        llm.store_response_as_vector(prompts)
    except Exception as e:
        print("Error occurred:", e)
    end_time = time.time()
    print(f"Responses stored successfully. Time taken: {end_time - start_time:.2f} seconds")

    query_prompts = [
        "What are the fundamental principles of AI?",
        "How do machines learn from data?",
        "Explain the inner workings of neural networks.",
        "What are some practical applications of NLP?",
        "How is AI changing the automotive industry?",
        "What are the moral implications of using AI in decision-making?",
        "How is AI transforming the healthcare sector?",
        "What are the key differences between AI learning paradigms?",
        "How does AI handle complex optimization problems?",
        "What are the challenges in making AI systems more transparent?"
    ]

    print("\nQuerying for similar responses:")
    for query_prompt in query_prompts:
        print(f"\nQuery: {query_prompt}")
        start_time = time.time()
        similar_responses = llm.find_similar_responses(query_prompt)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("Similar responses:")
        for i, response in enumerate(similar_responses, 1):
            print(f"{i}. {response}")

def main():
    print("Starting vector storage and retrieval test...")
    test_vector_storage_and_retrieval()

if __name__ == "__main__":
    main()