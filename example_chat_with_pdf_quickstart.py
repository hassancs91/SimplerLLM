"""
Example usage of the ChatWithPDF quickstart module.

This example demonstrates various ways to use the ChatWithPDF class for 
building PDF chat applications with minimal code.
"""

import os
from pathlib import Path
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider
from SimplerLLM.vectors import VectorProvider
from SimplerLLM.quickstart import ChatWithPDF


def example_1_simplest_usage():
    """
    Example 1: Simplest usage with all defaults.
    Uses OpenAI for both LLM and embeddings, local vector storage.
    """
    print("=" * 60)
    print("Example 1: Simplest Usage (All Defaults)")
    print("=" * 60)
    
    # Create chat instance with defaults
    chat = ChatWithPDF()
    
    # Load a PDF (replace with your PDF path)
    chat.load_pdf("cv.pdf")
    
    # Option 1: Direct API usage
    # response = chat.chat("What is this document about?")
    # print(response)
    
    # Option 2: Run terminal interface
    #chat.run(mode="terminal")
    
    # Option 3: Run UI interface
    chat.run(mode="ui")
    
    print("✓ ChatWithPDF instance created with defaults")
    print("  - LLM: OpenAI gpt-4o-mini")
    print("  - Embeddings: OpenAI text-embedding-3-small")
    print("  - Vector DB: Local")
    print()


def example_2_custom_llm_and_embeddings():
    """
    Example 2: Using custom LLM and embedding models.
    Shows how to use different providers like Anthropic, Gemini, etc.
    """
    print("=" * 60)
    print("Example 2: Custom LLM and Embeddings")
    print("=" * 60)
    
    # Create custom LLM instance (Anthropic Claude)
    llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.5
    )
    
    # Create custom embeddings instance (OpenAI)
    embeddings = EmbeddingsLLM.create(
        provider=EmbeddingsProvider.OPENAI,
        model_name="text-embedding-3-large"
    )
    
    # Create ChatWithPDF with custom instances
    chat = ChatWithPDF(
        llm_instance=llm,
        embeddings_instance=embeddings,
        chunk_strategy="semantic"  # Use semantic chunking
    )
    
    print("✓ ChatWithPDF instance created with custom models")
    print("  - LLM: Anthropic Claude 3 Haiku")
    print("  - Embeddings: OpenAI text-embedding-3-large")
    print("  - Chunking: Semantic")
    print()
    
    # Example usage
    # chat.load_pdf("document.pdf")
    # response = chat.chat("Summarize the key points")
    # print(response)


def example_3_different_chunking_strategies():
    """
    Example 3: Demonstrating different chunking strategies.
    """
    print("=" * 60)
    print("Example 3: Different Chunking Strategies")
    print("=" * 60)
    
    strategies = ["semantic", "sentence", "paragraph", "max_size"]
    
    for strategy in strategies:
        print(f"\n📄 Testing {strategy} chunking:")
        
        chat = ChatWithPDF(
            chunk_strategy=strategy,
            chunk_size=500 if strategy == "max_size" else None
        )
        
        print(f"  ✓ Created ChatWithPDF with {strategy} chunking")
        
        # You would load PDFs and test here
        # chat.load_pdf("document.pdf")
        # stats = chat.get_stats()
        # print(f"  Chunks created: {stats['total_chunks']}")


def example_4_using_qdrant_vector_store():
    """
    Example 4: Using Qdrant vector database instead of local storage.
    """
    print("=" * 60)
    print("Example 4: Using Qdrant Vector Store")
    print("=" * 60)
    
    try:
        # Create ChatWithPDF with Qdrant
        chat = ChatWithPDF(
            vector_provider=VectorProvider.QDRANT,
            vector_config={
                "url": "localhost",
                "port": 6333,
                "collection_name": "pdf_chat_collection"
            }
        )
        
        print("✓ ChatWithPDF instance created with Qdrant vector store")
        print("  - Vector DB: Qdrant")
        print("  - URL: localhost:6333")
        print("  - Collection: pdf_chat_collection")
        
    except Exception as e:
        print(f"⚠️ Could not connect to Qdrant: {e}")
        print("  Make sure Qdrant is running with:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
    
    print()


def example_5_batch_loading_and_searching():
    """
    Example 5: Loading multiple PDFs and performing searches.
    """
    print("=" * 60)
    print("Example 5: Batch Loading Multiple PDFs")
    print("=" * 60)
    
    chat = ChatWithPDF()
    
    # Example: Load multiple PDFs at once
    # pdf_files = [
    #     "research_paper_1.pdf",
    #     "research_paper_2.pdf",
    #     "research_paper_3.pdf"
    # ]
    # chat.load_pdfs(pdf_files)
    
    # Get statistics
    stats = chat.get_stats()
    print(f"📊 Database Statistics:")
    print(f"  - Documents loaded: {stats['loaded_documents']}")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Vector count: {stats['vector_count']}")
    print()
    
    # Example queries with different parameters
    # response = chat.chat("What are the main findings?", top_k=5, show_sources=True)
    # print(response)


def example_6_terminal_mode():
    """
    Example 6: Running in interactive terminal mode.
    """
    print("=" * 60)
    print("Example 6: Terminal Mode")
    print("=" * 60)
    print()
    print("Terminal mode provides an interactive command-line interface:")
    print("  - Commands: /load, /clear, /stats, /help, /exit")
    print("  - Colored output (if colorama is installed)")
    print("  - Chat history maintained during session")
    print()
    print("To run terminal mode:")
    print("  chat = ChatWithPDF()")
    print("  chat.run(mode='terminal')")
    print()


def example_7_gradio_ui_mode():
    """
    Example 7: Running with Gradio UI.
    """
    print("=" * 60)
    print("Example 7: Gradio UI Mode")
    print("=" * 60)
    print()
    print("UI mode provides a web interface with:")
    print("  - PDF upload widget")
    print("  - Chat interface with history")
    print("  - Settings panel for customization")
    print("  - Statistics display")
    print("  - Source attribution")
    print()
    print("To run UI mode:")
    print("  chat = ChatWithPDF()")
    print("  chat.run(mode='ui', port=7860, share=False)")
    print()
    print("Options:")
    print("  - port: Port to run the server (default: 7860)")
    print("  - share: Create public URL (default: False)")
    print("  - debug: Enable debug mode (default: False)")
    print()


def example_8_custom_system_prompt():
    """
    Example 8: Using a custom system prompt for specialized behavior.
    """
    print("=" * 60)
    print("Example 8: Custom System Prompt")
    print("=" * 60)
    
    # Create a specialized assistant for legal documents
    legal_prompt = """You are a legal document analyst. When answering questions:
    1. Always cite the specific section or page number when referencing the document
    2. Highlight any legal implications or important clauses
    3. Use formal legal language in your responses
    4. If something is ambiguous, clearly state so
    Only answer based on the provided document content."""
    
    chat = ChatWithPDF(system_prompt=legal_prompt)
    
    print("✓ Created ChatWithPDF with custom legal document prompt")
    print()
    
    # Another example: Technical documentation assistant
    tech_prompt = """You are a technical documentation expert. When answering:
    1. Provide code examples when relevant
    2. Explain technical concepts in simple terms
    3. Include any warnings or best practices mentioned in the docs
    4. Format your responses with clear sections and bullet points"""
    
    tech_chat = ChatWithPDF(system_prompt=tech_prompt)
    
    print("✓ Created ChatWithPDF with custom technical documentation prompt")
    print()


def example_9_method_chaining():
    """
    Example 9: Elegant one-liner usage with method chaining.
    """
    print("=" * 60)
    print("Example 9: Method Chaining (One-liner Usage)")
    print("=" * 60)
    print()
    print("You can chain methods for concise code:")
    print()
    print("# One-liner to load PDF and start UI:")
    print("ChatWithPDF().load_pdf('document.pdf').run('ui')")
    print()
    print("# One-liner with custom models:")
    print("ChatWithPDF(")
    print("    llm_instance=LLM.create(provider=LLMProvider.GEMINI, model_name='gemini-pro'),")
    print("    embeddings_instance=EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)")
    print(").load_pdfs(['doc1.pdf', 'doc2.pdf']).run('terminal')")
    print()


def example_10_complete_workflow():
    """
    Example 10: A complete workflow example.
    """
    print("=" * 60)
    print("Example 10: Complete Workflow")
    print("=" * 60)
    
    # Initialize with custom configuration
    print("1. Initializing ChatWithPDF...")
    chat = ChatWithPDF(
        chunk_strategy="semantic",
        chunk_size=600
    )
    
    # Load PDFs (uncomment with real files)
    print("2. Loading PDFs...")
    # chat.load_pdfs(["report1.pdf", "report2.pdf"])
    
    # Check statistics
    print("3. Checking statistics...")
    stats = chat.get_stats()
    print(f"   Loaded {stats['loaded_documents']} documents")
    
    # Perform some queries (uncomment with loaded PDFs)
    print("4. Performing queries...")
    # questions = [
    #     "What is the main topic of these documents?",
    #     "What are the key findings?",
    #     "Are there any recommendations?"
    # ]
    # 
    # for question in questions:
    #     print(f"\n   Q: {question}")
    #     response = chat.chat(question, top_k=3, show_sources=True)
    #     print(f"   A: {response[:200]}...")  # Show first 200 chars
    
    # Clear when done
    print("5. Cleaning up...")
    chat.clear()
    
    print("\n✓ Workflow complete!")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" ChatWithPDF Quickstart Examples")
    print("=" * 60)
    print()
    
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    
    print("🔑 API Keys detected:")
    print(f"  - OpenAI: {'✓' if has_openai else '✗'}")
    print(f"  - Anthropic: {'✓' if has_anthropic else '✗'}")
    print(f"  - Gemini: {'✓' if has_gemini else '✗'}")
    print()
    
    # Run examples
    example_1_simplest_usage()
    example_2_custom_llm_and_embeddings()
    example_3_different_chunking_strategies()
    example_4_using_qdrant_vector_store()
    example_5_batch_loading_and_searching()
    example_6_terminal_mode()
    example_7_gradio_ui_mode()
    example_8_custom_system_prompt()
    example_9_method_chaining()
    example_10_complete_workflow()
    
    print("=" * 60)
    print(" Examples Complete!")
    print("=" * 60)
    print()
    print("To actually run these examples with your PDFs:")
    print("1. Set up your API keys in environment variables")
    print("2. Replace 'document.pdf' with actual PDF paths")
    print("3. Uncomment the code sections marked with #")
    print()
    print("Quick start:")
    print("  from SimplerLLM.quickstart import ChatWithPDF")
    print("  ChatWithPDF().load_pdf('your.pdf').run('ui')")
    print()


if __name__ == "__main__":
    example_1_simplest_usage()