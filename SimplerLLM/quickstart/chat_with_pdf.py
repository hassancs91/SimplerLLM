"""
ChatWithPDF - A simple interface for chatting with PDF documents using LLMs and vector databases.

This module provides an easy-to-use class that combines PDF loading, text chunking, 
vector storage, and LLM querying into a simple interface with both terminal and UI modes.
"""

import os
import sys
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider
from SimplerLLM.vectors import VectorDB, VectorProvider
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.tools.text_chunker import (
    chunk_by_max_chunk_size,
    chunk_by_sentences,
    chunk_by_paragraphs,
    chunk_by_semantics
)


class ChatWithPDF:
    """
    A simple interface for chatting with PDF documents.
    
    Supports multiple chunking strategies, vector databases, and both terminal and UI modes.
    """
    
    def __init__(
        self,
        llm_instance=None,
        embeddings_instance=None,
        vector_provider=VectorProvider.LOCAL,
        vector_config=None,
        chunk_size=500,
        chunk_strategy="max_size",
        chunk_overlap=50,
        system_prompt=None
    ):
        """
        Initialize ChatWithPDF with customizable components.
        
        Args:
            llm_instance: An LLM instance for generating responses (optional)
            embeddings_instance: An EmbeddingsLLM instance for generating embeddings (optional)
            vector_provider: VectorProvider enum for vector database selection
            vector_config: Dictionary with vector database configuration
            chunk_size: Size of text chunks for splitting documents
            chunk_strategy: Strategy for chunking ("max_size", "semantic", "sentence", "paragraph")
            chunk_overlap: Overlap between chunks (only for max_size strategy)
            system_prompt: Custom system prompt for the LLM
        """
        # Initialize LLM if not provided
        if llm_instance is None:
            self.llm = LLM.create(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o-mini",
                temperature=0.7
            )
        else:
            self.llm = llm_instance
            
        # Initialize embeddings if not provided
        if embeddings_instance is None:
            self.embeddings = EmbeddingsLLM.create(
                provider=EmbeddingsProvider.OPENAI,
                model_name="text-embedding-3-small"
            )
        else:
            self.embeddings = embeddings_instance
            
        # Initialize vector database
        if vector_config is None:
            vector_config = {"db_folder": "./pdf_vector_db"} if vector_provider == VectorProvider.LOCAL else {}
            
        self.vector_db = VectorDB.create(
            provider=vector_provider,
            **vector_config
        )
        
        # Store configuration
        self.chunk_size = chunk_size
        self.chunk_strategy = chunk_strategy
        self.chunk_overlap = chunk_overlap
        
        # Initialize document tracking
        self.loaded_documents = []
        self.total_chunks = 0
        
        # Set system prompt
        if system_prompt is None:
            self.system_prompt = """You are a helpful assistant that answers questions based on the provided PDF documents. 
            Use the retrieved context to answer questions accurately. If the answer cannot be found in the context, 
            say so clearly. Always cite which part of the document you're referencing when possible."""
        else:
            self.system_prompt = system_prompt
            
        # Chat history for context
        self.chat_history = []
        
    def load_pdf(self, pdf_path: str) -> "ChatWithPDF":
        """
        Load a single PDF file into the vector database.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Self for method chaining
        """
        return self.load_pdfs([pdf_path])
        
    def load_pdfs(self, pdf_paths: List[str]) -> "ChatWithPDF":
        """
        Load multiple PDF files into the vector database.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Self for method chaining
        """
        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                print(f"Warning: File {pdf_path} does not exist. Skipping...")
                continue
                
            if pdf_path.suffix.lower() != '.pdf':
                print(f"Warning: File {pdf_path} is not a PDF. Skipping...")
                continue
                
            print(f"Loading {pdf_path.name}...")
            
            try:
                # Load PDF content
                document = load_content(str(pdf_path))
                
                # Chunk the document
                chunks = self._chunk_text(document.content)
                
                # Add chunks to vector database
                for i, chunk_info in enumerate(chunks.chunk_list):
                    # Generate embedding
                    embedding = self.embeddings.generate_embeddings(chunk_info.text)
                    
                    # Prepare metadata
                    metadata = {
                        "source": str(pdf_path),
                        "chunk_index": i,
                        "total_chunks": chunks.num_chunks,
                        "chunk_size": chunk_info.num_characters,
                        "chunk_words": chunk_info.num_words
                    }
                    
                    # Add to vector database
                    self.vector_db.add_text_with_embedding(
                        text=chunk_info.text,
                        embedding=embedding,
                        metadata=metadata
                    )
                
                self.loaded_documents.append(str(pdf_path))
                self.total_chunks += chunks.num_chunks
                print(f"✓ Loaded {pdf_path.name} ({chunks.num_chunks} chunks)")
                
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
                
        print(f"\nTotal documents loaded: {len(self.loaded_documents)}")
        print(f"Total chunks in database: {self.total_chunks}")
        return self
        
    def _chunk_text(self, text: str):
        """
        Chunk text based on the selected strategy.
        
        Args:
            text: Text to chunk
            
        Returns:
            TextChunks object with chunked text
        """
        if self.chunk_strategy == "semantic":
            return chunk_by_semantics(text, self.embeddings)
        elif self.chunk_strategy == "sentence":
            return chunk_by_sentences(text)
        elif self.chunk_strategy == "paragraph":
            return chunk_by_paragraphs(text)
        else:  # max_size
            return chunk_by_max_chunk_size(
                text, 
                self.chunk_size,
                preserve_sentence_structure=True
            )
            
    def chat(self, query: str, top_k: int = 3, show_sources: bool = False) -> str:
        """
        Query the loaded PDFs with a question.
        
        Args:
            query: The question to ask
            top_k: Number of relevant chunks to retrieve
            show_sources: Whether to include source information in response
            
        Returns:
            The LLM's response based on retrieved context
        """
        if self.total_chunks == 0:
            return "No documents loaded. Please load a PDF first using load_pdf() or load_pdfs()."
            
        # Search for relevant chunks
        results = self.vector_db.search_by_text(
            query_text=query,
            embeddings_llm_instance=self.embeddings,
            top_n=top_k
        )
        
        if not results:
            return "No relevant information found in the loaded documents."
            
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for vector_id, metadata, similarity in results:
            text = metadata.get('text', '')
            source = metadata.get('source', 'Unknown')
            chunk_index = metadata.get('chunk_index', 0)
            
            context_parts.append(text)
            sources.append(f"{Path(source).name} (chunk {chunk_index + 1})")
            
        context = "\n\n---\n\n".join(context_parts)
        
        # Build the prompt
        prompt = f"""{self.system_prompt}

Context from documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above."""

        # Generate response
        response = self.llm.generate_response(prompt=prompt)
        
        # Add to chat history
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Add sources if requested
        if show_sources:
            unique_sources = list(set(sources))
            response += f"\n\n📚 Sources: {', '.join(unique_sources)}"
            
        return response
        
    def clear(self):
        """Clear the vector database and reset document tracking."""
        self.vector_db.clear_database()
        self.loaded_documents = []
        self.total_chunks = 0
        self.chat_history = []
        print("✓ Database cleared successfully")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded documents and database.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "loaded_documents": len(self.loaded_documents),
            "document_list": self.loaded_documents,
            "total_chunks": self.total_chunks,
            "vector_count": self.vector_db.get_vector_count(),
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size if self.chunk_strategy == "max_size" else "N/A",
            "chat_history_length": len(self.chat_history)
        }
        
        # Add vector database specific stats if available
        try:
            db_stats = self.vector_db.get_stats()
            stats.update(db_stats)
        except:
            pass
            
        return stats
        
    def run(self, mode: str = "terminal", **kwargs):
        """
        Run the chat interface in terminal or UI mode.
        
        Args:
            mode: Either "terminal" or "ui"
            **kwargs: Additional arguments for the specific mode
                For UI mode: port, share, debug
        """
        if mode.lower() == "terminal":
            self._run_terminal()
        elif mode.lower() == "ui":
            self._run_ui(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'terminal' or 'ui'")
            
    def _run_terminal(self):
        """Run the interactive terminal interface."""
        try:
            from colorama import init, Fore, Style
            init()
            
            # Color definitions
            CYAN = Fore.CYAN
            GREEN = Fore.GREEN
            YELLOW = Fore.YELLOW
            RED = Fore.RED
            RESET = Style.RESET_ALL
            BRIGHT = Style.BRIGHT
            
        except ImportError:
            # Fallback if colorama is not installed
            CYAN = GREEN = YELLOW = RED = RESET = BRIGHT = ""
            
        print(f"\n{CYAN}{BRIGHT}📚 Chat with PDF - Terminal Mode{RESET}")
        print(f"{CYAN}{'='*50}{RESET}")
        
        if self.total_chunks == 0:
            print(f"{YELLOW}No documents loaded yet.{RESET}")
            print(f"Use {BRIGHT}/load <path/to/file.pdf>{RESET} to load a PDF")
        else:
            print(f"{GREEN}✓ {len(self.loaded_documents)} document(s) loaded with {self.total_chunks} chunks{RESET}")
            
        print(f"\n{CYAN}Available commands:{RESET}")
        print(f"  {BRIGHT}/load <pdf_path>{RESET} - Load a PDF file")
        print(f"  {BRIGHT}/clear{RESET} - Clear all loaded documents")
        print(f"  {BRIGHT}/stats{RESET} - Show database statistics")
        print(f"  {BRIGHT}/help{RESET} - Show this help message")
        print(f"  {BRIGHT}/exit{RESET} or {BRIGHT}/quit{RESET} - Exit the chat")
        print(f"\n{CYAN}Type your question or command:{RESET}\n")
        
        while True:
            try:
                # Get user input
                user_input = input(f"{GREEN}You: {RESET}").strip()
                
                if not user_input:
                    continue
                    
                # Check for commands
                if user_input.startswith('/'):
                    command_parts = user_input.split(maxsplit=1)
                    command = command_parts[0].lower()
                    
                    if command in ['/exit', '/quit']:
                        print(f"\n{CYAN}Goodbye! 👋{RESET}")
                        break
                        
                    elif command == '/help':
                        print(f"\n{CYAN}Available commands:{RESET}")
                        print(f"  {BRIGHT}/load <pdf_path>{RESET} - Load a PDF file")
                        print(f"  {BRIGHT}/clear{RESET} - Clear all loaded documents")
                        print(f"  {BRIGHT}/stats{RESET} - Show database statistics")
                        print(f"  {BRIGHT}/help{RESET} - Show this help message")
                        print(f"  {BRIGHT}/exit{RESET} or {BRIGHT}/quit{RESET} - Exit the chat\n")
                        
                    elif command == '/clear':
                        self.clear()
                        
                    elif command == '/stats':
                        stats = self.get_stats()
                        print(f"\n{CYAN}📊 Database Statistics:{RESET}")
                        for key, value in stats.items():
                            if key != "document_list":
                                print(f"  {key}: {value}")
                        if stats["document_list"]:
                            print(f"  Documents: {', '.join([Path(d).name for d in stats['document_list']])}")
                        print()
                        
                    elif command == '/load':
                        if len(command_parts) < 2:
                            print(f"{RED}Please provide a PDF path. Usage: /load <path/to/file.pdf>{RESET}")
                        else:
                            pdf_path = command_parts[1]
                            self.load_pdf(pdf_path)
                            
                    else:
                        print(f"{RED}Unknown command: {command}. Type /help for available commands.{RESET}")
                        
                else:
                    # Regular chat query
                    if self.total_chunks == 0:
                        print(f"{YELLOW}No documents loaded. Use /load <pdf_path> to load a PDF first.{RESET}")
                    else:
                        print(f"{CYAN}Thinking...{RESET}")
                        response = self.chat(user_input, show_sources=True)
                        print(f"\n{BRIGHT}Assistant:{RESET} {response}\n")
                        
            except KeyboardInterrupt:
                print(f"\n\n{CYAN}Goodbye! 👋{RESET}")
                break
            except Exception as e:
                print(f"{RED}Error: {e}{RESET}")
                
    def _run_ui(self, port: int = 7860, share: bool = False, debug: bool = False):
        """
        Run the Gradio UI interface.
        
        Args:
            port: Port to run the UI on
            share: Whether to create a public link
            debug: Whether to run in debug mode
        """
        try:
            import gradio as gr
        except ImportError:
            print("Gradio is not installed. Please install it with: pip install gradio")
            print("Falling back to terminal mode...")
            self._run_terminal()
            return
            
        # Create the Gradio interface
        with gr.Blocks(title="Chat with PDF", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 📚 Chat with PDF")
            gr.Markdown("Upload PDF documents and ask questions about their content.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Settings panel
                    gr.Markdown("### Settings")
                    
                    # File upload
                    file_upload = gr.File(
                        label="Upload PDF(s)",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    
                    # Upload button
                    upload_btn = gr.Button("📥 Load PDFs", variant="primary")
                    
                    # Number of chunks slider
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of context chunks",
                        info="How many relevant chunks to retrieve"
                    )
                    
                    # Show sources checkbox
                    show_sources = gr.Checkbox(
                        label="Show sources",
                        value=True,
                        info="Display which documents were used"
                    )
                    
                    # Statistics display
                    stats_display = gr.Textbox(
                        label="📊 Statistics",
                        lines=8,
                        max_lines=10,
                        interactive=False
                    )
                    
                    # Clear button
                    clear_btn = gr.Button("🗑️ Clear Database", variant="stop")
                    
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=500,
                        elem_id="chatbot",
                        type="messages"
                    )
                    
                    msg = gr.Textbox(
                        label="Ask a question",
                        placeholder="What is this document about?",
                        lines=2
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("📤 Send", variant="primary")
                        clear_chat_btn = gr.Button("🔄 Clear Chat")
                        
            # Status message
            status_msg = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
            
            # Event handlers
            def load_pdfs(files):
                """Handle PDF upload."""
                if not files:
                    return "No files uploaded", update_stats()
                    
                pdf_paths = [f.name for f in files]
                self.load_pdfs(pdf_paths)
                
                return f"✓ Loaded {len(pdf_paths)} PDF(s)", update_stats()
                
            def update_stats():
                """Update statistics display."""
                stats = self.get_stats()
                stats_text = ""
                for key, value in stats.items():
                    if key == "document_list":
                        if value:
                            docs = [Path(d).name for d in value]
                            stats_text += f"Documents:\n  " + "\n  ".join(docs) + "\n"
                    else:
                        stats_text += f"{key}: {value}\n"
                return stats_text
                
            def chat_response(message, history, top_k, sources):
                """Generate chat response."""
                if not message:
                    return history
                    
                response = self.chat(message, top_k=top_k, show_sources=sources)
                # For messages format, we need to append dictionaries with role and content
                if not history:
                    history = []
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return history
                
            def clear_database():
                """Clear the database."""
                self.clear()
                return [], "✓ Database cleared", update_stats()
                
            # Connect event handlers
            upload_btn.click(
                fn=load_pdfs,
                inputs=[file_upload],
                outputs=[status_msg, stats_display]
            ).then(
                lambda: gr.update(visible=True),
                outputs=[status_msg]
            )
            
            submit_btn.click(
                fn=chat_response,
                inputs=[msg, chatbot, top_k_slider, show_sources],
                outputs=[chatbot]
            ).then(
                lambda: "",
                outputs=[msg]
            )
            
            msg.submit(
                fn=chat_response,
                inputs=[msg, chatbot, top_k_slider, show_sources],
                outputs=[chatbot]
            ).then(
                lambda: "",
                outputs=[msg]
            )
            
            clear_btn.click(
                fn=clear_database,
                outputs=[chatbot, status_msg, stats_display]
            ).then(
                lambda: gr.update(visible=True),
                outputs=[status_msg]
            )
            
            clear_chat_btn.click(
                lambda: [],
                outputs=[chatbot]
            )
            
            # Load initial stats
            demo.load(
                fn=update_stats,
                outputs=[stats_display]
            )
            
        # Launch the app
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            debug=debug
        )