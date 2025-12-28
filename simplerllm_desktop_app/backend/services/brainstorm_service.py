"""
Brainstorm Service - Wrapper for SimplerLLM RecursiveBrainstorm with streaming support.
"""
import threading
import queue
import time
from typing import Dict, Any, Optional, Generator, Callable
from services.llm_service import LLMService

# Import RecursiveBrainstorm
try:
    from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm
    from SimplerLLM.language.llm_brainstorm.models import BrainstormIdea, BrainstormResult
except ImportError:
    from simplerllm.language.llm_brainstorm import RecursiveBrainstorm
    from simplerllm.language.llm_brainstorm.models import BrainstormIdea, BrainstormResult


class StreamingBrainstorm(RecursiveBrainstorm):
    """
    Extended RecursiveBrainstorm with streaming callback support.

    Overrides _generate_ideas to emit events for each idea generated,
    enabling real-time updates in the UI.
    """

    def __init__(self, *args, on_idea_callback: Optional[Callable] = None,
                 on_iteration_callback: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_idea_callback = on_idea_callback
        self._on_iteration_callback = on_iteration_callback

    def _generate_ideas(self, prompt, context, depth, parent_id):
        """Override to emit events during generation."""
        # Emit iteration start event
        if self._on_iteration_callback:
            self._on_iteration_callback({
                'type': 'iteration_start',
                'iteration': self._iteration_counter + 1,
                'depth': depth,
                'parent_id': parent_id
            })

        # Call parent implementation
        ideas = super()._generate_ideas(prompt, context, depth, parent_id)

        # Emit idea events for each generated idea
        for idea in ideas:
            if self._on_idea_callback:
                self._on_idea_callback({
                    'type': 'idea',
                    'idea': {
                        'id': idea.id,
                        'text': idea.text,
                        'reasoning': idea.reasoning,
                        'quality_score': idea.quality_score,
                        'depth': idea.depth,
                        'parent_id': idea.parent_id,
                        'iteration': idea.iteration,
                        'criteria_scores': idea.criteria_scores
                    }
                })

        # Emit iteration complete event
        if self._on_iteration_callback:
            self._on_iteration_callback({
                'type': 'iteration_complete',
                'iteration': self._iteration_counter,
                'depth': depth,
                'ideas_count': len(ideas)
            })

        return ideas


class BrainstormService:
    """Service for running brainstorming sessions with real-time streaming updates."""

    def __init__(self):
        self.llm_service = LLMService()
        self._active_sessions: Dict[str, Dict] = {}

    def run_brainstorm_streaming(
        self,
        prompt: str,
        provider: str,
        model: str,
        max_depth: int = 2,
        ideas_per_level: int = 5,
        top_n: int = 3,
        min_quality_threshold: float = 5.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run a brainstorming session and yield events as they occur.

        Args:
            prompt: The brainstorming prompt
            provider: LLM provider (e.g., 'openai')
            model: Model name (e.g., 'gpt-4o')
            max_depth: Maximum tree depth (1-5)
            ideas_per_level: Ideas per expansion (3-10)
            top_n: Top N ideas to expand in hybrid mode (1-5)
            min_quality_threshold: Minimum score to expand (1-10)

        Yields:
            Dict events with types: 'start', 'iteration_start', 'idea',
            'iteration_complete', 'complete', 'error'
        """
        # Event queue for communication between callback and generator
        event_queue = queue.Queue()

        def on_idea(event):
            event_queue.put(event)

        def on_iteration(event):
            event_queue.put(event)

        def run_brainstorm():
            try:
                # Get LLM instance
                llm = self.llm_service.get_llm(provider, model)
                if not llm:
                    event_queue.put({
                        'type': 'error',
                        'error': f'Failed to initialize LLM for {provider}/{model}'
                    })
                    return

                # Create streaming brainstorm instance
                brainstorm = StreamingBrainstorm(
                    llm=llm,
                    max_depth=max_depth,
                    ideas_per_level=ideas_per_level,
                    mode="hybrid",
                    top_n=top_n,
                    min_quality_threshold=min_quality_threshold,
                    verbose=False,
                    on_idea_callback=on_idea,
                    on_iteration_callback=on_iteration
                )

                # Run brainstorming
                result = brainstorm.brainstorm(prompt=prompt, mode="hybrid")

                # Send complete event with final result
                event_queue.put({
                    'type': 'complete',
                    'result': self._serialize_result(result)
                })

            except Exception as e:
                event_queue.put({
                    'type': 'error',
                    'error': str(e)
                })

        # Start brainstorm in background thread
        thread = threading.Thread(target=run_brainstorm)
        thread.start()

        # Yield start event
        yield {'type': 'start', 'timestamp': time.time()}

        # Yield events from queue until complete or error
        while True:
            try:
                event = event_queue.get(timeout=120)  # 2 minute timeout
                yield event

                if event['type'] in ('complete', 'error'):
                    break

            except queue.Empty:
                yield {
                    'type': 'error',
                    'error': 'Brainstorm session timed out'
                }
                break

        # Wait for thread to finish
        thread.join(timeout=5)

    def _serialize_result(self, result: BrainstormResult) -> Dict[str, Any]:
        """Serialize BrainstormResult to JSON-compatible dict."""
        return {
            'initial_prompt': result.initial_prompt,
            'mode': result.mode,
            'total_ideas': result.total_ideas,
            'total_iterations': result.total_iterations,
            'max_depth_reached': result.max_depth_reached,
            'execution_time': result.execution_time,
            'stopped_reason': result.stopped_reason,
            'tree_structure': result.tree_structure,
            'config_used': result.config_used,
            'overall_best_idea': self._serialize_idea(result.overall_best_idea) if result.overall_best_idea else None,
            'all_ideas': [self._serialize_idea(idea) for idea in result.all_ideas],
            'levels': [
                {
                    'depth': level.depth,
                    'total_ideas': level.total_ideas,
                    'average_score': level.average_score,
                    'best_idea': self._serialize_idea(level.best_idea) if level.best_idea else None
                }
                for level in result.levels
            ]
        }

    def _serialize_idea(self, idea: BrainstormIdea) -> Dict[str, Any]:
        """Serialize BrainstormIdea to JSON-compatible dict."""
        return {
            'id': idea.id,
            'text': idea.text,
            'reasoning': idea.reasoning,
            'quality_score': idea.quality_score,
            'depth': idea.depth,
            'parent_id': idea.parent_id,
            'iteration': idea.iteration,
            'criteria_scores': idea.criteria_scores
        }


# Singleton instance
brainstorm_service = BrainstormService()
