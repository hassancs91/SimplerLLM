"""
Feedback Service - Wrapper for SimplerLLM LLMFeedbackLoop with streaming support.
"""
import threading
import queue
import time
from typing import Dict, Any, Optional, Generator, List
from datetime import datetime
from services.llm_service import LLMService

# Import LLMFeedbackLoop
try:
    from SimplerLLM.language.llm_feedback import LLMFeedbackLoop
    from SimplerLLM.language.llm_feedback.models import (
        FeedbackResult, IterationResult, Critique
    )
except ImportError:
    from simplerllm.language.llm_feedback import LLMFeedbackLoop
    from simplerllm.language.llm_feedback.models import (
        FeedbackResult, IterationResult, Critique
    )


class FeedbackService:
    """Service for running LLM Feedback loops with real-time streaming updates."""

    def __init__(self):
        self.llm_service = LLMService()

    def run_feedback_streaming(
        self,
        prompt: str,
        architecture: str,
        generator_config: Optional[Dict[str, str]] = None,
        critic_config: Optional[Dict[str, str]] = None,
        providers_config: Optional[List[Dict[str, str]]] = None,
        max_iterations: int = 3,
        criteria: List[str] = None,
        initial_answer: Optional[str] = None,
        convergence_threshold: float = 0.1,
        quality_threshold: Optional[float] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run a feedback loop and yield events as they occur.

        Args:
            prompt: The prompt to improve an answer for
            architecture: 'single', 'dual', or 'multi'
            generator_config: Dict with 'provider' and 'model' for generator
            critic_config: Dict with 'provider' and 'model' for critic
            providers_config: List of provider configs for multi mode
            max_iterations: Maximum improvement iterations
            criteria: Evaluation criteria list
            initial_answer: Optional starting answer
            convergence_threshold: Stop if improvement < threshold
            quality_threshold: Stop if score >= threshold

        Yields:
            Dict events with types: 'start', 'iteration_start', 'iteration_complete',
            'complete', 'error'
        """
        # Event queue for communication between thread and generator
        event_queue = queue.Queue()

        def run_feedback():
            try:
                # Create LLM instances based on architecture
                feedback_loop = None

                if architecture == 'single':
                    # Single provider self-critique
                    if not generator_config:
                        event_queue.put({
                            'type': 'error',
                            'error': 'Generator config required for single architecture'
                        })
                        return

                    llm = self.llm_service.get_llm(
                        generator_config['provider'],
                        generator_config['model']
                    )
                    if not llm:
                        event_queue.put({
                            'type': 'error',
                            'error': f"Failed to initialize {generator_config['provider']}/{generator_config['model']}"
                        })
                        return

                    event_queue.put({
                        'type': 'llm_ready',
                        'message': f"Using {generator_config['provider']} for self-critique"
                    })

                    feedback_loop = LLMFeedbackLoop(
                        llm=llm,
                        max_iterations=max_iterations,
                        convergence_threshold=convergence_threshold,
                        quality_threshold=quality_threshold,
                        default_criteria=criteria or ["accuracy", "clarity", "completeness"],
                        verbose=False
                    )

                elif architecture == 'dual':
                    # Dual provider (generator + critic)
                    if not generator_config or not critic_config:
                        event_queue.put({
                            'type': 'error',
                            'error': 'Generator and critic configs required for dual architecture'
                        })
                        return

                    generator_llm = self.llm_service.get_llm(
                        generator_config['provider'],
                        generator_config['model']
                    )
                    if not generator_llm:
                        event_queue.put({
                            'type': 'error',
                            'error': f"Failed to initialize generator: {generator_config['provider']}/{generator_config['model']}"
                        })
                        return

                    critic_llm = self.llm_service.get_llm(
                        critic_config['provider'],
                        critic_config['model']
                    )
                    if not critic_llm:
                        event_queue.put({
                            'type': 'error',
                            'error': f"Failed to initialize critic: {critic_config['provider']}/{critic_config['model']}"
                        })
                        return

                    event_queue.put({
                        'type': 'llm_ready',
                        'message': f"Generator: {generator_config['provider']}, Critic: {critic_config['provider']}"
                    })

                    feedback_loop = LLMFeedbackLoop(
                        generator_llm=generator_llm,
                        critic_llm=critic_llm,
                        max_iterations=max_iterations,
                        convergence_threshold=convergence_threshold,
                        quality_threshold=quality_threshold,
                        default_criteria=criteria or ["accuracy", "clarity", "completeness"],
                        verbose=False
                    )

                elif architecture == 'multi':
                    # Multi-provider rotation
                    if not providers_config or len(providers_config) < 2:
                        event_queue.put({
                            'type': 'error',
                            'error': 'At least 2 providers required for multi architecture'
                        })
                        return

                    provider_llms = []
                    for config in providers_config:
                        llm = self.llm_service.get_llm(
                            config['provider'],
                            config['model']
                        )
                        if not llm:
                            event_queue.put({
                                'type': 'error',
                                'error': f"Failed to initialize {config['provider']}/{config['model']}"
                            })
                            return
                        provider_llms.append(llm)

                    provider_names = [c['provider'] for c in providers_config]
                    event_queue.put({
                        'type': 'llm_ready',
                        'message': f"Rotating providers: {', '.join(provider_names)}"
                    })

                    feedback_loop = LLMFeedbackLoop(
                        providers=provider_llms,
                        max_iterations=max_iterations,
                        convergence_threshold=convergence_threshold,
                        quality_threshold=quality_threshold,
                        default_criteria=criteria or ["accuracy", "clarity", "completeness"],
                        verbose=False
                    )

                else:
                    event_queue.put({
                        'type': 'error',
                        'error': f"Invalid architecture: {architecture}"
                    })
                    return

                # Emit running event
                event_queue.put({
                    'type': 'running',
                    'message': 'Starting improvement loop...'
                })

                # Run the feedback loop
                result = feedback_loop.improve(
                    prompt=prompt,
                    initial_answer=initial_answer,
                    focus_on=criteria
                )

                # Emit iteration events from result
                for iteration in result.all_iterations:
                    event_queue.put({
                        'type': 'iteration_complete',
                        'iteration': self._serialize_iteration(iteration)
                    })

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

        # Start feedback in background thread
        thread = threading.Thread(target=run_feedback)
        thread.start()

        # Yield start event
        yield {
            'type': 'start',
            'timestamp': time.time(),
            'max_iterations': max_iterations,
            'architecture': architecture
        }

        # Yield events from queue until complete or error
        while True:
            try:
                event = event_queue.get(timeout=600)  # 10 minute timeout
                yield event

                if event['type'] in ('complete', 'error'):
                    break

            except queue.Empty:
                yield {
                    'type': 'error',
                    'error': 'Feedback loop timed out'
                }
                break

        # Wait for thread to finish
        thread.join(timeout=5)

    def _serialize_result(self, result: FeedbackResult) -> Dict[str, Any]:
        """Serialize FeedbackResult to JSON-compatible dict."""
        return {
            'final_answer': result.final_answer,
            'all_iterations': [self._serialize_iteration(i) for i in result.all_iterations],
            'initial_score': result.initial_score,
            'final_score': result.final_score,
            'improvement_trajectory': result.improvement_trajectory,
            'total_iterations': result.total_iterations,
            'stopped_reason': result.stopped_reason,
            'convergence_detected': result.convergence_detected,
            'total_execution_time': result.total_execution_time,
            'architecture_used': result.architecture_used,
            'timestamp': result.timestamp.isoformat() if result.timestamp else datetime.now().isoformat()
        }

    def _serialize_iteration(self, iteration: IterationResult) -> Dict[str, Any]:
        """Serialize IterationResult to JSON-compatible dict."""
        return {
            'iteration_number': iteration.iteration_number,
            'answer': iteration.answer,
            'critique': self._serialize_critique(iteration.critique),
            'provider_used': iteration.provider_used,
            'model_used': iteration.model_used,
            'temperature_used': iteration.temperature_used,
            'execution_time': iteration.execution_time,
            'improvement_from_previous': iteration.improvement_from_previous,
            'timestamp': iteration.timestamp.isoformat() if iteration.timestamp else None
        }

    def _serialize_critique(self, critique: Critique) -> Dict[str, Any]:
        """Serialize Critique to JSON-compatible dict."""
        return {
            'strengths': critique.strengths or [],
            'weaknesses': critique.weaknesses or [],
            'improvement_suggestions': critique.improvement_suggestions or [],
            'quality_score': critique.quality_score,
            'specific_issues': critique.specific_issues or {},
            'reasoning': critique.reasoning
        }


# Singleton instance
feedback_service = FeedbackService()
