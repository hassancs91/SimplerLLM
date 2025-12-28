"""
Judge Service - Wrapper for SimplerLLM LLMJudge with streaming support.
"""
import threading
import queue
import time
from typing import Dict, Any, Optional, Generator, List
from datetime import datetime
from services.llm_service import LLMService

# Import LLMJudge
try:
    from SimplerLLM.language.llm_judge import LLMJudge
    from SimplerLLM.language.llm_judge.models import (
        JudgeMode, JudgeResult, ProviderResponse, ProviderEvaluation
    )
except ImportError:
    from simplerllm.language.llm_judge import LLMJudge
    from simplerllm.language.llm_judge.models import (
        JudgeMode, JudgeResult, ProviderResponse, ProviderEvaluation
    )


class JudgeService:
    """Service for running LLM Judge evaluations with real-time streaming updates."""

    def __init__(self):
        self.llm_service = LLMService()

    def run_judge_streaming(
        self,
        prompt: str,
        contestants: List[Dict[str, str]],
        judge_config: Dict[str, str],
        mode: str = "synthesize",
        criteria: List[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run a judge evaluation and yield events as they occur.

        Args:
            prompt: The prompt to evaluate
            contestants: List of dicts with 'provider' and 'model' keys
            judge_config: Dict with 'provider' and 'model' for the judge LLM
            mode: Evaluation mode - 'select_best', 'synthesize', or 'compare'
            criteria: Evaluation criteria list

        Yields:
            Dict events with types: 'start', 'contestant_start', 'contestant_complete',
            'judging_start', 'complete', 'error'
        """
        # Event queue for communication between thread and generator
        event_queue = queue.Queue()

        def run_evaluation():
            try:
                # Create contestant LLM instances
                contestant_llms = []
                for idx, contestant in enumerate(contestants):
                    event_queue.put({
                        'type': 'contestant_start',
                        'index': idx,
                        'provider': contestant['provider'],
                        'model': contestant['model']
                    })

                    llm = self.llm_service.get_llm(
                        contestant['provider'],
                        contestant['model']
                    )

                    if not llm:
                        event_queue.put({
                            'type': 'error',
                            'error': f"Failed to initialize {contestant['provider']}/{contestant['model']}"
                        })
                        return

                    contestant_llms.append(llm)

                    event_queue.put({
                        'type': 'contestant_ready',
                        'index': idx,
                        'provider': contestant['provider']
                    })

                # Create judge LLM
                judge_llm = self.llm_service.get_llm(
                    judge_config['provider'],
                    judge_config['model']
                )

                if not judge_llm:
                    event_queue.put({
                        'type': 'error',
                        'error': f"Failed to initialize judge LLM: {judge_config['provider']}/{judge_config['model']}"
                    })
                    return

                # Create LLMJudge instance
                judge = LLMJudge(
                    providers=contestant_llms,
                    judge_llm=judge_llm,
                    parallel=True,
                    default_criteria=criteria or ["accuracy", "clarity", "completeness"],
                    verbose=False
                )

                # Emit event that contestants are running
                event_queue.put({
                    'type': 'contestants_running',
                    'message': 'All contestants are generating responses...'
                })

                # Run evaluation
                result = judge.generate(
                    prompt=prompt,
                    mode=mode,
                    criteria=criteria
                )

                # Emit contestant complete events based on result
                for idx, response in enumerate(result.all_responses):
                    event_queue.put({
                        'type': 'contestant_complete',
                        'index': idx,
                        'provider': response.provider_name,
                        'model': response.model_name,
                        'execution_time': response.execution_time,
                        'has_error': response.error is not None
                    })

                # Emit judging event
                event_queue.put({
                    'type': 'judging_complete',
                    'message': 'Judge has evaluated all responses'
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

        # Start evaluation in background thread
        thread = threading.Thread(target=run_evaluation)
        thread.start()

        # Yield start event
        yield {
            'type': 'start',
            'timestamp': time.time(),
            'total_contestants': len(contestants)
        }

        # Yield events from queue until complete or error
        while True:
            try:
                event = event_queue.get(timeout=300)  # 5 minute timeout
                yield event

                if event['type'] in ('complete', 'error'):
                    break

            except queue.Empty:
                yield {
                    'type': 'error',
                    'error': 'Judge evaluation timed out'
                }
                break

        # Wait for thread to finish
        thread.join(timeout=5)

    def _serialize_result(self, result: JudgeResult) -> Dict[str, Any]:
        """Serialize JudgeResult to JSON-compatible dict."""
        return {
            'final_answer': result.final_answer,
            'mode': result.mode.value if hasattr(result.mode, 'value') else str(result.mode),
            'criteria_used': result.criteria_used,
            'total_execution_time': result.total_execution_time,
            'judge_reasoning': result.judge_reasoning,
            'confidence_scores': result.confidence_scores,
            'all_responses': [self._serialize_response(r) for r in result.all_responses],
            'evaluations': [self._serialize_evaluation(e) for e in result.evaluations],
            'timestamp': result.timestamp.isoformat() if result.timestamp else datetime.now().isoformat()
        }

    def _serialize_response(self, response: ProviderResponse) -> Dict[str, Any]:
        """Serialize ProviderResponse to JSON-compatible dict."""
        return {
            'provider_name': response.provider_name,
            'model_name': response.model_name,
            'response_text': response.response_text,
            'execution_time': response.execution_time,
            'error': response.error,
            'timestamp': response.timestamp.isoformat() if response.timestamp else None
        }

    def _serialize_evaluation(self, evaluation: ProviderEvaluation) -> Dict[str, Any]:
        """Serialize ProviderEvaluation to JSON-compatible dict."""
        return {
            'provider_name': evaluation.provider_name,
            'overall_score': evaluation.overall_score,
            'rank': evaluation.rank,
            'criterion_scores': evaluation.criterion_scores,
            'reasoning': evaluation.reasoning,
            'strengths': evaluation.strengths or [],
            'weaknesses': evaluation.weaknesses or []
        }


# Singleton instance
judge_service = JudgeService()
