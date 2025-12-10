"""
LLM Validator - Multi-provider validation system for AI-generated content.

This module provides the LLMValidator class for validating AI-generated content
using multiple LLM providers, with configurable aggregation methods.
"""

import time
import json
import re
import concurrent.futures
import statistics
from typing import List, Optional, Dict
from datetime import datetime

from SimplerLLM.language.llm.base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import (
    AggregationMethod,
    ValidatorScore,
    ValidationResult,
)


class LLMValidator:
    """
    Validates AI-generated content using multiple LLM providers.

    Each validator independently scores the content, and results are aggregated
    using the specified method (average, weighted, median, or consensus).

    Example:
        ```python
        from SimplerLLM.language import LLM, LLMProvider
        from SimplerLLM.language.llm_validator import LLMValidator

        validators = [
            LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
            LLM.create(LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022"),
        ]

        validator = LLMValidator(validators=validators)
        result = validator.validate(
            content="Paris is the capital of France.",
            validation_prompt="Check if the facts are accurate.",
            original_question="What is the capital of France?",
        )
        print(f"Score: {result.overall_score}, Valid: {result.is_valid}")
        ```
    """

    def __init__(
        self,
        validators: List[LLM],
        parallel: bool = True,
        default_threshold: float = 0.7,
        consensus_threshold: float = 0.2,
        verbose: bool = False,
    ):
        """
        Initialize the LLM Validator.

        Args:
            validators: List of LLM instances to use as validators
            parallel: If True, execute validators in parallel; if False, sequential
            default_threshold: Default score threshold for is_valid (0.0 to 1.0)
            consensus_threshold: Max score difference to consider consensus (default 0.2)
            verbose: Enable verbose logging
        """
        if not validators or len(validators) < 1:
            raise ValueError("At least one validator must be specified")

        self.validators = validators
        self.parallel = parallel
        self.default_threshold = default_threshold
        self.consensus_threshold = consensus_threshold
        self.verbose = verbose

        if self.verbose:
            verbose_print(
                f"Initialized LLMValidator with {len(validators)} validators, "
                f"parallel={parallel}, threshold={default_threshold}",
                "info"
            )

    def validate(
        self,
        content: str,
        validation_prompt: str,
        original_question: Optional[str] = None,
        context: Optional[str] = None,
        aggregation: str = "average",
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate AI-generated content using all configured validators.

        Args:
            content: The AI-generated content to validate
            validation_prompt: Instructions for what to validate (your custom criteria)
            original_question: Optional original question that generated the content
            context: Optional additional context for validation
            aggregation: Aggregation method - "average", "weighted", "median", or "consensus"
            weights: Provider weights for weighted aggregation (e.g., {"OPENAI": 1.5, "ANTHROPIC": 1.0})
            threshold: Override default threshold for is_valid

        Returns:
            ValidationResult containing overall score, individual scores, and metadata
        """
        start_time = time.time()

        # Validate aggregation method
        try:
            agg_method = AggregationMethod(aggregation)
        except ValueError:
            raise ValueError(
                f"Invalid aggregation '{aggregation}'. "
                f"Must be one of: {[m.value for m in AggregationMethod]}"
            )

        # Use provided threshold or default
        score_threshold = threshold if threshold is not None else self.default_threshold

        if self.verbose:
            verbose_print(
                f"Starting validation with {len(self.validators)} validators, "
                f"aggregation={aggregation}",
                "info"
            )

        # Build the validation prompt
        full_prompt = self._build_validation_prompt(
            content=content,
            validation_prompt=validation_prompt,
            original_question=original_question,
            context=context,
        )

        # Execute all validators
        validator_scores = self._execute_validators(full_prompt, score_threshold)

        if self.verbose:
            verbose_print(f"Received {len(validator_scores)} validator responses", "info")

        # Filter successful validations for aggregation
        successful_scores = [v for v in validator_scores if v.error is None]

        if not successful_scores:
            raise RuntimeError("All validators failed to validate the content")

        # Aggregate scores
        overall_score, overall_confidence = self._aggregate_scores(
            successful_scores, agg_method, weights
        )

        # Calculate consensus
        consensus, consensus_details = self._calculate_consensus(successful_scores)

        # Build result
        total_time = time.time() - start_time

        result = ValidationResult(
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            is_valid=overall_score >= score_threshold,
            validators=validator_scores,
            consensus=consensus,
            consensus_details=consensus_details,
            aggregation_method=agg_method,
            content_validated=content,
            validation_prompt=validation_prompt,
            original_question=original_question,
            total_execution_time=total_time,
            timestamp=datetime.now(),
        )

        if self.verbose:
            verbose_print(
                f"Validation complete in {total_time:.2f}s. "
                f"Score: {overall_score:.2f}, Valid: {result.is_valid}",
                "info"
            )

        return result

    def validate_batch(
        self,
        items: List[Dict],
        aggregation: str = "average",
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> List[ValidationResult]:
        """
        Validate multiple items in batch.

        Args:
            items: List of dicts with keys: content, validation_prompt,
                   and optionally: original_question, context
            aggregation: Aggregation method for all items
            weights: Provider weights for weighted aggregation
            threshold: Override default threshold

        Returns:
            List of ValidationResult objects, one per item
        """
        results = []
        total_items = len(items)

        if self.verbose:
            verbose_print(f"Starting batch validation of {total_items} items", "info")

        for idx, item in enumerate(items, 1):
            if self.verbose:
                verbose_print(f"Validating item {idx}/{total_items}", "info")

            result = self.validate(
                content=item["content"],
                validation_prompt=item["validation_prompt"],
                original_question=item.get("original_question"),
                context=item.get("context"),
                aggregation=aggregation,
                weights=weights,
                threshold=threshold,
            )
            results.append(result)

        return results

    # ==================== Private Methods ====================

    def _build_validation_prompt(
        self,
        content: str,
        validation_prompt: str,
        original_question: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Build the full prompt for validators."""
        prompt_parts = [
            "You are an expert validator. Your task is to validate the following AI-generated content.",
            ""
        ]

        if original_question:
            prompt_parts.extend([
                "ORIGINAL QUESTION:",
                original_question,
                ""
            ])

        prompt_parts.extend([
            "CONTENT TO VALIDATE:",
            content,
            ""
        ])

        if context:
            prompt_parts.extend([
                "ADDITIONAL CONTEXT:",
                context,
                ""
            ])

        prompt_parts.extend([
            "VALIDATION CRITERIA:",
            validation_prompt,
            "",
            "YOUR TASK:",
            "1. Carefully analyze the content against the validation criteria",
            "2. Provide a score from 0.0 to 1.0 (0 = completely invalid, 1 = perfectly valid)",
            "3. Provide your confidence in this score from 0.0 to 1.0",
            "4. Explain your reasoning in detail",
        ])

        return "\n".join(prompt_parts)

    def _execute_validators(
        self,
        prompt: str,
        threshold: float,
    ) -> List[ValidatorScore]:
        """Execute all validators and collect their scores."""
        scores = []

        if self.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.validators)) as executor:
                futures = {
                    executor.submit(self._execute_single_validator, validator, prompt, threshold): validator
                    for validator in self.validators
                }

                for future in concurrent.futures.as_completed(futures):
                    score = future.result()
                    scores.append(score)
        else:
            for validator in self.validators:
                score = self._execute_single_validator(validator, prompt, threshold)
                scores.append(score)

        return scores

    def _execute_single_validator(
        self,
        validator: LLM,
        prompt: str,
        threshold: float,
    ) -> ValidatorScore:
        """Execute a single validator and return its score."""
        start_time = time.time()
        error = None
        score = 0.0
        confidence = 0.0
        explanation = ""

        # Build JSON-requesting prompt
        json_prompt = prompt + """

Respond with a JSON object in this exact format:
{
    "score": <number between 0.0 and 1.0>,
    "confidence": <number between 0.0 and 1.0>,
    "explanation": "<your detailed explanation>"
}

OUTPUT:"""

        try:
            if self.verbose:
                verbose_print(
                    f"Executing validator: {validator.provider.name} ({validator.model_name})",
                    "debug"
                )

            # Call LLM directly (avoiding generate_pydantic_json_model which passes unsupported params)
            response = validator.generate_response(
                prompt=json_prompt,
                system_prompt="You are an expert validator. Respond only with valid JSON.",
                max_tokens=2048,
                temperature=0.7,
                json_mode=True,
            )

            # Parse JSON from response
            if response:
                parsed = self._parse_json_response(response)
                if parsed:
                    score = float(parsed.get("score", 0.0))
                    confidence = float(parsed.get("confidence", 0.0))
                    explanation = str(parsed.get("explanation", ""))

                    # Clamp values to valid range
                    score = max(0.0, min(1.0, score))
                    confidence = max(0.0, min(1.0, confidence))
                else:
                    error = "Failed to parse JSON from response"
            else:
                error = "Empty response from validator"

        except Exception as e:
            error = str(e)
            if self.verbose:
                verbose_print(f"Error from {validator.provider.name}: {error}", "error")

        execution_time = time.time() - start_time

        return ValidatorScore(
            provider_name=validator.provider.name,
            model_name=validator.model_name,
            score=score,
            confidence=confidence,
            explanation=explanation,
            is_valid=score >= threshold,
            execution_time=execution_time,
            error=error,
        )

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        try:
            # Try direct parse first
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_patterns = [
            r'\{[^{}]*"score"[^{}]*\}',  # Simple JSON object with score
            r'\{[\s\S]*?\}',  # Any JSON object
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if "score" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue

        return None

    def _aggregate_scores(
        self,
        scores: List[ValidatorScore],
        method: AggregationMethod,
        weights: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """Aggregate scores using the specified method. Returns (score, confidence)."""
        score_values = [s.score for s in scores]
        confidence_values = [s.confidence for s in scores]

        if method == AggregationMethod.AVERAGE:
            agg_score = sum(score_values) / len(score_values)
            agg_confidence = sum(confidence_values) / len(confidence_values)

        elif method == AggregationMethod.WEIGHTED:
            if not weights:
                # Fall back to average if no weights provided
                agg_score = sum(score_values) / len(score_values)
                agg_confidence = sum(confidence_values) / len(confidence_values)
            else:
                total_weight = 0.0
                weighted_score = 0.0
                weighted_confidence = 0.0

                for s in scores:
                    weight = weights.get(s.provider_name, 1.0)
                    weighted_score += s.score * weight
                    weighted_confidence += s.confidence * weight
                    total_weight += weight

                agg_score = weighted_score / total_weight if total_weight > 0 else 0.0
                agg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        elif method == AggregationMethod.MEDIAN:
            agg_score = statistics.median(score_values)
            agg_confidence = statistics.median(confidence_values)

        elif method == AggregationMethod.CONSENSUS:
            # For consensus, use average but report consensus status separately
            agg_score = sum(score_values) / len(score_values)
            agg_confidence = sum(confidence_values) / len(confidence_values)

        else:
            agg_score = sum(score_values) / len(score_values)
            agg_confidence = sum(confidence_values) / len(confidence_values)

        return agg_score, agg_confidence

    def _calculate_consensus(self, scores: List[ValidatorScore]) -> tuple:
        """Calculate if validators reached consensus. Returns (consensus, details)."""
        if len(scores) < 2:
            return True, "Only one validator, consensus by default"

        score_values = [s.score for s in scores]
        min_score = min(score_values)
        max_score = max(score_values)
        score_range = max_score - min_score

        # Check if all scores are within the consensus threshold
        if score_range <= self.consensus_threshold:
            return True, f"All validators scored within {self.consensus_threshold:.2f} of each other (range: {score_range:.2f})"

        # Check pass/fail consensus
        pass_count = sum(1 for s in scores if s.is_valid)
        fail_count = len(scores) - pass_count

        if pass_count == len(scores):
            return True, f"All {len(scores)} validators marked content as valid"
        elif fail_count == len(scores):
            return True, f"All {len(scores)} validators marked content as invalid"
        else:
            return False, f"Split decision: {pass_count} valid, {fail_count} invalid (score range: {score_range:.2f})"
