"""
Self-Consistency - Improve LLM reliability through majority voting.

This module provides the SelfConsistency class for generating multiple
responses to the same prompt and returning the most consistent answer.
"""

import time
import re
import json
import concurrent.futures
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from collections import defaultdict

from SimplerLLM.language.llm.base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import (
    AnswerType,
    SampleResponse,
    AnswerGroup,
    ConsistencyResult,
    ExtractedAnswer,
    SemanticGrouping,
)


class SelfConsistency:
    """
    Generates multiple responses to the same prompt and returns the most consistent answer.

    Self-consistency improves reliability by:
    1. Running the same prompt multiple times with temperature > 0
    2. Extracting the core answer from each response
    3. Grouping similar answers together
    4. Returning the answer that appears most frequently (majority voting)

    This is particularly effective for math, logic, reasoning, and factual questions.

    Example:
        ```python
        from SimplerLLM.language import LLM, LLMProvider
        from SimplerLLM.language.llm_consistency import SelfConsistency

        llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")

        consistency = SelfConsistency(
            llm=llm,
            num_samples=5,
            temperature=0.7,
        )

        result = consistency.generate(
            prompt="What is 17 x 24?",
            system_prompt="Solve step by step. End with 'Answer: X'"
        )

        print(f"Answer: {result.final_answer}")
        print(f"Confidence: {result.confidence:.0%}")
        ```
    """

    def __init__(
        self,
        llm: LLM,
        num_samples: int = 5,
        temperature: float = 0.7,
        parallel: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the Self-Consistency generator.

        Args:
            llm: LLM instance to use for generation
            num_samples: Number of times to run the prompt (default: 5)
            temperature: Temperature for generation diversity (default: 0.7)
            parallel: If True, generate samples in parallel (default: True)
            verbose: Enable verbose logging (default: False)
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 for self-consistency")

        if temperature <= 0:
            raise ValueError("temperature must be > 0 for self-consistency to work (need diversity)")

        self.llm = llm
        self.num_samples = num_samples
        self.temperature = temperature
        self.parallel = parallel
        self.verbose = verbose

        if self.verbose:
            verbose_print(
                f"Initialized SelfConsistency with {num_samples} samples, "
                f"temperature={temperature}, parallel={parallel}",
                "info"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        answer_type: Optional[AnswerType] = None,
        max_tokens: int = 1024,
    ) -> ConsistencyResult:
        """
        Generate multiple responses and return the most consistent answer.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for context
            answer_type: How to compare answers - "exact" or "semantic".
                        If None, auto-detects based on the answers.
            max_tokens: Maximum tokens per response (default: 1024)

        Returns:
            ConsistencyResult with the consensus answer and metadata
        """
        start_time = time.time()

        if self.verbose:
            verbose_print(f"Starting self-consistency with {self.num_samples} samples", "info")

        # Step 1: Generate all samples
        samples = self._generate_all_samples(prompt, system_prompt, max_tokens)

        # Filter successful samples
        successful_samples = [s for s in samples if s.error is None]

        if len(successful_samples) < 2:
            raise RuntimeError(
                f"Need at least 2 successful samples for self-consistency, "
                f"got {len(successful_samples)} of {len(samples)}"
            )

        if self.verbose:
            verbose_print(f"Got {len(successful_samples)} successful samples", "info")

        # Step 2: Extract answers from each response
        extracted_answers = self._extract_answers(successful_samples)

        # Update samples with extracted answers
        for i, sample in enumerate(successful_samples):
            sample.extracted_answer = extracted_answers[i]

        # Step 3: Detect answer type if not specified
        if answer_type is None:
            answer_type = self._detect_answer_type(extracted_answers)
            if self.verbose:
                verbose_print(f"Auto-detected answer type: {answer_type.value}", "info")

        # Step 4: Group answers
        if answer_type == AnswerType.EXACT:
            groups = self._group_answers_exact(successful_samples)
        else:
            groups = self._group_answers_semantic(successful_samples)

        # Sort groups by count descending
        groups.sort(key=lambda g: g.count, reverse=True)

        # Step 5: Find majority and check for ties
        final_answer, is_tie, tied_answers = self._find_majority(groups)

        # Calculate confidence
        winning_count = groups[0].count if groups else 0
        confidence = winning_count / len(successful_samples) if successful_samples else 0.0

        total_time = time.time() - start_time

        result = ConsistencyResult(
            final_answer=final_answer,
            confidence=confidence,
            num_samples=len(samples),
            num_agreeing=winning_count,
            is_tie=is_tie,
            tied_answers=tied_answers if is_tie else None,
            all_samples=samples,
            answer_groups=groups,
            answer_type=answer_type,
            prompt=prompt,
            execution_time=total_time,
            timestamp=datetime.now(),
        )

        if self.verbose:
            if is_tie:
                verbose_print(
                    f"Completed in {total_time:.2f}s. TIE between {len(tied_answers)} answers",
                    "warning"
                )
            else:
                verbose_print(
                    f"Completed in {total_time:.2f}s. "
                    f"Answer: {final_answer}, Confidence: {confidence:.0%}",
                    "info"
                )

        return result

    # ==================== Private Methods ====================

    def _generate_all_samples(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
    ) -> List[SampleResponse]:
        """Generate all samples, either in parallel or sequentially."""
        samples = []

        if self.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_samples) as executor:
                futures = {
                    executor.submit(
                        self._generate_single_sample,
                        idx, prompt, system_prompt, max_tokens
                    ): idx
                    for idx in range(self.num_samples)
                }

                for future in concurrent.futures.as_completed(futures):
                    sample = future.result()
                    samples.append(sample)
        else:
            for idx in range(self.num_samples):
                sample = self._generate_single_sample(idx, prompt, system_prompt, max_tokens)
                samples.append(sample)

        # Sort by index to maintain order
        samples.sort(key=lambda s: s.index)
        return samples

    def _generate_single_sample(
        self,
        index: int,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
    ) -> SampleResponse:
        """Generate a single sample response."""
        start_time = time.time()
        response = ""
        error = None

        try:
            if self.verbose:
                verbose_print(f"Generating sample {index + 1}/{self.num_samples}", "debug")

            # Build kwargs - only include system_prompt if it's not None
            kwargs = {
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            }
            if system_prompt is not None:
                kwargs["system_prompt"] = system_prompt

            response = self.llm.generate_response(**kwargs)

            if not response:
                error = "Empty response from LLM"

        except Exception as e:
            error = str(e)
            if self.verbose:
                verbose_print(f"Error generating sample {index}: {error}", "error")

        execution_time = time.time() - start_time

        return SampleResponse(
            index=index,
            response=response or "",
            extracted_answer="",  # Will be filled in later
            execution_time=execution_time,
            error=error,
        )

    def _extract_answers(self, samples: List[SampleResponse]) -> List[str]:
        """
        Extract the core answer from each response.

        Uses LLM to identify and extract the final answer from each response.
        """
        # Build a batch extraction prompt
        extraction_prompt = """Extract the final answer from each of the following responses.
For each response, identify ONLY the core answer (the actual result, not the explanation).

Examples:
- "Let me calculate: 17 x 24 = 408. So the answer is 408." → "408"
- "The capital of France is Paris, which has been the capital since..." → "Paris"
- "I believe the answer is TRUE because..." → "TRUE"
- "After analysis, my conclusion is that option B is correct." → "B"

RESPONSES TO ANALYZE:
"""
        for i, sample in enumerate(samples):
            extraction_prompt += f"\n--- Response {i + 1} ---\n{sample.response}\n"

        extraction_prompt += """
Return a JSON object with this format:
{
    "answers": ["answer1", "answer2", ...]
}

Where each answer is the extracted core answer from each response, in order.
Keep answers concise - just the essential answer, not explanations.
"""

        try:
            response = self.llm.generate_response(
                prompt=extraction_prompt,
                system_prompt="You are an answer extraction expert. Extract only the final answers, nothing else.",
                temperature=0.0,  # Use low temperature for extraction
                max_tokens=1024,
                json_mode=True,
            )

            parsed = self._parse_json_response(response)
            if parsed and "answers" in parsed:
                answers = parsed["answers"]
                # Ensure we have the right number of answers
                if len(answers) == len(samples):
                    return [str(a).strip() for a in answers]

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in batch extraction: {e}", "warning")

        # Fallback: try to extract individually or use simple heuristics
        return [self._extract_answer_simple(s.response) for s in samples]

    def _extract_answer_simple(self, response: str) -> str:
        """Simple heuristic extraction when LLM extraction fails."""
        # Look for common answer patterns
        patterns = [
            r"(?:answer|result|solution)[:\s]+(.+?)(?:\.|$)",
            r"(?:is|equals?|=)[:\s]+(.+?)(?:\.|$)",
            r"(?:therefore|thus|so)[,\s]+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                answer = match.group(1).strip()
                if len(answer) < 100:  # Reasonable answer length
                    return answer

        # If no pattern matches, use the last sentence or line
        lines = response.strip().split('\n')
        last_line = lines[-1].strip() if lines else response.strip()

        # Clean up common prefixes
        for prefix in ["Answer:", "Result:", "So,", "Therefore,", "Thus,"]:
            if last_line.lower().startswith(prefix.lower()):
                last_line = last_line[len(prefix):].strip()

        return last_line[:200] if last_line else response[:200]

    def _detect_answer_type(self, answers: List[str]) -> AnswerType:
        """
        Auto-detect whether to use exact or semantic comparison.

        Uses EXACT for:
        - Numbers (including floats)
        - Boolean-like values (yes/no, true/false)
        - Very short answers (< 20 chars)
        - Single word answers

        Uses SEMANTIC for:
        - Longer text answers
        - Multi-word explanations
        """
        def is_exact_type(answer: str) -> bool:
            answer = answer.strip().lower()

            # Check for numbers
            try:
                float(answer.replace(',', '').replace(' ', ''))
                return True
            except ValueError:
                pass

            # Check for boolean-like values
            if answer in ('yes', 'no', 'true', 'false', 'correct', 'incorrect'):
                return True

            # Check for single letter/option answers
            if len(answer) == 1 and answer.isalpha():
                return True

            # Check for short answers
            if len(answer) < 20 and ' ' not in answer:
                return True

            return False

        # If most answers look exact, use exact comparison
        exact_count = sum(1 for a in answers if is_exact_type(a))

        if exact_count >= len(answers) * 0.7:  # 70% threshold
            return AnswerType.EXACT
        else:
            return AnswerType.SEMANTIC

    def _group_answers_exact(self, samples: List[SampleResponse]) -> List[AnswerGroup]:
        """Group answers by exact string match (case-insensitive, normalized)."""
        groups: Dict[str, List[SampleResponse]] = defaultdict(list)

        for sample in samples:
            # Normalize the answer for comparison
            normalized = self._normalize_answer(sample.extracted_answer)
            groups[normalized].append(sample)

        result = []
        for normalized, group_samples in groups.items():
            # Use the first sample's answer as representative
            representative = group_samples[0].extracted_answer

            result.append(AnswerGroup(
                answer=representative,
                count=len(group_samples),
                sample_indices=[s.index for s in group_samples],
                responses=[s.response for s in group_samples],
                percentage=(len(group_samples) / len(samples)) * 100,
            ))

        return result

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        answer = answer.strip().lower()

        # Remove common punctuation
        answer = re.sub(r'[.,!?;:]+$', '', answer)

        # Normalize whitespace
        answer = ' '.join(answer.split())

        # Try to normalize numbers
        try:
            num = float(answer.replace(',', '').replace(' ', ''))
            # Return as clean number string
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            pass

        return answer

    def _group_answers_semantic(self, samples: List[SampleResponse]) -> List[AnswerGroup]:
        """Group answers by semantic similarity using LLM."""
        if len(samples) <= 1:
            if samples:
                return [AnswerGroup(
                    answer=samples[0].extracted_answer,
                    count=1,
                    sample_indices=[samples[0].index],
                    responses=[samples[0].response],
                    percentage=100.0,
                )]
            return []

        # Build a clustering prompt
        clustering_prompt = """Analyze these answers and group the ones that are semantically equivalent
(same meaning, even if worded differently).

ANSWERS:
"""
        for i, sample in enumerate(samples):
            clustering_prompt += f"{i}: {sample.extracted_answer}\n"

        clustering_prompt += """
Return a JSON object with this format:
{
    "groups": [[0, 2, 4], [1, 3]],
    "representative_answers": ["answer for group 1", "answer for group 2"]
}

Where:
- "groups" is a list of groups, each group is a list of answer indices that mean the same thing
- "representative_answers" is the best/clearest answer to represent each group

Every index (0 to """ + str(len(samples) - 1) + """) must appear in exactly one group.
"""

        try:
            response = self.llm.generate_response(
                prompt=clustering_prompt,
                system_prompt="You are a semantic analysis expert. Group answers by meaning.",
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            )

            parsed = self._parse_json_response(response)
            if parsed and "groups" in parsed:
                grouping = SemanticGrouping(
                    groups=parsed["groups"],
                    representative_answers=parsed.get("representative_answers", [])
                )

                # Validate and convert to AnswerGroups
                return self._convert_semantic_grouping(grouping, samples)

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in semantic grouping: {e}", "warning")

        # Fallback to exact matching if semantic fails
        if self.verbose:
            verbose_print("Falling back to exact matching", "warning")
        return self._group_answers_exact(samples)

    def _convert_semantic_grouping(
        self,
        grouping: SemanticGrouping,
        samples: List[SampleResponse],
    ) -> List[AnswerGroup]:
        """Convert LLM semantic grouping result to AnswerGroups."""
        result = []
        sample_by_index = {s.index: s for s in samples}

        for i, indices in enumerate(grouping.groups):
            group_samples = [sample_by_index.get(idx) for idx in indices if idx in sample_by_index]
            group_samples = [s for s in group_samples if s is not None]

            if not group_samples:
                continue

            # Use representative answer if available, otherwise first sample's answer
            if i < len(grouping.representative_answers):
                representative = grouping.representative_answers[i]
            else:
                representative = group_samples[0].extracted_answer

            result.append(AnswerGroup(
                answer=representative,
                count=len(group_samples),
                sample_indices=[s.index for s in group_samples],
                responses=[s.response for s in group_samples],
                percentage=(len(group_samples) / len(samples)) * 100,
            ))

        return result

    def _find_majority(
        self,
        groups: List[AnswerGroup],
    ) -> Tuple[str, bool, List[str]]:
        """
        Find the majority answer from sorted groups.

        Returns: (winner_answer, is_tie, tied_answers)
        """
        if not groups:
            return "", False, []

        max_count = groups[0].count
        tied_groups = [g for g in groups if g.count == max_count]

        if len(tied_groups) > 1:
            # We have a tie
            return (
                tied_groups[0].answer,
                True,
                [g.answer for g in tied_groups]
            )
        else:
            # Clear winner
            return groups[0].answer, False, []

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        if not response:
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_patterns = [
            r'\{[\s\S]*\}',  # Any JSON object (greedy)
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None
