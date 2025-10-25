"""
Query Classifier - Hybrid query classification system.

This module provides classification of user queries using pattern matching,
LLM-based classification, and caching for performance.
"""

import re
import hashlib
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from SimplerLLM.language.llm.base import LLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import QueryClassification, CachedClassification, LLMClassificationResponse


# Default pattern rules for common query types
DEFAULT_PATTERNS = {
    "coding": [
        r"write\s+(a\s+)?code",
        r"write\s+(a\s+)?function",
        r"write\s+(a\s+)?script",
        r"implement",
        r"debug",
        r"python",
        r"javascript",
        r"java\s+code",
        r"how\s+to\s+code",
        r"create\s+(a\s+)?script",
        r"programming",
        r"algorithm",
        r"fix\s+(this|my)\s+code",
        r"code\s+for",
    ],
    "creative_writing": [
        r"write\s+(a\s+)?story",
        r"write\s+(a\s+)?poem",
        r"write\s+(an\s+)?article",
        r"write\s+(a\s+)?blog",
        r"creative",
        r"narrative",
        r"fiction",
        r"essay",
        r"haiku",
        r"sonnet",
    ],
    "technical_explanation": [
        r"explain\s+how",
        r"what\s+is",
        r"how\s+does.*work",
        r"technical",
        r"describe",
        r"define",
        r"explain\s+the",
        r"tell\s+me\s+about",
    ],
    "analysis": [
        r"analyze",
        r"compare",
        r"evaluate",
        r"assess",
        r"review",
        r"pros\s+and\s+cons",
        r"advantages.*disadvantages",
        r"differences?\s+between",
        r"similarities?\s+between",
    ],
    "data_analysis": [
        r"analyze.*data",
        r"statistics",
        r"dataset",
        r"\bsql\b",
        r"query",
        r"data\s+science",
        r"visualization",
        r"pandas",
        r"dataframe",
    ],
    "general": [],  # Catch-all for unmatched queries
}


class QueryClassifier:
    """
    Hybrid query classification system.

    Supports three classification methods:
    - pattern: Fast regex-based matching
    - llm: Accurate LLM-based classification
    - hybrid: Try patterns first, fall back to LLM

    Example:
        ```python
        classifier = QueryClassifier(
            classifier_llm=llm,
            method="hybrid",
            enable_cache=True
        )

        classification = classifier.classify("Write a Python function")
        print(classification.query_type)  # "coding"
        ```
    """

    def __init__(
        self,
        classifier_llm: Optional[LLM] = None,
        method: str = "hybrid",
        enable_cache: bool = False,
        cache_ttl: Optional[int] = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        verbose: bool = False,
    ):
        """
        Initialize Query Classifier.

        Args:
            classifier_llm: LLM instance for LLM-based classification
            method: Classification method - "pattern", "llm", or "hybrid"
            enable_cache: Whether to cache classifications
            cache_ttl: Cache TTL in seconds (None = no expiration)
            custom_patterns: Custom pattern rules (merged with defaults)
            verbose: Enable detailed logging
        """
        self.classifier_llm = classifier_llm
        self.method = method
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.verbose = verbose

        # Validate method
        if method not in ["pattern", "llm", "hybrid"]:
            raise ValueError(f"Invalid method '{method}'. Must be 'pattern', 'llm', or 'hybrid'")

        # Validate LLM requirement
        if method in ["llm", "hybrid"] and classifier_llm is None:
            raise ValueError(f"Method '{method}' requires classifier_llm to be provided")

        # Build pattern rules (merge defaults with custom)
        self.patterns = DEFAULT_PATTERNS.copy()
        if custom_patterns:
            for query_type, patterns in custom_patterns.items():
                if query_type in self.patterns:
                    self.patterns[query_type].extend(patterns)
                else:
                    self.patterns[query_type] = patterns

        # Compile regex patterns for performance
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for query_type, pattern_list in self.patterns.items():
            self.compiled_patterns[query_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
            ]

        # Initialize cache
        self.cache: Dict[str, CachedClassification] = {}

        if self.verbose:
            verbose_print(
                f"Initialized QueryClassifier with method='{method}', "
                f"cache={enable_cache}, patterns={len(self.patterns)} types",
                "info"
            )

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query using the configured method.

        Args:
            query: The query to classify

        Returns:
            QueryClassification with type, confidence, and reasoning
        """
        # Step 1: Check cache (if enabled)
        if self.enable_cache:
            cached = self._check_cache(query)
            if cached:
                if self.verbose:
                    verbose_print(f"Cache hit for query: '{query[:50]}...'", "debug")
                return cached

        # Step 2: Classify based on method
        if self.method == "pattern":
            classification = self._classify_by_pattern(query)
            if classification is None:
                # No pattern match, default to "general"
                classification = QueryClassification(
                    query_type="general",
                    confidence=0.5,
                    reasoning="No specific patterns matched, defaulting to general",
                    matched_by="pattern"
                )

        elif self.method == "llm":
            classification = self._classify_by_llm(query)

        else:  # hybrid
            # Try pattern first
            pattern_result = self._classify_by_pattern(query)

            if pattern_result and pattern_result.confidence >= 0.8:
                # High confidence pattern match
                classification = pattern_result
                if self.verbose:
                    verbose_print(f"High confidence pattern match: {pattern_result.query_type}", "debug")
            else:
                # Low confidence or no match, use LLM
                if self.verbose:
                    verbose_print("Pattern confidence low, using LLM classification", "debug")
                classification = self._classify_by_llm(query)

        # Step 3: Cache result (if enabled)
        if self.enable_cache:
            self._cache_classification(query, classification)

        return classification

    def _classify_by_pattern(self, query: str) -> Optional[QueryClassification]:
        """Classify query using regex pattern matching."""
        query_lower = query.lower().strip()

        # Try each query type
        for query_type, compiled_patterns in self.compiled_patterns.items():
            if query_type == "general":
                continue  # Skip general (catch-all)

            for pattern in compiled_patterns:
                if pattern.search(query_lower):
                    return QueryClassification(
                        query_type=query_type,
                        confidence=0.85,  # Pattern matches have good confidence
                        reasoning=f"Query matched pattern '{pattern.pattern}' for {query_type}",
                        matched_by="pattern"
                    )

        # No match found
        return None

    def _classify_by_llm(self, query: str) -> QueryClassification:
        """Classify query using LLM with structured output."""
        if self.verbose:
            verbose_print("Classifying query with LLM...", "debug")

        # Build classification prompt
        query_types = [qt for qt in self.patterns.keys() if qt != "general"]
        types_text = ", ".join(query_types)

        classification_prompt = f"""Classify the following query into one of these types:

Query Types: {types_text}, general

Query: "{query}"

Determine:
1. The primary query type that best matches
2. Your confidence in this classification (0-1 scale)
3. Clear reasoning for your choice
4. Up to 3 alternative types if applicable

If the query doesn't clearly fit any specific type, classify as "general"."""

        try:
            # Use structured output
            llm_result = generate_pydantic_json_model(
                model_class=LLMClassificationResponse,
                prompt=classification_prompt,
                llm_instance=self.classifier_llm,
                max_retries=2,
                system_prompt="You are an expert query classifier. Provide accurate classifications with reasoning.",
            )

            if isinstance(llm_result, str):
                # Error occurred
                raise RuntimeError(f"LLM classification failed: {llm_result}")

            # Convert to QueryClassification
            return QueryClassification(
                query_type=llm_result.query_type,
                confidence=llm_result.confidence,
                reasoning=llm_result.reasoning,
                matched_by="llm",
                alternative_types=llm_result.alternative_types if llm_result.alternative_types else None
            )

        except Exception as e:
            if self.verbose:
                verbose_print(f"LLM classification error: {str(e)}", "error")

            # Fallback to general
            return QueryClassification(
                query_type="general",
                confidence=0.3,
                reasoning=f"LLM classification failed: {str(e)}. Defaulting to general.",
                matched_by="llm"
            )

    def _get_query_hash(self, query: str) -> str:
        """Generate hash for cache key."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _check_cache(self, query: str) -> Optional[QueryClassification]:
        """Check if query classification is cached and valid."""
        query_hash = self._get_query_hash(query)

        if query_hash not in self.cache:
            return None

        cached_entry = self.cache[query_hash]

        # Check TTL if set
        if self.cache_ttl is not None:
            age = (datetime.now() - cached_entry.timestamp).total_seconds()
            if age > self.cache_ttl:
                # Expired
                del self.cache[query_hash]
                return None

        # Update hits
        cached_entry.hits += 1

        # Return classification with cache marker
        classification = cached_entry.classification
        classification.matched_by = "cache"

        return classification

    def _cache_classification(self, query: str, classification: QueryClassification):
        """Cache a query classification."""
        query_hash = self._get_query_hash(query)

        self.cache[query_hash] = CachedClassification(
            query_hash=query_hash,
            classification=classification,
            timestamp=datetime.now(),
            hits=1
        )

        if self.verbose:
            verbose_print(f"Cached classification for '{query[:50]}...'", "debug")

    def add_pattern_rule(self, query_type: str, pattern: str):
        """
        Add a custom pattern rule.

        Args:
            query_type: The query type this pattern should match
            pattern: Regex pattern to match
        """
        if query_type not in self.patterns:
            self.patterns[query_type] = []
            self.compiled_patterns[query_type] = []

        self.patterns[query_type].append(pattern)
        self.compiled_patterns[query_type].append(re.compile(pattern, re.IGNORECASE))

        if self.verbose:
            verbose_print(f"Added pattern '{pattern}' for query type '{query_type}'", "info")

    def clear_cache(self):
        """Clear all cached classifications."""
        self.cache.clear()
        if self.verbose:
            verbose_print("Classification cache cleared", "info")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_hits = sum(entry.hits for entry in self.cache.values())
        return {
            "total_entries": len(self.cache),
            "total_hits": total_hits,
        }

    def get_supported_types(self) -> List[str]:
        """Get list of supported query types."""
        return list(self.patterns.keys())
