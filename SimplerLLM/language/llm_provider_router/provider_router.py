"""
LLM Provider Router - Smart wrapper for intelligent provider routing.

This module provides a high-level API for automatically routing queries to
the most appropriate LLM provider based on query classification.
"""

import time
import json
from typing import List, Optional, Dict, Tuple, Union
from datetime import datetime

from SimplerLLM.language.llm.base import LLM, LLMProvider
from SimplerLLM.language.llm_router import LLMRouter
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import ProviderConfig, QueryClassification, RoutingResult, RouterConfig
from .query_classifier import QueryClassifier


class LLMProviderRouter:
    """
    Smart wrapper for intelligent provider routing.

    Automatically classifies queries and routes to the most appropriate LLM provider.
    Uses existing LLMRouter internally for provider selection logic.

    Example:
        ```python
        from SimplerLLM.language import LLM, LLMProvider, LLMProviderRouter, ProviderConfig

        providers = [
            ProviderConfig(
                llm_provider="OPENAI",
                llm_model="gpt-4",
                specialties=["coding", "technical"],
                description="Best for code"
            ),
        ]

        # Note: Must pass actual LLM instances separately (not in config)
        llm_instances = [LLM.create(LLMProvider.OPENAI, model_name="gpt-4")]

        router = LLMProviderRouter(
            provider_configs=providers,
            llm_instances=llm_instances
        )

        result = router.route("Write a Python function")
        print(result.answer)
        ```
    """

    def __init__(
        self,
        provider_configs: List[ProviderConfig],
        llm_instances: List[LLM],
        default_provider: Optional[LLM] = None,
        classifier_llm: Optional[LLM] = None,
        classification_method: str = "hybrid",
        enable_cache: bool = False,
        cache_ttl: Optional[int] = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        verbose: bool = False,
    ):
        """
        Initialize LLM Provider Router.

        Args:
            provider_configs: List of ProviderConfig objects
            llm_instances: List of LLM instances (same length and order as configs)
            default_provider: Optional default LLM for fallback
            classifier_llm: LLM for query classification (defaults to first provider)
            classification_method: "pattern", "llm", or "hybrid"
            enable_cache: Enable classification caching
            cache_ttl: Cache TTL in seconds
            custom_patterns: Custom pattern rules
            verbose: Enable detailed logging
        """
        if len(provider_configs) != len(llm_instances):
            raise ValueError("provider_configs and llm_instances must have same length")

        if len(provider_configs) == 0:
            raise ValueError("At least one provider must be configured")

        self.provider_configs = provider_configs
        self.llm_instances = llm_instances
        self.default_provider = default_provider
        self.classifier_llm = classifier_llm or llm_instances[0]
        self.classification_method = classification_method
        self.verbose = verbose

        # Create query classifier
        self.classifier = QueryClassifier(
            classifier_llm=self.classifier_llm,
            method=classification_method,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
            custom_patterns=custom_patterns,
            verbose=verbose,
        )

        if self.verbose:
            verbose_print(
                f"Initialized LLMProviderRouter with {len(provider_configs)} providers, "
                f"method='{classification_method}'",
                "info"
            )

    def route(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_k: Optional[int] = None,
        force_provider: Optional[str] = None,
    ) -> RoutingResult:
        """
        Route query to best provider and execute.

        Args:
            query: The user's query
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            max_tokens: Max tokens to generate
            top_k: Try top K providers until success (None = try best only)
            force_provider: Force specific provider (bypass routing)

        Returns:
            RoutingResult with answer and metadata
        """
        start_time = time.time()

        if self.verbose:
            verbose_print(f"Routing query: '{query[:50]}...'", "info")

        # Step 1: Classify query (unless forcing provider)
        if force_provider:
            # Skip classification
            classification = QueryClassification(
                query_type="forced",
                confidence=1.0,
                reasoning=f"Provider '{force_provider}' forced by user",
                matched_by="manual"
            )
            if self.verbose:
                verbose_print(f"Forcing provider: {force_provider}", "info")
        else:
            classification = self.classifier.classify(query)
            if self.verbose:
                verbose_print(
                    f"Classified as '{classification.query_type}' "
                    f"(confidence: {classification.confidence:.2f}, method: {classification.matched_by})",
                    "info"
                )

        # Step 2: Find matching providers
        if force_provider:
            matching_providers = self._find_provider_by_name(force_provider)
            if not matching_providers:
                raise ValueError(f"Forced provider '{force_provider}' not found")
        else:
            matching_providers = self._find_matching_providers(classification.query_type)

        if not matching_providers:
            if self.verbose:
                verbose_print(
                    f"No providers match '{classification.query_type}', using default or first",
                    "warning"
                )
            # Use default or first provider
            if self.default_provider:
                provider_idx = self._get_default_provider_index()
            else:
                provider_idx = 0
            matching_providers = [(provider_idx, self.provider_configs[provider_idx])]

        # Step 3: Select best provider from matches
        selected_idx, selected_config, routing_reasoning, routing_confidence = self._select_best_provider(
            query, matching_providers
        )

        if self.verbose:
            verbose_print(
                f"Selected provider: {selected_config.llm_provider} ({selected_config.llm_model})",
                "info"
            )

        # Step 4: Execute with selected provider (with fallback if needed)
        if top_k and top_k > 1:
            # Try top K providers
            answer, used_fallback, fallback_reason = self._execute_with_top_k(
                query=query,
                matching_providers=matching_providers[:top_k],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if used_fallback:
                # Find which provider actually succeeded
                for idx, config in matching_providers[:top_k]:
                    try:
                        # This is a simplified check - we already have the answer
                        selected_idx = idx
                        selected_config = config
                        break
                    except:
                        continue
        else:
            # Execute with single selected provider
            answer, used_fallback, fallback_reason = self._execute_with_provider(
                provider_index=selected_idx,
                provider_config=selected_config,
                query=query,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Step 5: Build result
        total_time = time.time() - start_time

        result = RoutingResult(
            answer=answer,
            provider_used=selected_config.llm_provider,
            model_used=selected_config.llm_model,
            query_classification=classification,
            routing_confidence=routing_confidence,
            routing_reasoning=routing_reasoning,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            execution_time=total_time,
            timestamp=datetime.now(),
        )

        if self.verbose:
            verbose_print(
                f"Routing complete: {total_time:.2f}s, fallback={used_fallback}",
                "info"
            )

        return result

    async def route_async(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_k: Optional[int] = None,
        force_provider: Optional[str] = None,
    ) -> RoutingResult:
        """
        Async version of route().

        Note: Currently runs synchronously. Full async requires async support in LLM wrappers.
        """
        # TODO: Implement true async when LLM wrappers support it
        return self.route(query, system_prompt, temperature, max_tokens, top_k, force_provider)

    def get_provider_for_query(
        self,
        query: str,
        return_classification: bool = False,
    ) -> Union[LLM, Tuple[LLM, QueryClassification]]:
        """
        Get best provider for query without executing.

        Args:
            query: The query to route
            return_classification: If True, also return classification

        Returns:
            LLM instance, or (LLM, QueryClassification) if return_classification=True
        """
        # Classify
        classification = self.classifier.classify(query)

        # Find matching providers
        matching_providers = self._find_matching_providers(classification.query_type)

        if not matching_providers:
            # Use default or first
            idx = self._get_default_provider_index() if self.default_provider else 0
        else:
            idx, _, _, _ = self._select_best_provider(query, matching_providers)

        provider = self.llm_instances[idx]

        if return_classification:
            return provider, classification
        return provider

    def add_provider(
        self,
        llm: LLM,
        specialties: List[str],
        description: str,
        priority: int = 5,
        fallback_llm: Optional[LLM] = None,
    ) -> int:
        """
        Add a provider dynamically.

        Args:
            llm: LLM instance
            specialties: List of query types this provider handles
            description: Description of provider's strengths
            priority: Priority 1-10
            fallback_llm: Optional fallback LLM

        Returns:
            Index of added provider
        """
        config = ProviderConfig(
            llm_provider=llm.provider.name,
            llm_model=llm.model_name,
            specialties=specialties,
            description=description,
            priority=priority,
            has_fallback=fallback_llm is not None,
            fallback_provider=fallback_llm.provider.name if fallback_llm else None,
            fallback_model=fallback_llm.model_name if fallback_llm else None,
            enabled=True,
        )

        self.provider_configs.append(config)
        self.llm_instances.append(llm)

        if self.verbose:
            verbose_print(
                f"Added provider: {llm.provider.name} ({llm.model_name})",
                "info"
            )

        return len(self.provider_configs) - 1

    def remove_provider(self, index: int):
        """Remove a provider by index."""
        if 0 <= index < len(self.provider_configs):
            removed_config = self.provider_configs.pop(index)
            self.llm_instances.pop(index)

            if self.verbose:
                verbose_print(
                    f"Removed provider: {removed_config.llm_provider}",
                    "info"
                )
        else:
            raise IndexError("Provider index out of range")

    def add_pattern_rule(self, query_type: str, pattern: str):
        """Add custom pattern rule to classifier."""
        self.classifier.add_pattern_rule(query_type, pattern)

    def export_config(self, filepath: str):
        """
        Export router configuration to JSON.

        Note: LLM instances and API keys are NOT exported.
        Configuration includes provider settings, patterns, and classifier settings.
        """
        # Find default provider index
        default_idx = None
        if self.default_provider:
            for idx, llm in enumerate(self.llm_instances):
                if llm == self.default_provider:
                    default_idx = idx
                    break

        # Build config
        config = RouterConfig(
            providers=self.provider_configs,
            default_provider_index=default_idx,
            classification_method=self.classification_method,
            cache_enabled=self.classifier.enable_cache,
            cache_ttl=self.classifier.cache_ttl,
            pattern_rules={k: v for k, v in self.classifier.patterns.items() if k not in ["general"]},
            classifier_provider=self.classifier_llm.provider.name if self.classifier_llm else None,
            classifier_model=self.classifier_llm.model_name if self.classifier_llm else None,
        )

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(config.model_dump(), f, indent=2, default=str)

        if self.verbose:
            verbose_print(f"Configuration exported to {filepath}", "info")

    @classmethod
    def from_config(
        cls,
        filepath: str,
        llm_instances: List[LLM],
        classifier_llm: Optional[LLM] = None,
        default_provider: Optional[LLM] = None,
        verbose: bool = False,
    ) -> "LLMProviderRouter":
        """
        Load router configuration from JSON.

        Note: LLM instances must be provided separately (cannot serialize API keys).

        Args:
            filepath: Path to config JSON
            llm_instances: List of LLM instances (must match config order)
            classifier_llm: Optional classifier LLM
            default_provider: Optional default provider
            verbose: Enable logging

        Returns:
            Configured LLMProviderRouter instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        config = RouterConfig(**config_dict)

        if len(llm_instances) != len(config.providers):
            raise ValueError(
                f"Config has {len(config.providers)} providers but "
                f"{len(llm_instances)} LLM instances provided"
            )

        # Extract custom patterns (exclude defaults)
        custom_patterns = config.pattern_rules if config.pattern_rules else None

        return cls(
            provider_configs=config.providers,
            llm_instances=llm_instances,
            default_provider=default_provider,
            classifier_llm=classifier_llm,
            classification_method=config.classification_method,
            enable_cache=config.cache_enabled,
            cache_ttl=config.cache_ttl,
            custom_patterns=custom_patterns,
            verbose=verbose,
        )

    # ==================== Private Methods ====================

    def _find_matching_providers(
        self,
        query_type: str,
    ) -> List[Tuple[int, ProviderConfig]]:
        """
        Find providers that match the query type.

        Returns list of (index, config) tuples sorted by priority.
        """
        matches = []

        for idx, config in enumerate(self.provider_configs):
            if not config.enabled:
                continue

            if query_type in config.specialties or "general" in config.specialties:
                matches.append((idx, config))

        # Sort by priority (higher first)
        matches.sort(key=lambda x: x[1].priority, reverse=True)

        return matches

    def _find_provider_by_name(self, provider_name: str) -> List[Tuple[int, ProviderConfig]]:
        """Find provider by name."""
        matches = []
        for idx, config in enumerate(self.provider_configs):
            if config.enabled and config.llm_provider == provider_name:
                matches.append((idx, config))
        return matches

    def _select_best_provider(
        self,
        query: str,
        matching_providers: List[Tuple[int, ProviderConfig]],
    ) -> Tuple[int, ProviderConfig, str, float]:
        """
        Select best provider from matches using internal LLMRouter.

        Returns: (provider_index, provider_config, reasoning, confidence)
        """
        if len(matching_providers) == 1:
            # Only one match, use it
            idx, config = matching_providers[0]
            return (
                idx,
                config,
                f"Only provider matching query type: {config.description}",
                0.9,
            )

        # Multiple matches - use LLMRouter to select best
        router = LLMRouter(
            llm_instance=self.classifier_llm,
            confidence_threshold=0.3,
        )

        # Add providers as choices
        for idx, config in matching_providers:
            router.add_choice(
                content=config.description,
                metadata={"index": idx, "config": config},
            )

        # Route to best choice
        result = router.route(query)

        if result is None:
            # Fallback to first match
            idx, config = matching_providers[0]
            return (
                idx,
                config,
                "Router failed, using highest priority provider",
                0.5,
            )

        # Extract selected provider
        selected_choice = router.get_choice(result.selected_index)
        selected_idx = selected_choice[1]["index"]
        selected_config = selected_choice[1]["config"]

        return (
            selected_idx,
            selected_config,
            result.reasoning,
            result.confidence_score,
        )

    def _execute_with_provider(
        self,
        provider_index: int,
        provider_config: ProviderConfig,
        query: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Execute query with provider (with fallback if available).

        Returns: (answer, used_fallback, fallback_reason)
        """
        llm = self.llm_instances[provider_index]

        try:
            if self.verbose:
                verbose_print(f"Executing with {provider_config.llm_provider}...", "debug")

            answer = llm.generate_response(
                prompt=query,
                system_prompt=system_prompt or "You are a helpful AI assistant.",
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return (answer, False, None)

        except Exception as e:
            error_msg = str(e)

            if self.verbose:
                verbose_print(
                    f"Provider {provider_config.llm_provider} failed: {error_msg}",
                    "warning"
                )

            # Try fallback if available
            if provider_config.has_fallback:
                # Note: Fallback LLM is not stored in instance - would need to be configured
                if self.verbose:
                    verbose_print("Provider has fallback configured but not implemented", "warning")

            # Try router's default provider
            if self.default_provider:
                try:
                    if self.verbose:
                        verbose_print("Trying default provider...", "info")

                    answer = self.default_provider.generate_response(
                        prompt=query,
                        system_prompt=system_prompt or "You are a helpful AI assistant.",
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    return (
                        answer,
                        True,
                        f"Primary provider failed: {error_msg}. Used default provider.",
                    )

                except Exception as e2:
                    raise RuntimeError(
                        f"Both primary and default providers failed: {error_msg}, {str(e2)}"
                    )

            # No fallback available
            raise RuntimeError(f"Provider failed and no fallback configured: {error_msg}")

    def _execute_with_top_k(
        self,
        query: str,
        matching_providers: List[Tuple[int, ProviderConfig]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, bool, Optional[str]]:
        """Try top K providers until one succeeds."""
        errors = []

        for idx, config in matching_providers:
            try:
                answer, used_fallback, fallback_reason = self._execute_with_provider(
                    provider_index=idx,
                    provider_config=config,
                    query=query,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (answer, used_fallback, fallback_reason)

            except Exception as e:
                errors.append(f"{config.llm_provider}: {str(e)}")
                continue

        # All failed
        raise RuntimeError(f"All top-K providers failed: {'; '.join(errors)}")

    def _get_default_provider_index(self) -> int:
        """Get index of default provider."""
        if not self.default_provider:
            return 0

        for idx, llm in enumerate(self.llm_instances):
            if llm == self.default_provider:
                return idx

        return 0
