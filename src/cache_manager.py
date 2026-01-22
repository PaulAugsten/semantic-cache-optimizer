"""
Cache manager module.

Provides unified interface for different cache strategies.
"""

from typing import Dict, Any, Optional, Tuple
import time
from loguru import logger

from .similarity_evaluators import (
    FixedThresholdCache,
    LengthBasedCache,
    DensityBasedCache,
    ScoreGapCache,
    AdaptiveThresholdCache,
)
from .llm_adapter import create_llm


class CacheManager:
    """
    Unified cache manager supporting multiple threshold strategies.
    """

    STRATEGIES = {
        "fixed": FixedThresholdCache,
        "length_based": LengthBasedCache,
        "density_based": DensityBasedCache,
        "score_gap": ScoreGapCache,
        "adaptive": AdaptiveThresholdCache,
    }

    def __init__(self, config: Dict[str, Any], strategy: str = "fixed"):
        """
        Initialize cache manager.

        Args:
            config: Configuration dictionary.
            strategy: Cache strategy name.
        """
        self.config = config
        self.strategy_name = strategy
        self.cache_instance = None
        self.llm = None
        self.query_log = []

    def initialize(self) -> None:
        """Initialize cache and LLM."""
        if self.strategy_name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

        # Create and initialize cache
        cache_class = self.STRATEGIES[self.strategy_name]
        self.cache_instance = cache_class(self.config)
        self.cache_instance.initialize()

        # Create LLM
        self.llm = create_llm(self.config)

        logger.info(f"Cache manager initialized with strategy: {self.strategy_name}")

    def query(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Query the cache or LLM.

        Args:
            prompt: User query.
            ground_truth_is_duplicate: Optional ground truth label.

        Returns:
            Tuple of (response, metadata).
        """

        start_time = time.time()
        response = self.llm(prompt)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        metadata = {
            "latency_ms": latency_ms,
            "strategy": self.strategy_name,
        }

        self.query_log.append(metadata)

        return response, metadata

    def get_query_log(self) -> list:
        """Return the query log."""
        return self.query_log

    def clear_log(self) -> None:
        """Clear the query log."""
        self.query_log = []

    def get_evaluator(self):
        """Return the similarity evaluator."""
        if self.cache_instance:
            return self.cache_instance.get_evaluator()
        return None
