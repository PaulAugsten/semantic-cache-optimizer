"""
Base classes for similarity evaluation.

Provides abstract base class implementing the SimilarityEvaluation interface.
The interface expects evaluation(src_dict, cache_dict) where dicts contain 'question' key.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
from loguru import logger


class BaseSimilarityEvaluation(ABC):
    """
    Abstract base class for similarity evaluation.

    Implements the SimilarityEvaluation interface from gptcache.
    Each subclass should contain its own embedding model.

    The evaluation method receives dictionaries with 'question' keys,
    not raw embeddings.
    """

    @abstractmethod
    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        Evaluate similarity between user request and cached data.

        Args:
            src_dict: Dictionary with 'question' key for source query.
            cache_dict: Dictionary with 'question' key for cached query.
            **kwargs: Additional arguments.

        Returns:
            Similarity score (higher = more similar).
        """
        pass

    @abstractmethod
    def range(self) -> Tuple[float, float]:
        """
        Return the minimum and maximum values for similarity scores.

        Returns:
            Tuple of (min_value, max_value).
        """
        pass

    def get_threshold_info(self) -> Dict[str, Any]:
        """Get threshold info for logging. Override in subclasses."""
        return {}


class BaseCache(ABC):
    """
    Abstract base class for cache implementations.

    Initializes gptcache with the appropriate similarity evaluator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cache.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.cache = None
        self.evaluator = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the cache with appropriate settings."""
        pass

    def get_cache(self):
        """Return the initialized cache."""
        return self.cache

    def get_evaluator(self):
        """Return the similarity evaluator."""
        return self.evaluator
