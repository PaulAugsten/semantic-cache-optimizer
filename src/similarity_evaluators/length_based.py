"""
Length-based adaptive threshold similarity evaluation.

For short queries (< 5 words): increase threshold to 0.92 due to ambiguity.
For longer queries (> 10 words): decrease threshold to 0.78 as context is clearer.
"""

from typing import Any, Dict, Tuple

import numpy as np
from loguru import logger

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.embedding import Huggingface
from gptcache import cache
from gptcache.manager import CacheBase, VectorBase, get_data_manager

from ..preprocessing import count_words
from gptcache.processor.pre import get_prompt


class LengthBasedSimilarityEvaluation(SimilarityEvaluation):
    """
    Length-based adaptive threshold similarity evaluation.

    Adjusts the effective similarity score based on query length.
    Contains the embedding model internally using gptcache Huggingface.

    Example:
        .. code-block:: python

            from src.similarity_evaluators import LengthBasedSimilarityEvaluation

            evaluation = LengthBasedSimilarityEvaluation(config)
            score = evaluation.evaluation(
                {"question": "What is AI?"},
                {"question": "What is artificial intelligence?"}
            )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: str = None,
        device: str = None,
    ):
        """
        Initialize with length-based config and embedding model.

        Args:
            config: Configuration with threshold settings.
            model: Huggingface model name (overrides config).
            device: Device to run model on (overrides config).
        """
        # Get embedding config
        embedding_config = config.get("embedding", {})
        model_name = model or embedding_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        device_name = device or embedding_config.get("device", "cpu")

        self.model = Huggingface(model_name)
        self.model_name = model_name
        self.device = device_name

        # Length-based threshold config
        length_config = config.get("adaptive_thresholds", {}).get("length_based", {})
        self.short_query_words = length_config.get("short_query_words", 5)
        self.long_query_words = length_config.get("long_query_words", 10)
        self.short_threshold = length_config.get("short_threshold", 0.92)
        self.medium_threshold = length_config.get("medium_threshold", 0.80)
        self.long_threshold = length_config.get("long_threshold", 0.78)

        # Base threshold for normalization
        self.base_threshold = config.get("cache", {}).get("similarity_threshold", 0.80)

        # Logging state
        self.last_computed_threshold = None
        self.last_similarity = None
        self.last_query_length = None

        logger.info(f"LengthBasedSimilarityEvaluation initialized: model={model_name}")

    def _compute_adaptive_threshold(self, query: str) -> float:
        """
        Compute adaptive threshold based on query length.

        Args:
            query: The query text.

        Returns:
            Computed adaptive threshold.
        """
        word_count = count_words(query)
        self.last_query_length = word_count

        if word_count < self.short_query_words:
            threshold = self.short_threshold
        elif word_count > self.long_query_words:
            threshold = self.long_threshold
        else:
            threshold = self.medium_threshold

        self.last_computed_threshold = threshold
        return threshold

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **_
    ) -> float:
        """
        Evaluate similarity with length-based adjustment.

        Args:
            src_dict: Dictionary with 'question' key for source query.
            cache_dict: Dictionary with 'question' key for cached query.

        Returns:
            Adjusted similarity score.
        """
        try:
            src_question = src_dict.get("question", "")
            cache_question = cache_dict.get("question", "")

            # Exact match check
            if src_question.lower() == cache_question.lower():
                self.last_similarity = 1.0
                self.last_computed_threshold = self.base_threshold
                return 1.0

            # Compute embeddings using Huggingface to_embeddings
            src_embedding = np.array(self.model.to_embeddings(src_question))
            cache_embedding = np.array(self.model.to_embeddings(cache_question))

            # Normalize embeddings for cosine similarity
            src_norm = src_embedding / np.linalg.norm(src_embedding)
            cache_norm = cache_embedding / np.linalg.norm(cache_embedding)

            # Cosine similarity
            raw_similarity = float(np.dot(src_norm, cache_norm))
            self.last_similarity = raw_similarity

            # Compute adaptive threshold based on source query length
            adaptive_threshold = self._compute_adaptive_threshold(src_question)

            # Scale similarity to achieve adaptive threshold effect
            threshold_ratio = self.base_threshold / adaptive_threshold
            adjusted_similarity = raw_similarity * threshold_ratio

            logger.debug(
                f"Length-based: words={self.last_query_length}, "
                f"threshold={adaptive_threshold:.2f}, "
                f"raw_sim={raw_similarity:.4f}, "
                f"adj_sim={adjusted_similarity:.4f}"
            )

            return float(min(adjusted_similarity, 1.0))

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return 0.0

    def range(self) -> Tuple[float, float]:
        """Return similarity score range."""
        return (0.0, 1.0)

    def get_threshold_info(self) -> Dict[str, Any]:
        """Return last computed threshold info for logging."""
        return {
            "query_length": self.last_query_length,
            "computed_threshold": self.last_computed_threshold,
            "raw_similarity": self.last_similarity,
        }

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.dimension


class LengthBasedCache:
    """
    Cache implementation with length-based adaptive threshold.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cache.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.evaluator = None
        self.cache_initialized = False

    def initialize(self) -> None:
        """Initialize the cache with length-based evaluation."""
        # Create evaluator with embedded model
        self.evaluator = LengthBasedSimilarityEvaluation(self.config)

        embedding_dim = self.evaluator.embedding_dimension
        cache_config = self.config.get("cache", {})
        backend = cache_config.get("backend", "faiss")

        # Create data manager
        cache_base = CacheBase("sqlite")

        if backend == "faiss":
            vector_base = VectorBase(
                "faiss",
                dimension=embedding_dim,
                top_k=cache_config.get("top_k", 5),
            )
        else:
            vector_base = VectorBase("sqlite", dimension=embedding_dim)

        data_manager = get_data_manager(cache_base, vector_base)

        # Initialize gptcache
        cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=self.evaluator.model.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=self.evaluator,
        )

        self.cache_initialized = True
        logger.info("Length-based adaptive cache initialized")

    def get_evaluator(self) -> LengthBasedSimilarityEvaluation:
        """Return the similarity evaluator."""
        return self.evaluator
