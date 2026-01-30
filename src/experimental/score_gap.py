"""
Score-gap adaptive threshold similarity evaluation.

Uses the difference between Top-1 and Top-2 similarity scores:
- Small gap (uncertainty): Higher threshold (stricter matching).
- Large gap (confidence): Lower threshold (more lenient matching).
"""

from typing import Any, Dict, Tuple, List

import numpy as np
from loguru import logger

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.embedding import Huggingface
from gptcache import cache
from gptcache.manager import CacheBase, VectorBase, get_data_manager

from gptcache.processor.pre import get_prompt


class ScoreGapSimilarityEvaluation(SimilarityEvaluation):
    """
    Score-gap adaptive threshold similarity evaluation.

    Adjusts similarity based on the gap between top-1 and top-2 scores.
    Contains the embedding model internally using gptcache Huggingface.

    Example:
        .. code-block:: python

            from src.similarity_evaluators import ScoreGapSimilarityEvaluation

            evaluation = ScoreGapSimilarityEvaluation(config)
            score = evaluation.evaluation(
                {"question": "What is Python?"},
                {"question": "Explain Python programming"}
            )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: str = None,
        device: str = None,
    ):
        """
        Initialize with score-gap config and embedding model.

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

        # Score-gap threshold config
        gap_config = config.get("adaptive_thresholds", {}).get("score_gap", {})
        self.small_gap_threshold = gap_config.get("small_gap_threshold", 0.92)
        self.large_gap_threshold = gap_config.get("large_gap_threshold", 0.78)
        self.gap_cutoff = gap_config.get("gap_cutoff", 0.1)

        self.base_threshold = config.get("cache", {}).get("similarity_threshold", 0.80)

        # Logging state
        self.last_computed_threshold = None
        self.last_similarity = None
        self.last_top_k_similarities: List[float] = []
        self.last_score_gap = None

        # Store recent similarities for gap calculation
        self._recent_similarities: List[float] = []

        logger.info(f"ScoreGapSimilarityEvaluation initialized: model={model_name}")

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        Evaluate similarity with score-gap adjustment.

        Args:
            src_dict: Dictionary with 'question' key for source query.
            cache_dict: Dictionary with 'question' key for cached query.
            **kwargs: May contain 'top_k_similarities' for gap calc.

        Returns:
            Adjusted similarity score.
        """
        try:
            src_question = src_dict.get("question", "")
            cache_question = cache_dict.get("question", "")

            # Exact match check
            if src_question.lower() == cache_question.lower():
                self.last_similarity = 1.0
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

            # Get top-k similarities from kwargs or use recent history
            top_k_sims = kwargs.get("top_k_similarities", [])
            if not top_k_sims:
                # Use recent similarities
                self._recent_similarities.append(raw_similarity)
                if len(self._recent_similarities) > 5:
                    self._recent_similarities = self._recent_similarities[-5:]
                top_k_sims = self._recent_similarities

            self.last_top_k_similarities = sorted(top_k_sims, reverse=True)

            # Compute score gap between top-1 and top-2
            if len(self.last_top_k_similarities) >= 2:
                score_gap = (
                    self.last_top_k_similarities[0] - self.last_top_k_similarities[1]
                )
            else:
                # Only one result - assume high confidence
                score_gap = self.gap_cutoff * 2

            self.last_score_gap = float(score_gap)

            # Compute adaptive threshold based on gap
            if score_gap < self.gap_cutoff:
                adaptive_threshold = self.small_gap_threshold
            else:
                adaptive_threshold = self.large_gap_threshold

            self.last_computed_threshold = adaptive_threshold

            # Scale similarity to achieve adaptive threshold effect
            threshold_ratio = self.base_threshold / adaptive_threshold
            adjusted_similarity = raw_similarity * threshold_ratio

            logger.debug(
                f"Score-gap: gap={score_gap:.4f}, "
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
            "score_gap": self.last_score_gap,
            "computed_threshold": self.last_computed_threshold,
            "raw_similarity": self.last_similarity,
            "top_k_similarities": self.last_top_k_similarities,
        }

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.dimension


class ScoreGapCache:
    """
    Cache implementation with score-gap adaptive threshold.
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
        """Initialize the cache with score-gap evaluation."""
        # Create evaluator with embedded model
        self.evaluator = ScoreGapSimilarityEvaluation(self.config)

        embedding_dim = self.evaluator.embedding_dimension
        cache_config = self.config.get("cache", {})
        backend = cache_config.get("backend", "faiss")
        top_k = max(cache_config.get("top_k", 5), 2)

        # Create data manager
        cache_base = CacheBase("sqlite")

        if backend == "faiss":
            vector_base = VectorBase(
                "faiss",
                dimension=embedding_dim,
                top_k=top_k,
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
        logger.info("Score-gap adaptive cache initialized")

    def get_evaluator(self) -> ScoreGapSimilarityEvaluation:
        """Return the similarity evaluator."""
        return self.evaluator
