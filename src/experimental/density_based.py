"""
Density-based adaptive threshold similarity evaluation.

Uses the average similarity of top-k neighbors:
- High average (dense cluster): Higher threshold (stricter matching).
- Low average (sparse): Lower threshold (more lenient matching).
"""

from typing import Any, Dict, Tuple, List

import numpy as np
from loguru import logger

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.embedding import Huggingface
from gptcache import cache
from gptcache.manager import CacheBase, VectorBase, get_data_manager

from gptcache.processor.pre import get_prompt


class DensityBasedSimilarityEvaluation(SimilarityEvaluation):
    """
    Density-based adaptive threshold similarity evaluation.

    Adjusts similarity based on the density of nearby cached items.
    Contains the embedding model internally using gptcache Huggingface.

    Example:
        .. code-block:: python

            from src.similarity_evaluators import DensityBasedSimilarityEvaluation

            evaluation = DensityBasedSimilarityEvaluation(config)
            score = evaluation.evaluation(
                {"question": "What is machine learning?"},
                {"question": "Explain ML"}
            )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: str = None,
        device: str = None,
    ):
        """
        Initialize with density-based config and embedding model.

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

        # Density-based threshold config
        density_config = config.get("adaptive_thresholds", {}).get("density_based", {})
        self.top_k = density_config.get("top_k", 5)
        self.high_density_threshold = density_config.get("high_density_threshold", 0.90)
        self.low_density_threshold = density_config.get("low_density_threshold", 0.80)
        self.density_cutoff = density_config.get("density_cutoff", 0.75)

        self.base_threshold = config.get("cache", {}).get("similarity_threshold", 0.80)

        # Logging state
        self.last_computed_threshold = None
        self.last_similarity = None
        self.last_top_k_similarities: List[float] = []
        self.last_average_density = None

        # Store recent similarities for density calculation
        self._recent_similarities: List[float] = []

        logger.info(f"DensityBasedSimilarityEvaluation initialized: model={model_name}")

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        Evaluate similarity with density-based adjustment.

        Args:
            src_dict: Dictionary with 'question' key for source query.
            cache_dict: Dictionary with 'question' key for cached query.
            **kwargs: May contain 'top_k_similarities' for density calc.

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
                # Use recent similarities as proxy for density
                self._recent_similarities.append(raw_similarity)
                if len(self._recent_similarities) > self.top_k:
                    self._recent_similarities = self._recent_similarities[-self.top_k :]
                top_k_sims = self._recent_similarities

            self.last_top_k_similarities = list(top_k_sims)

            # Compute average density
            avg_density = np.mean(top_k_sims) if top_k_sims else raw_similarity
            self.last_average_density = float(avg_density)

            # Compute adaptive threshold based on density
            if avg_density > self.density_cutoff:
                adaptive_threshold = self.high_density_threshold
            else:
                adaptive_threshold = self.low_density_threshold

            self.last_computed_threshold = adaptive_threshold

            # Scale similarity to achieve adaptive threshold effect
            threshold_ratio = self.base_threshold / adaptive_threshold
            adjusted_similarity = raw_similarity * threshold_ratio

            logger.debug(
                f"Density-based: avg_density={avg_density:.4f}, "
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
            "average_density": self.last_average_density,
            "computed_threshold": self.last_computed_threshold,
            "raw_similarity": self.last_similarity,
            "top_k_similarities": self.last_top_k_similarities,
        }

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.dimension


class DensityBasedCache:
    """
    Cache implementation with density-based adaptive threshold.
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
        """Initialize the cache with density-based evaluation."""
        # Create evaluator with embedded model
        self.evaluator = DensityBasedSimilarityEvaluation(self.config)

        embedding_dim = self.evaluator.embedding_dimension
        cache_config = self.config.get("cache", {})
        backend = cache_config.get("backend", "faiss")
        top_k = cache_config.get("top_k", 5)

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
        logger.info("Density-based adaptive cache initialized")

    def get_evaluator(self) -> DensityBasedSimilarityEvaluation:
        """Return the similarity evaluator."""
        return self.evaluator
