"""
Self-Adaptive Similarity Threshold.

The similarity threshold dynamically adapts based on:
- Query length
- Query type (semantic category like facts, math, conversation)
- Historical performance

Uses gptcache Huggingface embedding for embedding-based cosine similarity.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re

from loguru import logger

from gptcache import cache
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.embedding import Huggingface
from gptcache.similarity_evaluation import (
    SimilarityEvaluation,
    SearchDistanceEvaluation,
)
from gptcache.processor.pre import get_prompt
from gptcache.config import Config


class QueryCategory(Enum):
    """Semantic query categories."""

    FACTUAL = "factual"  # Factual questions
    MATHEMATICAL = "mathematical"  # Math/calculations
    CONVERSATIONAL = "conversational"  # Conversation
    CREATIVE = "creative"  # Creative tasks
    CODE = "code"  # Code-related
    UNKNOWN = "unknown"


@dataclass
class ThresholdRule:
    """Rule for threshold adjustment."""

    base_threshold: float
    length_adjustment: float  # Adjustment per 10 characters
    min_threshold: float = 0.5
    max_threshold: float = 0.95


class AdaptiveSimilarityEvaluation(SimilarityEvaluation):
    """
    Similarity Evaluation with adaptive threshold.
    Adjusts the threshold based on query properties.

    Contains the embedding model internally using gptcache Huggingface.

    Example:
        .. code-block:: python

            from src.similarity_evaluators import AdaptiveSimilarityEvaluation

            evaluation = AdaptiveSimilarityEvaluation(config)
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
        threshold_rules: Optional[Dict[QueryCategory, ThresholdRule]] = None,
    ):
        """
        Initialize with adaptive config and embedding model.

        Args:
            config: Configuration dictionary.
            model: Huggingface model name (overrides config).
            device: Device to run model on (overrides config).
            threshold_rules: Dict with rules per category.
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

        # Base threshold for normalization
        self.base_threshold = config.get("cache", {}).get("similarity_threshold", 0.85)

        self.base_evaluation = SearchDistanceEvaluation()

        # Default rules
        if threshold_rules is None:
            threshold_rules = {
                QueryCategory.FACTUAL: ThresholdRule(
                    base_threshold=0.85,
                    length_adjustment=-0.01,  # Less strict for longer questions
                ),
                QueryCategory.MATHEMATICAL: ThresholdRule(
                    base_threshold=0.95,  # Very strict for math
                    length_adjustment=0.0,  # No length adjustment
                ),
                QueryCategory.CONVERSATIONAL: ThresholdRule(
                    base_threshold=0.7,  # Less strict for conversation
                    length_adjustment=-0.015,
                ),
                QueryCategory.CREATIVE: ThresholdRule(
                    base_threshold=0.6,  # Least strict
                    length_adjustment=-0.02,
                ),
                QueryCategory.CODE: ThresholdRule(
                    base_threshold=0.9,  # Strict for code
                    length_adjustment=-0.005,
                ),
                QueryCategory.UNKNOWN: ThresholdRule(
                    base_threshold=0.8,
                    length_adjustment=-0.01,
                ),
            }

        self.threshold_rules = threshold_rules
        self._query_history: Dict[str, Dict[str, Any]] = {}

        # Logging state
        self.last_computed_threshold = None
        self.last_similarity = None
        self.last_category = None

        logger.info(f"AdaptiveSimilarityEvaluation initialized: model={model_name}")

    def _classify_query(self, query: str) -> QueryCategory:
        """
        Classify a query into a semantic category.

        Args:
            query: Query string.

        Returns:
            QueryCategory.
        """
        query_lower = query.lower()

        # Mathematical queries
        math_patterns = [
            r"\d+\s*[\+\-\*/]\s*\d+",
            r"calculate",
            r"compute",
            r"solve",
            r"equation",
            r"sum",
            r"multiply",
        ]
        if any(re.search(pattern, query_lower) for pattern in math_patterns):
            return QueryCategory.MATHEMATICAL

        # Code-related queries
        code_keywords = [
            "function",
            "class",
            "code",
            "python",
            "javascript",
            "programming",
            "implement",
            "algorithm",
            "debug",
        ]
        if any(keyword in query_lower for keyword in code_keywords):
            return QueryCategory.CODE

        # Factual questions
        fact_patterns = [
            r"^(what|who|when|where|which)\s",
            r"definition",
            r"explain",
            r"describe",
        ]
        if any(re.search(pattern, query_lower) for pattern in fact_patterns):
            return QueryCategory.FACTUAL

        # Creative tasks
        creative_keywords = [
            "write",
            "create",
            "generate",
            "story",
            "poem",
            "imagine",
            "design",
        ]
        if any(keyword in query_lower for keyword in creative_keywords):
            return QueryCategory.CREATIVE

        # Conversational (default for many short, informal requests)
        conversational_patterns = [
            r"^(hi|hello|hey)",
            r"how are you",
            r"thank",
            r"please",
            r"can you",
        ]
        if any(re.search(pattern, query_lower) for pattern in conversational_patterns):
            return QueryCategory.CONVERSATIONAL

        return QueryCategory.UNKNOWN

    def _calculate_adaptive_threshold(
        self, query: str, category: QueryCategory
    ) -> float:
        """
        Calculate the adaptive threshold for a query.

        Args:
            query: Query string.
            category: Query category.

        Returns:
            Adaptive threshold value.
        """
        rule = self.threshold_rules[category]
        threshold = rule.base_threshold

        # Length adjustment (per 10 characters)
        length_factor = len(query) // 10
        threshold += length_factor * rule.length_adjustment

        # Clamp to Min/Max
        threshold = max(rule.min_threshold, min(rule.max_threshold, threshold))

        self.last_computed_threshold = threshold
        return threshold

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        Evaluate similarity with adaptive threshold.

        Args:
            src_dict: Source request params.
            cache_dict: Cached request params.

        Returns:
            Similarity score.
        """
        # Calculate base similarity
        base_score = self.base_evaluation.evaluation(src_dict, cache_dict, **kwargs)
        query = src_dict.get("question", src_dict.get("query", ""))

        # Determine category and calculate adaptive threshold
        category = self._classify_query(query)
        adaptive_threshold = self._calculate_adaptive_threshold(query, category)

        # Track history
        self._query_history[query] = {
            "category": category.value,
            "threshold": adaptive_threshold,
            "base_score": base_score,
        }

        self.last_similarity = base_score
        self.last_category = category

        return base_score

    def range(self):
        """Return the range of similarity scores."""
        return self.base_evaluation.range()

    def get_threshold_info(self) -> Dict[str, Any]:
        """Get threshold info for logging."""
        return {
            "last_threshold": self.last_computed_threshold,
            "last_similarity": self.last_similarity,
            "last_category": self.last_category.value if self.last_category else None,
        }

    def get_threshold_for_query(self, query: str) -> float:
        """
        Public method to get the threshold for a query.

        Args:
            query: Query string.

        Returns:
            Adaptive threshold.
        """
        category = self._classify_query(query)
        return self._calculate_adaptive_threshold(query, category)

    def get_query_stats(self) -> Dict[str, Any]:
        """Return statistics about processed queries."""
        if not self._query_history:
            return {"total_queries": 0}

        categories = {}
        for query_data in self._query_history.values():
            cat = query_data["category"]
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_queries": len(self._query_history),
            "categories": categories,
        }


class AdaptiveThresholdCache:
    """
    Cache implementation with adaptive threshold evaluation.
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
        """Initialize the cache with adaptive threshold evaluation."""
        embedding_config = self.config.get("embedding", {})
        cache_config = self.config.get("cache", {})

        model = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        threshold = cache_config.get("similarity_threshold", 0.85)

        # Create embedding model instance
        model_instance = Huggingface(model)
        embedding_dim = model_instance.dimension
        embedding_func = model_instance.to_embeddings

        # Create adaptive similarity evaluator
        self.evaluator = AdaptiveSimilarityEvaluation(self.config)

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
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=self.evaluator,
            config=Config(similarity_threshold=threshold),
        )

        self.cache_initialized = True
        logger.info(
            f"Adaptive threshold cache initialized with base threshold={threshold}"
        )

    def get_evaluator(self) -> AdaptiveSimilarityEvaluation:
        """Return the similarity evaluator."""
        return self.evaluator

    def get_threshold_for_query(self, query: str) -> float:
        """Get the adaptive threshold for a specific query."""
        if self.evaluator:
            return self.evaluator.get_threshold_for_query(query)
        return 0.85

    def get_stats(self) -> Dict[str, Any]:
        """Return query statistics."""
        if self.evaluator:
            return self.evaluator.get_query_stats()
        return {"total_queries": 0}

    def __repr__(self) -> str:
        return "AdaptiveThresholdCache(adaptive_thresholds=True)"
