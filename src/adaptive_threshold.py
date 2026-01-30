"""
Self-Adaptive Similarity Threshold for Semantic Caching.

Dynamically adapts cache hit threshold based on query category and length.

Usage:
    from src import AdaptiveSimilarityEvaluation
    
    evaluator = AdaptiveSimilarityEvaluation(config)
    score = evaluator.evaluation(src_dict, cache_dict)
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

    FACTUAL = "factual"
    SUBJECTIVE = "subjective"
    COMPARISON = "comparison"
    MATHEMATICAL = "mathematical"
    CREATIVE = "creative"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class ThresholdRule:
    """Threshold adjustment rule per category."""

    base_threshold: float
    length_adjustment: float
    min_threshold: float = 0.5
    max_threshold: float = 0.95


class AdaptiveSimilarityEvaluation(SimilarityEvaluation):
    """
    Adaptive threshold similarity evaluator.
    
    Adjusts cache hit threshold based on query category and length.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: str = None,
        device: str = None,
        threshold_rules: Optional[Dict[QueryCategory, ThresholdRule]] = None,
        threshold_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize adaptive evaluator.

        Args:
            config: Configuration dictionary.
            model: Huggingface model name (overrides config).
            device: Device (overrides config).
            threshold_rules: Full rule specification per category.
            threshold_overrides: Override specific categories:
                {"FACTUAL": {"base": 0.89, "adj": -0.008}}
        """
        embedding_config = config.get("embedding", {})
        model_name = model or embedding_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        device_name = device or embedding_config.get("device", "cpu")

        self.model = Huggingface(model_name)
        self.model_name = model_name
        self.device = device_name

        self.base_threshold = config.get("cache", {}).get("similarity_threshold", 0.80)
        self.base_evaluation = SearchDistanceEvaluation()

        default_rules = {
            QueryCategory.FACTUAL: ThresholdRule(
                base_threshold=0.880,
                length_adjustment=-0.010,
                min_threshold=0.75,
                max_threshold=0.95,
            ),
            QueryCategory.SUBJECTIVE: ThresholdRule(
                base_threshold=0.850,
                length_adjustment=-0.015,
                min_threshold=0.70,
                max_threshold=0.93,
            ),
            QueryCategory.COMPARISON: ThresholdRule(
                base_threshold=0.890,
                length_adjustment=-0.012,
                min_threshold=0.78,
                max_threshold=0.95,
            ),
            QueryCategory.MATHEMATICAL: ThresholdRule(
                base_threshold=0.890,
                length_adjustment=-0.005,
                min_threshold=0.80,
                max_threshold=0.95,
            ),
            QueryCategory.CREATIVE: ThresholdRule(
                base_threshold=0.610,
                length_adjustment=-0.025,
                min_threshold=0.45,
                max_threshold=0.80,
            ),
            QueryCategory.CODE: ThresholdRule(
                base_threshold=0.910,
                length_adjustment=-0.007,
                min_threshold=0.85,
                max_threshold=0.95,
            ),
            QueryCategory.UNKNOWN: ThresholdRule(
                base_threshold=0.800,
                length_adjustment=0.000,
            ),
        }

        if threshold_rules is not None:
            self.threshold_rules = threshold_rules
        elif threshold_overrides is not None:
            self.threshold_rules = default_rules.copy()
            for category_str, values in threshold_overrides.items():
                try:
                    category = QueryCategory[category_str.upper()]
                    self.threshold_rules[category] = ThresholdRule(
                        base_threshold=values.get("base", 0.80),
                        length_adjustment=values.get("adj", 0.0),
                    )
                except KeyError:
                    logger.warning(f"Unknown category '{category_str}', skipping")
        else:
            self.threshold_rules = default_rules

        self._query_history: Dict[str, Dict[str, Any]] = {}
        self.last_computed_threshold = None
        self.last_similarity = None
        self.last_category = None

        logger.info(f"AdaptiveSimilarityEvaluation initialized: model={model_name}")

    def _classify_query(self, query: str) -> QueryCategory:
        """Classify query into semantic category."""
        query_lower = query.lower()

        math_patterns = [
            r"\d+\s*[\+\-\*/\^]\s*\d+",
            r"\bcalculate\s+\w+",
            r"\bcompute\s+\w+",
            r"\bsolve\s+(for|the|this)",
            r"\bequation\b",
            r"\bformula\b",
            r"\bderivative\b",
            r"\bintegral\b",
            r"\bsum\s+of\s+\d+",
            r"\bsquare\s+root\b",
            r"\btimes\s+\d+",
            r"\bmultipl(y|ied)\s+\d+",
            r"\bdivide\s+\d+",
        ]
        if any(re.search(pattern, query_lower) for pattern in math_patterns):
            return QueryCategory.MATHEMATICAL

        comparison_patterns = [
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bdifference\s+between\b",
            r"\bdifferences?\s+between\b",
            r"\bcompare\s+\w+\s+(and|with|to)\b",
            r"\bbetter\s+than\b",
            r"\bworse\s+than\b",
            r"\bsimilar(ities)?\s+(to|with)\b",
            r"\bwhich\s+is\s+(better|worse|more|less)\b",
            r"\b(differ|differs?)\s+(from|between)\b",
        ]
        if any(re.search(pattern, query_lower) for pattern in comparison_patterns):
            return QueryCategory.COMPARISON

        subjective_patterns = [
            r"\bopinion\b",
            r"\bdo\s+you\s+(think|believe)\b",
            r"\bthink\s+(about|of|that)\b",
            r"\bfeel\s+about\b",
            r"\bthoughts\s+on\b",
            r"\bfavorite\b",
            r"\bprefer\b",
            r"\brecommend\b",
            r"\bsuggestion\b",
            r"\badvice\b",
            r"\bbelieve\s+(in|that)\b",
            r"\bshould\s+i\s+(buy|get|choose|learn|start|try|go|do|use)\b",
            r"\bis\s+it\s+worth\b",
            r"\bworth\s+(it|buying|getting|learning)\b",
            r"\bwhich\s+is\s+better\b",
            r"\bis\s+\w+\s+better\s+than\b",
            r"\bis\s+it\s+(better|worse|good|bad)\b",
            r"\bwhat'?s\s+(the\s+)?(best|better|worst|worse)\b",
            r"\bwhat\s+is\s+the\s+best\s+(way\s+to|method|approach|option)\b",
            r"\bbest\s+way\s+to\s+(learn|study|prepare|improve|start)\b",
            r"\bhow\s+(do|can|should)\s+i\s+(start|learn|improve|prepare|become|get)\b",
            r"\bwhat\s+should\s+i\s+(do|learn|study|practice|use)\b",
            r"\bguide\s+(me|for|to)\b",
            r"\bhelp\s+me\s+(decide|choose|learn|understand)\b",
            r"\btips\s+(for|to)\b",
            r"\bsteps\s+to\s+(become|learn|start|get)\b",
            r"\bways\s+to\s+(improve|learn|start)\b",
            r"\bhow\s+to\s+",
            r"\bcan\s+you\s+(help|show|tell|explain|suggest|recommend)\b",
            r"\bcould\s+you\s+(help|show|tell|explain)\b",
        ]
        if any(re.search(pattern, query_lower) for pattern in subjective_patterns):
            return QueryCategory.SUBJECTIVE

        code_keywords = [
            "function", "method", "class", "variable", "loop", "array",
            "algorithm", "syntax", "compile", "debug", "error message",
            "exception", "implement", "code snippet", "api call", "library",
        ]
        code_count = sum(1 for kw in code_keywords if kw in query_lower)
        specific_code = any(phrase in query_lower for phrase in [
            "python code", "javascript code", "write a function",
            "implement a", "code snippet", "api call", "error message",
            "syntax error", "compile error",
        ])
        if code_count >= 2 or specific_code:
            return QueryCategory.CODE

        fact_patterns = [
            r"\bwhat'?s\s+(?!(the\s+)?(best|better|worst|worse))",
            r"\bwhat\s+(is|are|was|were)\s+(?!(the\s+)?(best|better|worst|worse))",
            r"\bwhat\s+(if|about)\s+",
            r"\bwhat\s+causes\b",
            r"\bwhat\s+type\s+",
            r"\bwhy\s+(does|do|did|is|are|was|were|would|should|can|cannot)\s+",
            r"\bwhy\s+not\b",
            r"\bwho'?s?\s+(is|are|was|were)\s+",
            r"\bwho\s+(is|are|was|were|invented|discovered)\s+",
            r"\bwhen\s+(did|does|is|was|were|will)\s+",
            r"\bwhere\s+(is|are|can|could|did|do)\s+",
            r"\bwhich\s+(country|city|place|year|one)\s+",
            r"\bdoes\s+\w+\s+(have|offer|provide|support|work|mean|exist|make\s+sense)\b",
            r"\bdo\s+(they|people|we|you)\s+(have|offer|provide|know)\b",
            r"\bdid\s+\w+\s+(happen|exist|work|succeed)\b",
            r"\bhas\s+(anyone|someone|it|this)\s+",
            r"\bhave\s+(you|they|people)\s+(seen|tried|heard)\b",
            r"\bis\s+there\s+(a|an|any)\s+",
            r"\bis\s+it\s+possible\s+to\b",
            r"\bare\s+there\s+(any|some|many)\s+",
            r"\bare\s+(you|they)\s+able\s+to\b",
            r"\bwas\s+\w+\s+(always|originally|initially)\b",
            r"\bwere\s+(they|these|those)\s+",
            r"\bwill\s+\w+\s+(happen|work|be|exist)\b",
            r"\bwould\s+\w+\s+(work|be|happen)\b",
            r"\bdefine\s+\w+",
            r"\bdefinition\s+of\b",
            r"\bwhat\s+does\s+\w+\s+mean\b",
            r"\bwhat\s+is\s+the\s+(capital|population|currency|meaning|purpose)\b",
        ]
        if any(re.search(pattern, query_lower) for pattern in fact_patterns):
            return QueryCategory.FACTUAL

        creative_keywords = [
            "write a story", "write a poem", "create a", "generate a",
            "design a", "compose a", "imagine", "invent a",
        ]
        if any(keyword in query_lower for keyword in creative_keywords):
            return QueryCategory.CREATIVE

        return QueryCategory.UNKNOWN

    def _calculate_adaptive_threshold(
        self, query: str, category: QueryCategory
    ) -> float:
        """Calculate adaptive threshold for query."""
        rule = self.threshold_rules[category]
        threshold = rule.base_threshold
        length_factor = len(query) // 10
        threshold += length_factor * rule.length_adjustment
        threshold = max(rule.min_threshold, min(rule.max_threshold, threshold))
        self.last_computed_threshold = threshold
        return threshold

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """Evaluate similarity with adaptive threshold."""
        base_score = self.base_evaluation.evaluation(src_dict, cache_dict, **kwargs)
        query = src_dict.get("question", src_dict.get("query", ""))
        category = self._classify_query(query)
        adaptive_threshold = self._calculate_adaptive_threshold(query, category)

        logger.debug(
            f"Query: '{query[:60]}...' | Category: {category.value} | "
            f"Threshold: {adaptive_threshold:.3f} | Similarity: {base_score:.3f} | "
            f"Match: {base_score >= adaptive_threshold}"
        )

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
        """Get the adaptive threshold for a query."""
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
    """Cache implementation with adaptive threshold evaluation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the cache."""
        self.config = config
        self.evaluator = None
        self.cache_initialized = False

    def initialize(self) -> None:
        """Initialize the cache with adaptive threshold evaluation."""
        embedding_config = self.config.get("embedding", {})
        cache_config = self.config.get("cache", {})
        model = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        threshold = cache_config.get("similarity_threshold", 0.80)

        model_instance = Huggingface(model)
        embedding_dim = model_instance.dimension
        embedding_func = model_instance.to_embeddings

        self.evaluator = AdaptiveSimilarityEvaluation(self.config)

        backend = cache_config.get("backend", "faiss")
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
        return 0.80

    def get_stats(self) -> Dict[str, Any]:
        """Return query statistics."""
        if self.evaluator:
            return self.evaluator.get_query_stats()
        return {"total_queries": 0}

    def __repr__(self) -> str:
        return "AdaptiveThresholdCache(adaptive_thresholds=True)"
