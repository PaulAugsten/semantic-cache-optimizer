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

    FACTUAL = "factual"  # Pure factual questions: "What IS X?"
    SUBJECTIVE = "subjective"  # Opinion, advice, recommendations: "Should I...", "Best way to..."
    COMPARISON = "comparison"  # Comparing entities: "X vs Y", "difference between"
    MATHEMATICAL = "mathematical"  # Math/calculations
    CREATIVE = "creative"  # Creative tasks
    CODE = "code"  # Code-related
    UNKNOWN = "unknown"  # Fallback category


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
        threshold_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize with adaptive config and embedding model.

        Args:
            config: Configuration dictionary.
            model: Huggingface model name (overrides config).
            device: Device to run model on (overrides config).
            threshold_rules: Dict with rules per category (full specification).
            threshold_overrides: Dict to override specific categories, format:
                {"FACTUAL": {"base": 0.89, "adj": -0.008}, "ADVICE": {"base": 0.78, "adj": -0.01}}
                If both threshold_rules and threshold_overrides are provided, threshold_rules takes precedence.
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
        self.base_threshold = config.get("cache", {}).get("similarity_threshold", 0.80)

        self.base_evaluation = SearchDistanceEvaluation()

        # Default rules (GPTCache-optimized baseline)
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

        # Apply threshold logic
        if threshold_rules is not None:
            # Full custom rules provided - use as-is
            self.threshold_rules = threshold_rules
        elif threshold_overrides is not None:
            # Start with defaults and override specific categories
            self.threshold_rules = default_rules.copy()
            for category_str, values in threshold_overrides.items():
                try:
                    category = QueryCategory[category_str.upper()]
                    self.threshold_rules[category] = ThresholdRule(
                        base_threshold=values.get("base", 0.80),
                        length_adjustment=values.get("adj", 0.0),
                    )
                except KeyError:
                    logger.warning(f"Unknown category '{category_str}' in threshold_overrides, skipping")
        else:
            # No custom config - use defaults
            self.threshold_rules = default_rules

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

        # PRIORITY 1: Mathematical queries - very specific patterns
        # Checked first to avoid "misclassification" as FACTUAL or ADVICE
        math_patterns = [
            r"\d+\s*[\+\-\*/\^]\s*\d+",  # Actual math operations: "5 + 3"
            r"\bcalculate\s+\w+",  # "calculate the..."
            r"\bcompute\s+\w+",
            r"\bsolve\s+(for|the|this)",
            r"\bequation\b",
            r"\bformula\b",
            r"\bderivative\b",
            r"\bintegral\b",
            r"\bsum\s+of\s+\d+",  # "sum of 5 and 3"
            r"\bsquare\s+root\b",
            r"\btimes\s+\d+",  # "25 times 4"
            r"\bmultipl(y|ied)\s+\d+",
            r"\bdivide\s+\d+",
        ]
        if any(re.search(pattern, query_lower) for pattern in math_patterns):
            return QueryCategory.MATHEMATICAL

        # PRIORITY 2: Comparison queries - explicit comparison patterns
        # Check early to avoid misclassification as OPINION/CODE
        comparison_patterns = [
            r"\bvs\.?\b",  # "X vs Y"
            r"\bversus\b",
            r"\bdifference\s+between\b",
            r"\bdifferences?\s+between\b",
            r"\bcompare\s+\w+\s+(and|with|to)\b",
            r"\bbetter\s+than\b",
            r"\bworse\s+than\b",
            r"\bsimilar(ities)?\s+(to|with)\b",
            r"\bwhich\s+is\s+(better|worse|more|less)\b",
            r"\b(differ|differs?)\s+(from|between)\b",  # "differ from", "differs from"
        ]
        if any(re.search(pattern, query_lower) for pattern in comparison_patterns):
            return QueryCategory.COMPARISON

        # PRIORITY 3: SUBJECTIVE queries (merged OPINION + ADVICE)
        # Combines opinion-seeking and advice-seeking patterns
        subjective_patterns = [
            # Direct opinion/preference keywords
            r"\bopinion\b",
            r"\bdo\s+you\s+think\b",  # "Do you think..."
            r"\bthink\s+(about|of|that)\b",
            r"\bfeel\s+about\b",
            r"\bthoughts\s+on\b",
            r"\bfavorite\b",
            r"\bprefer\b",
            
            # Recommendations
            r"\brecommend\b",
            r"\bsuggestion\b",
            r"\badvice\b",
            
            # "Should I" with specific context
            r"\bshould\s+i\s+(buy|get|choose|learn|start|try|go|do|use)\b",
            r"\bis\s+it\s+worth\b",
            r"\bworth\s+(it|buying|getting|learning)\b",
            
            # Comparisons (subjective evaluation)
            r"\bwhich\s+is\s+better\b",
            r"\bis\s+\w+\s+better\s+than\b",
            r"\bis\s+it\s+(better|worse|good|bad)\b",
            
            # "Best" with specific context
            r"\bwhat\s+is\s+the\s+best\s+(way\s+to|method|approach|option)\b",
            r"\bbest\s+way\s+to\s+(learn|study|prepare|improve|start)\b",
            
            # "How do/can/should I" with personal/learning context
            r"\bhow\s+(do|can|should)\s+i\s+(start|learn|improve|prepare|become|get)\b",
            r"\bwhat\s+should\s+i\s+(do|learn|study|practice|use)\b",
            
            # Guidance/help seeking
            r"\bguide\s+(me|for|to)\b",
            r"\bhelp\s+me\s+(decide|choose|learn|understand)\b",
            r"\btips\s+(for|to)\b",
            r"\bsteps\s+to\s+(become|learn|start|get)\b",
            r"\bways\s+to\s+(improve|learn|start)\b",
            
            # "How to" standalone (generic but common advice pattern)
            r"\bhow\s+to\s+",
            
            # Specific advice-seeking modals
            r"\bcan\s+you\s+(help|show|tell|explain|suggest|recommend)\b",
            r"\bcould\s+you\s+(help|show|tell|explain)\b",
        ]
        if any(re.search(pattern, query_lower) for pattern in subjective_patterns):
            return QueryCategory.SUBJECTIVE

        # PRIORITY 4: Code-related queries - specific programming implementation
        # Checked AFTER comparison/subjective to avoid false positives
        # Only match when clearly about code implementation, not learning/comparison
        code_keywords = [
            "function", "method", "class", "variable", "loop", "array",
            "algorithm", "syntax", "compile", "debug", "error message",
            "exception", "implement", "code snippet", "api call", "library",
        ]
        # Require at least 2 technical code keywords
        code_count = sum(1 for kw in code_keywords if kw in query_lower)
        # Or very specific code-related phrases
        specific_code = any(phrase in query_lower for phrase in [
            "python code", "javascript code", "write a function",
            "implement a", "code snippet", "api call", "error message",
            "syntax error", "compile error",
        ])
        if code_count >= 2 or specific_code:
            return QueryCategory.CODE

        # PRIORITY 5: Factual questions - pure information seeking
        # More lenient patterns to catch common factual questions
        fact_patterns = [
            # "What is/are" (including contractions) - allow at word boundary, not just start
            r"\bwhat'?s\s+(?!(the\s+)?(best|better|worst|worse))",  # "What's X?" or "What is X?"
            r"\bwhat\s+(is|are|was|were)\s+(?!(the\s+)?(best|better|worst|worse))",
            r"\bwhat\s+(if|about)\s+",  # "What if...", "What about..."
            r"\bwhat\s+causes\b",  # "What causes stool color to change?"
            r"\bwhat\s+type\s+",  # "What type of government..."
            # "Why" questions - explanations and reasoning
            r"\bwhy\s+(does|do|did|is|are|was|were|would|should|can|cannot)\s+",
            r"\bwhy\s+not\b",
            # Other question words
            r"\bwho'?s?\s+(is|are|was|were)\s+",  # "Who's..." or "Who is..."
            r"\bwho\s+(is|are|was|were|invented|discovered)\s+",
            r"\bwhen\s+(did|does|is|was|were|will)\s+",  # "When did..."
            r"\bwhere\s+(is|are|can|could|did|do)\s+",  # "Where is..."
            r"\bwhich\s+(country|city|place|year|one)\s+",  # Geographic/temporal facts
            # Yes/No questions seeking factual information
            r"\bdoes\s+\w+\s+(have|offer|provide|support|work|mean|exist|make\s+sense)\b",
            r"\bdo\s+(they|people|we|you)\s+(have|offer|provide|know)\b",
            r"\bdid\s+\w+\s+(happen|exist|work|succeed)\b",
            r"\bhas\s+(anyone|someone|it|this)\s+",
            r"\bhave\s+(you|they|people)\s+(seen|tried|heard)\b",
            r"\bis\s+there\s+(a|an|any)\s+",  # "Is there a way..."
            r"\bis\s+it\s+possible\s+to\b",  # "Is it possible to..."
            r"\bare\s+there\s+(any|some|many)\s+",
            r"\bare\s+(you|they)\s+able\s+to\b",  # "Are you able to..."
            r"\bwas\s+\w+\s+(always|originally|initially)\b",
            r"\bwere\s+(they|these|those)\s+",
            r"\bwill\s+\w+\s+(happen|work|be|exist)\b",
            r"\bwould\s+\w+\s+(work|be|happen)\b",
            # Definition patterns
            r"\bdefine\s+\w+",  # "define X"
            r"\bdefinition\s+of\b",
            r"\bwhat\s+does\s+\w+\s+mean\b",  # "What does X mean?"
            r"\bwhat\s+is\s+the\s+(capital|population|currency|meaning|purpose)\b",
        ]
        if any(re.search(pattern, query_lower) for pattern in fact_patterns):
            return QueryCategory.FACTUAL

        # PRIORITY 7: Creative tasks - writing, generation, design
        creative_keywords = [
            "write a story", "write a poem", "create a", "generate a",
            "design a", "compose a", "imagine", "invent a",
        ]
        if any(keyword in query_lower for keyword in creative_keywords):
            return QueryCategory.CREATIVE

        # All other queries fall into UNKNOWN
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

        # Debug logging
        logger.debug(
            f"Query: '{query[:60]}...' | Category: {category.value} | "
            f"Threshold: {adaptive_threshold:.3f} | Similarity: {base_score:.3f} | "
            f"Match: {base_score >= adaptive_threshold}"
        )

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
        threshold = cache_config.get("similarity_threshold", 0.80)

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
        return 0.80

    def get_stats(self) -> Dict[str, Any]:
        """Return query statistics."""
        if self.evaluator:
            return self.evaluator.get_query_stats()
        return {"total_queries": 0}

    def __repr__(self) -> str:
        return "AdaptiveThresholdCache(adaptive_thresholds=True)"
