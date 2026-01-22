"""
Fixed threshold similarity evaluation.

Baseline approach using a constant similarity threshold.
Uses gptcache Huggingface embedding for embedding-based cosine similarity.
"""

from typing import Any, Dict, Tuple

import numpy as np
from loguru import logger

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.embedding import Huggingface
from gptcache import cache
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.config import Config
from gptcache.similarity_evaluation import SearchDistanceEvaluation

from gptcache.processor.pre import get_prompt


class FixedThresholdCache:
    """
    Cache implementation with fixed threshold evaluation.
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
        """Initialize the cache with fixed threshold evaluation."""
        embedding_config = self.config.get("embedding", {})
        cache_config = self.config.get("cache", {})

        model = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        threshold = cache_config.get("similarity_threshold", 0.85)

        # Create embedding model instance (shared between evaluator or distance evaluator)
        model_instance = Huggingface(model)
        embedding_dim = model_instance.dimension
        embedding_func = model_instance.to_embeddings

        self.evaluator = SearchDistanceEvaluation()

        backend = cache_config.get("backend", "faiss")

        # Create data manager
        cache_base = CacheBase("sqlite")

        if backend == "faiss":
            vector_base = VectorBase(
                "faiss",
                dimension=embedding_dim,
                top_k=cache_config.get("top_k", 5), # ???
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
        logger.info(f"Fixed threshold cache initialized with threshold={threshold}")

    def get_evaluator(self) -> SearchDistanceEvaluation:
        """Return the similarity evaluator."""
        return self.evaluator
    

