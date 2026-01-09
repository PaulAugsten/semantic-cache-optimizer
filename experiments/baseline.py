"""
Baseline Implementation - Wrapper für Original GPTCache.

Dient als Vergleichsbasis für alle Experimente.
"""

from typing import Optional, Callable
from gptcache import Cache
from gptcache.manager import get_data_manager
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import SearchDistanceEvaluation
from gptcache.processor.pre import last_content
from gptcache.utils.cache_func import cache_all


class BaselineCache:
    """Wrapper für GPTCache mit Standard-Konfiguration."""
    
    def __init__(
        self,
        embedding_func: Optional[Callable] = None,
        similarity_threshold: float = 0.8,
        cache_enable_func: Callable = cache_all,
        pre_embedding_func: Callable = last_content,
    ):
        """
        Args:
            embedding_func: Embedding-Funktion (default: Onnx)
            similarity_threshold: Similarity threshold für Cache-Hits
            cache_enable_func: Funktion die entscheidet, ob Cache aktiviert wird
            pre_embedding_func: Pre-processing für Embeddings
        """
        self.cache = Cache()
        self.similarity_threshold = similarity_threshold
        
        if embedding_func is None:
            onnx = Onnx()
            embedding_func = onnx.to_embeddings
            dimension = onnx.dimension
        else:
            dimension = 768  # Default dimension
        
        from gptcache.manager import VectorBase
        vector_base = VectorBase('faiss', dimension=dimension)
        data_manager = get_data_manager('sqlite', vector_base)
        
        # Initialize cache
        self.cache.init(
            cache_enable_func=cache_enable_func,
            pre_embedding_func=pre_embedding_func,
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
        )
    
    def get_cache(self) -> Cache:
        return self.cache
    
    def __repr__(self) -> str:
        return f"BaselineCache(threshold={self.similarity_threshold})"


if __name__ == "__main__":
    cache = BaselineCache(similarity_threshold=0.8)
    print(f"Initialized: {cache}")
