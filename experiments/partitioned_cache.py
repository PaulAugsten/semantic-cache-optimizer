"""
Dynamic Cache Partitioning - Idee 1

Cache wird in mehrere Partitionen aufgeteilt basierend auf Query-Eigenschaften:
- Short queries vs. Long queries
- Low cost vs. High cost operations
- Verschiedene Query-Typen

Jede Partition hat eigene:
- Eviction Policy
- Max Speicherkapazität  
- Similarity Threshold
"""

from typing import Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from gptcache import Cache
from gptcache.manager import get_data_manager, VectorBase
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import SearchDistanceEvaluation
from gptcache.processor.pre import last_content
from gptcache.utils.cache_func import cache_all


class PartitionType(Enum):
    """Query-Partition-Typen."""
    SHORT_QUERY = "short_query"      # < 50 Zeichen
    LONG_QUERY = "long_query"        # >= 50 Zeichen
    LOW_COST = "low_cost"            # Einfache Anfragen
    HIGH_COST = "high_cost"          # Komplexe Anfragen


@dataclass
class PartitionConfig:
    """Konfiguration für eine Cache-Partition."""
    name: str
    similarity_threshold: float
    max_size: Optional[int] = None
    eviction_policy: str = "LRU"  # LRU, LFU, FIFO
    

class PartitionedCache:
    """
    Cache mit dynamischer Partitionierung.
    
    Queries werden basierend auf ihren Eigenschaften in verschiedene
    Partitionen geroutet, die jeweils eigene Policies haben.
    """
    
    def __init__(
        self,
        partition_configs: Optional[Dict[PartitionType, PartitionConfig]] = None,
        embedding_func: Optional[Callable] = None,
    ):
        """
        Args:
            partition_configs: Dict mit Konfiguration pro Partition
            embedding_func: Embedding-Funktion (default: Onnx)
        """
        if partition_configs is None:
            partition_configs = {
                PartitionType.SHORT_QUERY: PartitionConfig(
                    name="short_query",
                    similarity_threshold=0.9,  # Höher für kurze Queries
                    max_size=1000,
                    eviction_policy="LRU"
                ),
                PartitionType.LONG_QUERY: PartitionConfig(
                    name="long_query",
                    similarity_threshold=0.7,  # Niedriger für lange Queries
                    max_size=500,
                    eviction_policy="LFU"
                ),
            }
        
        self.partition_configs = partition_configs
        self.caches: Dict[PartitionType, Cache] = {}
        
        if embedding_func is None:
            onnx = Onnx()
            embedding_func = onnx.to_embeddings
            dimension = onnx.dimension
        else:
            dimension = 768
        
        # Erstelle einen Cache pro Partition
        for partition_type, config in partition_configs.items():
            cache = Cache()
            
            vector_base = VectorBase('faiss', dimension=dimension)
            data_manager = get_data_manager('sqlite', vector_base)
            
            cache.init(
                cache_enable_func=cache_all,
                pre_embedding_func=last_content,
                embedding_func=embedding_func,
                data_manager=data_manager,
                similarity_evaluation=SearchDistanceEvaluation(),
            )
            
            self.caches[partition_type] = cache
    
    def _classify_query(self, query: str) -> PartitionType:
        """
        Klassifiziert eine Query in eine Partition.
        
        Args:
            query: Die Query-String
            
        Returns:
            Der zugehörige PartitionType
        """
        query_length = len(query)
        
        # Einfache Klassifikation basierend auf Länge
        if query_length < 50:
            return PartitionType.SHORT_QUERY
        else:
            return PartitionType.LONG_QUERY
    
    def get_cache_for_query(self, query: str) -> Tuple[Cache, PartitionConfig]:
        """
        Gibt den passenden Cache für eine Query zurück.
        
        Args:
            query: Die Query-String
            
        Returns:
            Tuple von (Cache-Objekt, PartitionConfig)
        """
        partition_type = self._classify_query(query)
        cache = self.caches.get(partition_type)
        config = self.partition_configs.get(partition_type)
        
        if cache is None or config is None:
            partition_type = list(self.caches.keys())[0]
            cache = self.caches[partition_type]
            config = self.partition_configs[partition_type]
        
        return cache, config
    
    def get_partition_stats(self) -> Dict[str, Any]:
        """Statistiken über alle Partitionen."""
        stats = {}
        for partition_type, cache in self.caches.items():
            config = self.partition_configs[partition_type]
            stats[partition_type.value] = {
                "threshold": config.similarity_threshold,
                "max_size": config.max_size,
                "eviction_policy": config.eviction_policy,
            }
        return stats
    
    def __repr__(self) -> str:
        return f"PartitionedCache(partitions={len(self.caches)})"


if __name__ == "__main__":
    # Beispiel-Usage
    cache = PartitionedCache()
    print(f"Initialized: {cache}")
    print(f"Partition stats: {cache.get_partition_stats()}")
    
    # Test Query-Klassifikation
    short_query = "What is 2+2?"
    long_query = "Can you explain in detail how quantum computing differs from classical computing?"
    
    cache_obj, config = cache.get_cache_for_query(short_query)
    print(f"Short query -> Partition: {config.name}, Threshold: {config.similarity_threshold}")
    
    cache_obj, config = cache.get_cache_for_query(long_query)
    print(f"Long query -> Partition: {config.name}, Threshold: {config.similarity_threshold}")
