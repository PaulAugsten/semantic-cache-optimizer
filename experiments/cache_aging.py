"""
Cache Aging & Recency Decay - Idee 3

Jede gecachte Antwort verliert an Gewicht, je länger sie nicht benutzt wurde.
Dies ermöglicht zeitbasierte Eviction und bevorzugt häufig/kürzlich genutzte Einträge.
"""

from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import time
from gptcache import Cache
from gptcache.manager import get_data_manager, VectorBase
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.processor.pre import last_content
from gptcache.utils.cache_func import cache_all


@dataclass
class AgingConfig:
    """Konfiguration für Cache Aging."""
    decay_rate: float = 0.1  # Gewichtsverlust pro Zeiteinheit
    time_unit: int = 3600  # Zeiteinheit in Sekunden (default: 1 Stunde)
    min_weight: float = 0.1  # Minimales Gewicht
    max_age_seconds: Optional[int] = None  # Max Alter bevor Auto-Eviction


class AgingAwareSimilarityEvaluation(SimilarityEvaluation):
    """
    Similarity Evaluation die Cache Aging berücksichtigt.
    
    Der finale Score wird mit einem Recency-Faktor multipliziert:
    final_score = base_score * recency_weight
    
    recency_weight = max(min_weight, 1 - (age / time_unit) * decay_rate)
    """
    
    def __init__(
        self,
        base_evaluation: Optional[SimilarityEvaluation] = None,
        aging_config: Optional[AgingConfig] = None,
    ):
        """
        Args:
            base_evaluation: Basis-Similarity-Evaluator
            aging_config: Aging-Konfiguration
        """
        from gptcache.similarity_evaluation import SearchDistanceEvaluation
        
        self.base_evaluation = base_evaluation or SearchDistanceEvaluation()
        self.aging_config = aging_config or AgingConfig()
        
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
    
    def _calculate_recency_weight(self, cache_id: str, current_time: float) -> float:
        """
        Berechnet das Recency-Gewicht für einen Cache-Eintrag.
        
        Args:
            cache_id: ID des Cache-Eintrags
            current_time: Aktuelle Zeit (timestamp)
            
        Returns:
            Recency-Gewicht zwischen min_weight und 1.0
        """
        if cache_id not in self._access_times:
            return 1.0
        
        last_access = self._access_times[cache_id]
        age_seconds = current_time - last_access
        
        # Check max age
        if (self.aging_config.max_age_seconds and 
            age_seconds > self.aging_config.max_age_seconds):
            return 0.0
        
        # Berechne Gewicht basierend auf Alter
        age_units = age_seconds / self.aging_config.time_unit
        weight = 1.0 - (age_units * self.aging_config.decay_rate)
        
        # Clamp auf min_weight
        weight = max(self.aging_config.min_weight, weight)
        
        return weight
    
    def _update_access_time(self, cache_id: str, current_time: float):
        """Aktualisiert die Last-Access-Zeit für einen Eintrag."""
        self._access_times[cache_id] = current_time
        if cache_id not in self._creation_times:
            self._creation_times[cache_id] = current_time
    
    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        Evaluiert Similarity mit Aging-Awareness.
        
        Args:
            src_dict: Source request params
            cache_dict: Cached request params
            
        Returns:
            Age-adjusted similarity score
        """
        # Basis-Similarity berechnen
        base_score = self.base_evaluation.evaluation(src_dict, cache_dict, **kwargs)
        
        # Cache-ID extrahieren
        cache_id = str(cache_dict.get('id', cache_dict.get('cache_id', '')))
        
        if not cache_id:
            return base_score
        
        # Recency-Gewicht berechnen
        current_time = time.time()
        recency_weight = self._calculate_recency_weight(cache_id, current_time)
        
        # Access-Zeit aktualisieren (wird als Hit gezählt)
        self._update_access_time(cache_id, current_time)
        
        # Finaler Score = base_score * recency_weight
        final_score = base_score * recency_weight
        
        return final_score
    
    def range(self):
        """Range des Similarity Scores."""
        return self.base_evaluation.range()
    
    def get_aging_stats(self) -> Dict[str, Any]:
        """Statistiken über Cache Aging."""
        if not self._access_times:
            return {"total_entries": 0}
        
        current_time = time.time()
        ages = []
        weights = []
        
        for cache_id, last_access in self._access_times.items():
            age = current_time - last_access
            weight = self._calculate_recency_weight(cache_id, current_time)
            ages.append(age)
            weights.append(weight)
        
        return {
            "total_entries": len(self._access_times),
            "avg_age_seconds": sum(ages) / len(ages) if ages else 0,
            "avg_weight": sum(weights) / len(weights) if weights else 0,
            "min_weight": min(weights) if weights else 0,
            "max_age_seconds": max(ages) if ages else 0,
        }
    
    def get_old_entries(self, threshold_weight: float = 0.3) -> list:
        """
        Gibt Cache-IDs zurück, die unter einem Gewichts-Threshold liegen.
        
        Args:
            threshold_weight: Gewichts-Threshold
            
        Returns:
            Liste von Cache-IDs
        """
        current_time = time.time()
        old_entries = []
        
        for cache_id in self._access_times:
            weight = self._calculate_recency_weight(cache_id, current_time)
            if weight < threshold_weight:
                old_entries.append(cache_id)
        
        return old_entries


class CacheAgingCache:
    """Cache mit Recency Decay / Aging."""
    
    def __init__(
        self,
        aging_config: Optional[AgingConfig] = None,
        embedding_func: Optional[Callable] = None,
    ):
        """
        Args:
            aging_config: Aging-Konfiguration
            embedding_func: Embedding-Funktion
        """
        self.cache = Cache()
        
        if embedding_func is None:
            onnx = Onnx()
            embedding_func = onnx.to_embeddings
            dimension = onnx.dimension
        else:
            dimension = 768
        
        self.similarity_eval = AgingAwareSimilarityEvaluation(
            aging_config=aging_config
        )
        
        vector_base = VectorBase('faiss', dimension=dimension)
        data_manager = get_data_manager('sqlite', vector_base)
        
        self.cache.init(
            cache_enable_func=cache_all,
            pre_embedding_func=last_content,
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=self.similarity_eval,
        )
    
    def get_cache(self) -> Cache:
        return self.cache
    
    def get_aging_stats(self) -> Dict[str, Any]:
        return self.similarity_eval.get_aging_stats()
    
    def get_old_entries(self, threshold_weight: float = 0.3) -> list:
        return self.similarity_eval.get_old_entries(threshold_weight)
    
    def __repr__(self) -> str:
        config = self.similarity_eval.aging_config
        return (f"CacheAgingCache(decay_rate={config.decay_rate}, "
                f"time_unit={config.time_unit}s)")


if __name__ == "__main__":
    # Beispiel-Usage
    aging_config = AgingConfig(
        decay_rate=0.2,  # 20% Gewichtsverlust pro Zeiteinheit
        time_unit=3600,  # 1 Stunde
        min_weight=0.1,
        max_age_seconds=86400,  # 24 Stunden max
    )
    
    cache = CacheAgingCache(aging_config=aging_config)
