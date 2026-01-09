"""
Self-Adaptive Similarity Threshold - Idee 2

Der Similarity Threshold passt sich dynamisch an basierend auf:
- Query-Länge
- Query-Typ (semantische Kategorie wie Fakten, Mathe, Konversation)
- Historische Performance
"""

from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import re
from gptcache import Cache
from gptcache.manager import get_data_manager, VectorBase
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.processor.pre import last_content
from gptcache.utils.cache_func import cache_all


class QueryCategory(Enum):
    """Semantische Query-Kategorien."""
    FACTUAL = "factual"          # Faktenfragen
    MATHEMATICAL = "mathematical" # Mathematik/Berechnungen
    CONVERSATIONAL = "conversational"  # Konversation
    CREATIVE = "creative"        # Kreative Aufgaben
    CODE = "code"                # Code-bezogen
    UNKNOWN = "unknown"


@dataclass
class ThresholdRule:
    """Regel für Threshold-Anpassung."""
    base_threshold: float
    length_adjustment: float  # Anpassung pro 10 Zeichen
    min_threshold: float = 0.5
    max_threshold: float = 0.95


class AdaptiveSimilarityEvaluation(SimilarityEvaluation):
    """
    Similarity Evaluation mit adaptivem Threshold.
    Passt den Threshold basierend auf Query-Eigenschaften an.
    """
    
    def __init__(
        self,
        base_evaluation: Optional[SimilarityEvaluation] = None,
        threshold_rules: Optional[Dict[QueryCategory, ThresholdRule]] = None,
    ):
        """
        Args:
            base_evaluation: Basis-Similarity-Evaluator (default: SearchDistance)
            threshold_rules: Dict mit Regeln pro Kategorie
        """
        from gptcache.similarity_evaluation import SearchDistanceEvaluation
        
        self.base_evaluation = base_evaluation or SearchDistanceEvaluation()
        
        # Default Regeln
        if threshold_rules is None:
            threshold_rules = {
                QueryCategory.FACTUAL: ThresholdRule(
                    base_threshold=0.85,
                    length_adjustment=-0.01,  # Weniger strikt bei längeren Fragen
                ),
                QueryCategory.MATHEMATICAL: ThresholdRule(
                    base_threshold=0.95,  # Sehr strikt bei Mathe
                    length_adjustment=0.0,  # Keine Längen-Anpassung
                ),
                QueryCategory.CONVERSATIONAL: ThresholdRule(
                    base_threshold=0.7,  # Weniger strikt bei Konversation
                    length_adjustment=-0.015,
                ),
                QueryCategory.CREATIVE: ThresholdRule(
                    base_threshold=0.6,  # Am wenigsten strikt
                    length_adjustment=-0.02,
                ),
                QueryCategory.CODE: ThresholdRule(
                    base_threshold=0.9,  # Strikt bei Code
                    length_adjustment=-0.005,
                ),
                QueryCategory.UNKNOWN: ThresholdRule(
                    base_threshold=0.8,
                    length_adjustment=-0.01,
                ),
            }
        
        self.threshold_rules = threshold_rules
        self._query_history: Dict[str, Dict[str, Any]] = {}
    
    def _classify_query(self, query: str) -> QueryCategory:
        """
        Klassifiziert eine Query in eine semantische Kategorie.
        
        Args:
            query: Query-String
            
        Returns:
            QueryCategory
        """
        query_lower = query.lower()
        
        # Mathematische Queries
        math_patterns = [r'\d+\s*[\+\-\*/]\s*\d+', r'calculate', r'compute', 
                        r'solve', r'equation', r'sum', r'multiply']
        if any(re.search(pattern, query_lower) for pattern in math_patterns):
            return QueryCategory.MATHEMATICAL
        
        # Code-bezogene Queries
        code_keywords = ['function', 'class', 'code', 'python', 'javascript', 
                        'programming', 'implement', 'algorithm', 'debug']
        if any(keyword in query_lower for keyword in code_keywords):
            return QueryCategory.CODE
        
        # Faktenfragen
        fact_patterns = [r'^(what|who|when|where|which)\s', r'definition', 
                        r'explain', r'describe']
        if any(re.search(pattern, query_lower) for pattern in fact_patterns):
            return QueryCategory.FACTUAL
        
        # Kreative Aufgaben
        creative_keywords = ['write', 'create', 'generate', 'story', 'poem', 
                           'imagine', 'design']
        if any(keyword in query_lower for keyword in creative_keywords):
            return QueryCategory.CREATIVE
        
        # Konversation (default für viele kurze, informelle Anfragen)
        conversational_patterns = [r'^(hi|hello|hey)', r'how are you', 
                                  r'thank', r'please', r'can you']
        if any(re.search(pattern, query_lower) for pattern in conversational_patterns):
            return QueryCategory.CONVERSATIONAL
        
        return QueryCategory.UNKNOWN
    
    def _calculate_adaptive_threshold(self, query: str, category: QueryCategory) -> float:
        """
        Berechnet den adaptiven Threshold für eine Query.
        
        Args:
            query: Query-String
            category: Query-Kategorie
            
        Returns:
            Adaptiver Threshold-Wert
        """
        rule = self.threshold_rules[category]
        threshold = rule.base_threshold
        
        # Längen-Anpassung (pro 10 Zeichen)
        length_factor = len(query) // 10
        threshold += length_factor * rule.length_adjustment
        
        # Clamp auf Min/Max
        threshold = max(rule.min_threshold, min(rule.max_threshold, threshold))
        
        return threshold
    
    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        Evaluiert Similarity mit adaptivem Threshold.
        
        Args:
            src_dict: Source request params
            cache_dict: Cached request params
            
        Returns:
            Similarity score
        """
        # Basis-Similarity berechnen
        base_score = self.base_evaluation.evaluation(src_dict, cache_dict, **kwargs)
        query = src_dict.get('question', src_dict.get('query', ''))
        
        # Kategorie bestimmen und adaptiven Threshold berechnen
        category = self._classify_query(query)
        adaptive_threshold = self._calculate_adaptive_threshold(query, category)
        
        # History tracken
        self._query_history[query] = {
            'category': category.value,
            'threshold': adaptive_threshold,
            'base_score': base_score,
        }
        
        return base_score
    
    def range(self):
        """Range des Similarity Scores."""
        return self.base_evaluation.range()
    
    def get_threshold_for_query(self, query: str) -> float:
        """
        Public Method um den Threshold für eine Query zu bekommen.
        
        Args:
            query: Query-String
            
        Returns:
            Adaptiver Threshold
        """
        category = self._classify_query(query)
        return self._calculate_adaptive_threshold(query, category)
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über verarbeitete Queries zurück."""
        if not self._query_history:
            return {"total_queries": 0}
        
        categories = {}
        for query_data in self._query_history.values():
            cat = query_data['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_queries": len(self._query_history),
            "categories": categories,
        }


class AdaptiveThresholdCache:
    """Cache mit adaptivem Similarity Threshold."""
    
    def __init__(
        self,
        threshold_rules: Optional[Dict[QueryCategory, ThresholdRule]] = None,
        embedding_func: Optional[Callable] = None,
    ):
        """
        Args:
            threshold_rules: Custom Threshold-Regeln
            embedding_func: Embedding-Funktion
        """
        self.cache = Cache()
        
        # Setup Embedding
        if embedding_func is None:
            onnx = Onnx()
            embedding_func = onnx.to_embeddings
            dimension = onnx.dimension
        else:
            dimension = 768
        
        # Adaptive Similarity Evaluation
        self.similarity_eval = AdaptiveSimilarityEvaluation(
            threshold_rules=threshold_rules
        )
        
        # Setup data manager
        vector_base = VectorBase('faiss', dimension=dimension)
        data_manager = get_data_manager('sqlite', vector_base)
        
        # Initialize cache
        self.cache.init(
            cache_enable_func=cache_all,
            pre_embedding_func=last_content,
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=self.similarity_eval,
        )
    
    def get_cache(self) -> Cache:
        return self.cache
    
    def get_threshold_for_query(self, query: str) -> float:
        return self.similarity_eval.get_threshold_for_query(query)
    
    def get_stats(self) -> Dict[str, Any]:
        return self.similarity_eval.get_query_stats()
    
    def __repr__(self) -> str:
        return "AdaptiveThresholdCache(adaptive_thresholds=True)"


if __name__ == "__main__":
    # Beispiel-Usage
    cache = AdaptiveThresholdCache()
    print(f"Initialized: {cache}")
    
    # Test verschiedene Query-Typen
    test_queries = [
        "What is 2+2?",
        "Write me a poem about the ocean",
        "How do I implement a binary search in Python?",
        "What is the capital of France?",
        "Hello, how are you today?",
    ]
    
    print("\nAdaptive Thresholds:")
    for query in test_queries:
        threshold = cache.get_threshold_for_query(query)
        print(f"  '{query[:40]}...' -> Threshold: {threshold:.3f}")
