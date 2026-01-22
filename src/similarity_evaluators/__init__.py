"""
Similarity evaluators package.

Contains different threshold strategies for cache evaluation.
"""

from .base import BaseSimilarityEvaluation, BaseCache
from .fixed_threshold import FixedThresholdCache
from .length_based import LengthBasedSimilarityEvaluation, LengthBasedCache
from .density_based import DensityBasedSimilarityEvaluation, DensityBasedCache
from .score_gap import ScoreGapSimilarityEvaluation, ScoreGapCache
from .adaptive_threshold import AdaptiveSimilarityEvaluation, AdaptiveThresholdCache

__all__ = [
    "BaseSimilarityEvaluation",
    "BaseCache",
    "FixedThresholdCache",
    "LengthBasedSimilarityEvaluation",
    "LengthBasedCache",
    "DensityBasedSimilarityEvaluation",
    "DensityBasedCache",
    "ScoreGapSimilarityEvaluation",
    "ScoreGapCache",
    "AdaptiveSimilarityEvaluation",
    "AdaptiveThresholdCache",
]
