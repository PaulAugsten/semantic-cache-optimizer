#!/usr/bin/env python3
"""
Demo script showing the semantic cache in action.

This script demonstrates:
1. Cache initialization with different strategies
2. Making queries and observing cache hits/misses
3. Logging threshold decisions
"""
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import clean_text
from src.similarity_evaluators import (
    LengthBasedSimilarityEvaluation,
    DensityBasedSimilarityEvaluation,
    ScoreGapSimilarityEvaluation,
)
from gptcache.similarity_evaluation import SearchDistanceEvaluation


def load_config(path: str = "config.yaml"):
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def demo_similarity_evaluation():
    """Demonstrate similarity evaluation with different strategies."""
    config = load_config()

    # Example query pairs
    test_pairs = [
        # Similar questions (should match)
        ("What is the capital of France?", "What's France's capital city?"),
        ("How do I learn Python?", "What's the best way to learn Python programming?"),
        # Different questions (should not match)
        ("What is the capital of France?", "What is the population of Germany?"),
        ("How do I learn Python?", "What is machine learning?"),
        # Short ambiguous queries
        ("What is AI?", "What's AI?"),
        # Longer, more specific queries
        (
            "Can you explain the differences between supervised and unsupervised machine learning algorithms?",
            "What are the key differences between supervised learning and unsupervised learning in ML?",
        ),
    ]

    embedding_config = config.get("embedding", {})
    model = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    device = embedding_config.get("device", "cpu")

    # Create evaluators - each now contains its own embedding model
    print("\nLoading embedding models (this may take a moment)...")

    evaluators = {
        "fixed": SearchDistanceEvaluation(),
        "length_based": LengthBasedSimilarityEvaluation(config),
        "density_based": DensityBasedSimilarityEvaluation(config),
        "score_gap": ScoreGapSimilarityEvaluation(config),
    }

    threshold = config.get("cache", {}).get("similarity_threshold", 0.85)

    print("\n" + "=" * 80)
    print("SEMANTIC CACHE DEMO - SIMILARITY EVALUATION")
    print("=" * 80)

    for q1, q2 in test_pairs:
        q1_clean = clean_text(q1)
        q2_clean = clean_text(q2)

        # Use the new dict-based interface
        src_dict = {"question": q1_clean}
        cache_dict = {"question": q2_clean}

        print(f"\nQ1: {q1}")
        print(f"Q2: {q2}")
        print("-" * 60)

        for name, evaluator in evaluators.items():
            # Use new interface: evaluation(src_dict, cache_dict)
            similarity = evaluator.evaluation(src_dict, cache_dict)

            would_cache = similarity >= threshold
            status = "✓ CACHE HIT" if would_cache else "✗ CACHE MISS"

            extra_info = ""
            if hasattr(evaluator, "get_threshold_info"):
                info = evaluator.get_threshold_info()
                if "computed_threshold" in info and info["computed_threshold"]:
                    extra_info = (
                        f" (adaptive threshold: {info['computed_threshold']:.2f})"
                    )

            print(f"  {name:15s}: {similarity:.4f} -> {status}{extra_info}")

    print("\n" + "=" * 80)


def demo_cache_flow():
    """Demonstrate a simple cache flow."""
    print("\n" + "=" * 80)
    print("CACHE FLOW DEMO")
    print("=" * 80)

    # Simulated cache behavior
    cache = {}
    queries = [
        "What is machine learning?",
        "What's ML?",  # Similar to first
        "How does Python work?",  # Different
        "What is machine learning exactly?",  # Similar to first
        "Explain neural networks",  # Different
    ]

    print("\nSimulated cache interactions:")
    print("-" * 60)

    for i, query in enumerate(queries, 1):
        # In real implementation, this would use gptcache
        cache_hit = query.lower().startswith("what is machine")

        if cache_hit and cache:
            print(f"{i}. '{query}'")
            print(f"   -> CACHE HIT (returning cached response)")
        else:
            response = f"[LLM Response for: {query[:30]}...]"
            cache[query] = response
            print(f"{i}. '{query}'")
            print(f"   -> CACHE MISS (calling LLM, storing response)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("\n🚀 SEMANTIC CACHING MVP DEMO")
    print(
        "This demonstrates similarity evaluation with different threshold strategies.\n"
    )

    try:
        demo_similarity_evaluation()
        #demo_cache_flow()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.info(
            "Make sure to install dependencies: pip install -r requirements.txt"
        )
        raise
