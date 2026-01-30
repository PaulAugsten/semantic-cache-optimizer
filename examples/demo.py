#!/usr/bin/env python3
"""
Demo: Adaptive Threshold System.

Shows category-based threshold adaptation with example query pairs.

Usage:
    poetry run python examples/demo.py
"""
import sys
from pathlib import Path

import yaml
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from gptcache.embedding import Huggingface
from src.adaptive_threshold import AdaptiveSimilarityEvaluation


def load_config(path: str = "config.yaml"):
    """Load configuration."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def show_category_examples():
    """Show query category classification."""
    config = load_config()
    evaluator = AdaptiveSimilarityEvaluation(config)
    
    print("\n" + "=" * 108)
    print("QUERY CATEGORY CLASSIFICATION")
    print("=" * 108)
    
    examples = [
        "What is the capital of France?",
        "How do I learn Python programming?",
        "What do you think about climate change?",
        "What are the differences between Python and Java?",
        "What is 25 times 4?",
        "Write a poem about coding",
        "How to implement binary search in Python?",
        "Hello there!",
    ]
    
    print("\nClassifying queries:")
    print("-" * 108)
    
    for query in examples:
        category = evaluator._classify_query(query)
        threshold = evaluator._calculate_adaptive_threshold(query, category)
        print(f"  {category.name:15s} (threshold: {threshold:.3f}) | {query}")
    
    print("\n" + "=" * 108)


def demo_adaptive_threshold():
    """Demonstrate adaptive threshold with query pairs."""
    config = load_config()
    
    test_pairs = [
        ("What is the capital of France?", "What's France's capital city?", True),
        ("What is the capital of France?", "What is the population of Germany?", False),
        ("How do I learn Python?", "Best way to learn Python programming?", True),
        ("How do I learn Python?", "What is machine learning?", False),
        ("What do you think about AI?", "Do you believe AI is useful?", True),
        ("What do you think about AI?", "What is artificial intelligence?", False),
        ("Python vs Java differences?", "How do Python and Java differ?", True),
        ("Python vs Java differences?", "What is the best language?", False),
        ("What is 15 * 7?", "Calculate 15 times 7", True),
        ("What is 15 * 7?", "What is 20 + 5?", False),
    ]

    print("\nLoading model...")
    evaluator = AdaptiveSimilarityEvaluation(config)
    embedding_model = Huggingface(config["embedding"]["model"])

    print("\n" + "=" * 108)
    print("ADAPTIVE THRESHOLD EVALUATION")
    print("=" * 108)

    for q1, q2, expected_similar in test_pairs:
        category1 = evaluator._classify_query(q1)
        category2 = evaluator._classify_query(q2)
        
        emb1 = np.array(embedding_model.to_embeddings(q1))
        emb2 = np.array(embedding_model.to_embeddings(q2))
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        similarity = float(np.dot(emb1, emb2))
        threshold = evaluator._calculate_adaptive_threshold(q1, category1)
        is_similar = similarity >= threshold
        match_status = "✓" if is_similar == expected_similar else "✗"
        
        print(f"\n{match_status} Q1: {q1}")
        print(f"  Q2: {q2}")
        print("-" * 108)
        print(f"  Category Q1: {category1.name:15s} | Category Q2: {category2.name:15s}")
        print(f"  Similarity:  {similarity:.4f}")
        print(f"  Threshold:   {threshold:.4f} (adaptive for {category1.name})")
        print(f"  Decision:    {'CACHE HIT ✓' if is_similar else 'CACHE MISS ✗':15s} | Expected: {'SIMILAR' if expected_similar else 'DIFFERENT'}")

    print("\n" + "=" * 108)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("\nADAPTIVE THRESHOLD SEMANTIC CACHING DEMO")

    try:
        show_category_examples()
        demo_adaptive_threshold()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        import traceback
        traceback.print_exc()
        sys.exit(1)
