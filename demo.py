#!/usr/bin/env python3
"""
Demo script showing the adaptive threshold system in action.

This script demonstrates:
1. Adaptive threshold evaluation per category
2. Category classification
3. Threshold adjustments based on query characteristics
"""
import sys
from pathlib import Path

import yaml
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from gptcache.embedding import Huggingface
from src.similarity_evaluators.adaptive_threshold import (
    AdaptiveSimilarityEvaluation,
    QueryCategory,
)


def load_config(path: str = "config.yaml"):
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def demo_adaptive_threshold():
    """Demonstrate adaptive threshold evaluation."""
    config = load_config()

    # Example query pairs with different categories
    test_pairs = [
        # FACTUAL queries
        ("What is the capital of France?", "What's France's capital city?", True),
        ("What is the capital of France?", "What is the population of Germany?", False),
        
        # ADVICE queries
        ("How do I learn Python?", "What's the best way to learn Python programming?", True),
        ("How do I learn Python?", "What is machine learning?", False),
        
        # OPINION queries
        ("What do you think about climate change?", "Do you think climate change is real?", True),
        ("What do you think about climate change?", "What is climate change?", False),
        
        # COMPARISON queries
        ("What are the differences between Python and Java?", "How do Python and Java differ?", True),
        ("What are the differences between Python and Java?", "What is the best programming language?", False),
        
        # MATHEMATICAL queries
        ("What is 15 * 7?", "Calculate 15 times 7", True),
        ("What is 15 * 7?", "What is 20 + 5?", False),
    ]

    print("\nLoading embedding model and initializing adaptive evaluator...")
    
    # Create adaptive evaluator
    evaluator = AdaptiveSimilarityEvaluation(config)
    
    # Get embedding model for computing similarities
    embedding_model = Huggingface(config["embedding"]["model"])

    print("\n" + "=" * 100)
    print("ADAPTIVE THRESHOLD DEMO - CATEGORY-SPECIFIC EVALUATION")
    print("=" * 100)

    for q1, q2, expected_similar in test_pairs:
        # Classify queries
        category1 = evaluator._classify_query(q1)
        category2 = evaluator._classify_query(q2)
        
        # Compute embeddings
        emb1 = np.array(embedding_model.to_embeddings(q1))
        emb2 = np.array(embedding_model.to_embeddings(q2))
        
        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Compute similarity
        similarity = float(np.dot(emb1, emb2))
        
        # Get adaptive threshold for q1
        threshold = evaluator._calculate_adaptive_threshold(q1, category1)
        
        # Determine if it would be a cache hit
        is_similar = similarity >= threshold
        match_status = "✓" if is_similar == expected_similar else "✗"
        
        print(f"\n{match_status} Q1: {q1}")
        print(f"  Q2: {q2}")
        print("-" * 100)
        print(f"  Category Q1: {category1.name:15s} | Category Q2: {category2.name:15s}")
        print(f"  Similarity:  {similarity:.4f}")
        print(f"  Threshold:   {threshold:.4f} (adaptive for {category1.name})")
        print(f"  Decision:    {'CACHE HIT ✓' if is_similar else 'CACHE MISS ✗':15s} | Expected: {'SIMILAR' if expected_similar else 'DIFFERENT'}")
        
        # Show threshold info
        if hasattr(evaluator, 'threshold_rules') and category1 in evaluator.threshold_rules:
            rule = evaluator.threshold_rules[category1]
            print(f"  Threshold Rule: base={rule.base_threshold:.3f}, length_adj={rule.length_adjustment:.3f}")

    print("\n" + "=" * 100)


def show_category_examples():
    """Show examples of different query categories."""
    config = load_config()
    evaluator = AdaptiveSimilarityEvaluation(config)
    
    print("\n" + "=" * 100)
    print("QUERY CATEGORY CLASSIFICATION DEMO")
    print("=" * 100)
    
    examples = [
        "What is the capital of France?",                                    # FACTUAL
        "How do I learn Python programming?",                               # ADVICE
        "What do you think about climate change?",                          # OPINION
        "What are the differences between supervised and unsupervised ML?", # COMPARISON
        "What is 25 times 4?",                                              # MATHEMATICAL
        "Hello there!",                                                     # CONVERSATIONAL
        "Write a poem about coding",                                        # CREATIVE
        "How to implement a binary search in Python?",                     # CODE
        "This is a test query",                                             # UNKNOWN
    ]
    
    print("\nClassifying various query types:")
    print("-" * 100)
    
    for query in examples:
        category = evaluator._classify_query(query)
        threshold = evaluator._calculate_adaptive_threshold(query, category)
        
        print(f"  {category.name:15s} (threshold: {threshold:.3f}) | {query}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # Suppress info logs for cleaner demo

    print("\nADAPTIVE THRESHOLD SEMANTIC CACHING DEMO")
    print("This demonstrates category-based adaptive threshold evaluation.\n")

    try:
        show_category_examples()
        demo_adaptive_threshold()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
