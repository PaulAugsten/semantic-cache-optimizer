#!/usr/bin/env python3
"""
Test: Adaptive Threshold System.

Systematic tests for category classification, threshold calculation,
similarity computation, and cache decisions.

Usage:
    poetry run python examples/test_adaptive_threshold.py
"""
import sys
from pathlib import Path

import yaml
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from gptcache.embedding import Huggingface
from src.adaptive_threshold import AdaptiveSimilarityEvaluation


def load_config(config_path: str = "config.yaml"):
    """Load configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_category_classification():
    """Test category classification."""
    print("\n" + "=" * 108)
    print("TEST 1: CATEGORY CLASSIFICATION")
    print("=" * 108)
    
    config = load_config()
    evaluator = AdaptiveSimilarityEvaluation(config)
    
    test_queries = {
        "FACTUAL": [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When did World War 2 end?",
        ],
        "SUBJECTIVE": [
            "How do I learn Python?",
            "What's the best way to lose weight?",
            "How can I improve my writing skills?",
            "What do you think about climate change?",
            "What's your favorite programming language?",
            "Do you think AI is dangerous?",
        ],
        "COMPARISON": [
            "What are the differences between Python and Java?",
            "Compare supervised and unsupervised learning",
            "How do cats differ from dogs?",
        ],
        "MATHEMATICAL": [
            "What is 25 times 4?",
            "Calculate the square root of 144",
            "What is 10 + 5 * 2?",
        ],
        "CREATIVE": [
            "Write a poem about coding",
            "Generate a story about space",
        ],
        "CODE": [
            "How to implement binary search in Python?",
            "Write a function to reverse a string",
        ],
    }
    
    correct = 0
    total = 0
    
    for expected_category, queries in test_queries.items():
        print(f"\n{expected_category}:")
        for query in queries:
            predicted = evaluator._classify_query(query)
            match = "✓" if predicted.name == expected_category else "✗"
            print(f"  {match} {query[:60]:60s} -> {predicted.name}")
            if predicted.name == expected_category:
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy > 50


def test_threshold_calculation():
    """Test adaptive threshold calculation."""
    print("\n" + "=" * 108)
    print("TEST 2: THRESHOLD CALCULATION")
    print("=" * 108)
    
    config = load_config()
    evaluator = AdaptiveSimilarityEvaluation(config)
    
    test_cases = [
        ("What is AI?", "Short query"),
        ("How do I learn Python programming?", "Medium query"),
        ("Can you explain the differences between supervised and unsupervised ML?", "Long query"),
    ]
    
    print("\nThresholds adjust based on query length:")
    print("-" * 108)
    
    for query, description in test_cases:
        category = evaluator._classify_query(query)
        threshold = evaluator._calculate_adaptive_threshold(query, category)
        length = len(query.split())
        print(f"  {description:15s} ({length:2d} words) | Category: {category.name:15s} | Threshold: {threshold:.4f}")
    
    return True


def test_similarity_and_decisions():
    """Test similarity computation and cache decisions."""
    print("\n" + "=" * 108)
    print("TEST 3: SIMILARITY & CACHE DECISIONS")
    print("=" * 108)
    
    config = load_config()
    evaluator = AdaptiveSimilarityEvaluation(config)
    embedding_model = Huggingface(config["embedding"]["model"])
    
    test_pairs = [
        ("What is Python?", "What's Python?", True, "Near-duplicate"),
        ("How to learn Python?", "Best way to learn Python?", True, "Paraphrase"),
        ("What is Python?", "What is Java?", False, "Different topic"),
        ("Capital of France?", "Population of Germany?", False, "Different question"),
    ]
    
    print("\nTesting cache decisions:")
    print("-" * 108)
    
    correct_decisions = 0
    
    for q1, q2, should_match, description in test_pairs:
        category = evaluator._classify_query(q1)
        threshold = evaluator._calculate_adaptive_threshold(q1, category)
        
        emb1 = np.array(embedding_model.to_embeddings(q1))
        emb2 = np.array(embedding_model.to_embeddings(q2))
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        similarity = float(np.dot(emb1, emb2))
        
        is_match = similarity >= threshold
        correct = is_match == should_match
        status = "✓" if correct else "✗"
        
        if correct:
            correct_decisions += 1
        
        print(f"\n  {status} {description}")
        print(f"     Q1: {q1}")
        print(f"     Q2: {q2}")
        print(f"     Sim: {similarity:.4f} | Thr: {threshold:.4f} | Decision: {'HIT' if is_match else 'MISS'} | Expected: {'MATCH' if should_match else 'NO MATCH'}")
    
    accuracy = (correct_decisions / len(test_pairs)) * 100
    print(f"\n  Accuracy: {correct_decisions}/{len(test_pairs)} ({accuracy:.1f}%)")
    
    return accuracy >= 75


def test_threshold_overrides():
    """Test threshold override functionality."""
    print("\n" + "=" * 108)
    print("TEST 4: THRESHOLD OVERRIDES")
    print("=" * 108)
    
    config = load_config()
    
    evaluator_default = AdaptiveSimilarityEvaluation(config)
    
    custom_thresholds = {
        "FACTUAL": {"base": 0.95, "adj": 0.0},
        "SUBJECTIVE": {"base": 0.70, "adj": -0.02},
    }
    evaluator_custom = AdaptiveSimilarityEvaluation(
        config,
        threshold_overrides=custom_thresholds
    )
    
    test_query = "What is the capital of France?"
    category = evaluator_default._classify_query(test_query)
    
    default_threshold = evaluator_default._calculate_adaptive_threshold(test_query, category)
    custom_threshold = evaluator_custom._calculate_adaptive_threshold(test_query, category)
    
    print(f"\nQuery: {test_query}")
    print(f"Category: {category.name}")
    print(f"  Default Threshold: {default_threshold:.4f}")
    print(f"  Custom Threshold:  {custom_threshold:.4f}")
    
    if category.name == "FACTUAL":
        print(f"  Override Applied: {'✓' if abs(custom_threshold - 0.95) < 0.01 else '✗'}")
        return abs(custom_threshold - 0.95) < 0.01
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "=" * 108)
    print("ADAPTIVE THRESHOLD SYSTEM - TEST SUITE")
    print("=" * 108)
    
    tests = [
        ("Category Classification", test_category_classification),
        ("Threshold Calculation", test_threshold_calculation),
        ("Similarity & Decisions", test_similarity_and_decisions),
        ("Threshold Overrides", test_threshold_overrides),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 108)
    print("TEST SUMMARY")
    print("=" * 108)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status:10s} | {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    print("=" * 108)
    
    return all(p for _, p in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)