#!/usr/bin/env python3
"""
Test script for the CacheManager.

Demonstrates actual cache initialization, filling with example questions,
and retrieval (cache hits) vs LLM calls (cache misses).

Usage:
    python test_cache_manager.py
"""
import sys
from pathlib import Path

import yaml
from loguru import logger
from gptcache import cache

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cache_manager import CacheManager


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_cache_manager():
    """
    Test the CacheManager with real cache operations.

    This test:
    1. Initializes the cache with a chosen strategy
    2. Adds several questions (cache misses -> LLM calls)
    3. Queries with similar questions (should be cache hits)
    4. Queries with different questions (should be cache misses)
    """

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("=" * 70)
    print("CACHE MANAGER TEST")
    print("=" * 70)

    # Load config
    config = load_config()

    # Use simulator mode for fast testing
    config["llm"]["mode"] = "hf"

    # Test with fixed threshold strategy
    strategy = "fixed"
    print(f"\n📦 Initializing CacheManager with '{strategy}' strategy...")

    cache_manager = CacheManager(config, strategy=strategy)
    cache_manager.initialize()

    cache.import_data(["What is the capital of France?"], ["Paris"])

    print("✅ Cache initialized successfully!\n")

    # Define test questions
    # Group 1: Questions about Python
    python_questions = [
        "How do I learn Python programming?",
        "What is the best way to learn Python?",  # Similar - should hit cache
        "How can I start learning Python coding?",  # Similar - should hit cache
    ]

    # Group 2: Questions about machine learning
    ml_questions = [
        "What is machine learning?",
        "What is machine learning?",
        "What is machine learning?",
        "Can you explain machine learning?",  # Similar - should hit cache
        "What does machine learning mean?",  # Similar - should hit cache
    ]

    # Group 3: Different questions (should all miss)
    different_questions = [
        "What is the capital of France?",
        # "How do I make a cake?",
        # "What time is it in Tokyo?",
    ]

    print("-" * 70)
    print("PHASE 1: Seeding the cache with initial questions")
    print("-" * 70)

    # Seed the cache with first questions from each group
    seed_questions = [python_questions[0], ml_questions[0]]

    for question in seed_questions:
        print(f'\n🔍 Query: "{question}"')
        response, metadata = cache_manager.query(question)

        print(f"metadata: {metadata}")
        print(f"   Latency: {metadata['latency_ms']:.1f}ms")
        print(f'   Response: "{response[:60]}..."')

    print("\n" + "-" * 70)
    print("PHASE 2: Testing similar questions (expecting cache hits)")
    print("-" * 70)

    # Test similar questions
    similar_questions = python_questions[1:] + ml_questions[1:]

    for question in similar_questions:
        print(f'\n🔍 Query: "{question}"')
        response, metadata = cache_manager.query(question)

        print(f"metadata: {metadata}")
        print(f"(Latency: {metadata['latency_ms']:.1f}ms)")
        print(f'   Response: "{response[:60]}..."')

    print("\n" + "-" * 70)
    print("PHASE 3: Testing different questions (expecting cache misses)")
    print("-" * 70)

    for question in different_questions:
        print(f'\n🔍 Query: "{question}"')
        response, metadata = cache_manager.query(question)

        print(f"metadata: {metadata}")
        print(f"   Latency: {metadata['latency_ms']:.1f}ms")
        print(f'   Response: "{response[:60]}..."')

    # # Summary
    # print("\n" + "=" * 70)
    # print("TEST SUMMARY")
    # print("=" * 70)

    # query_log = cache_manager.get_query_log()
    # total_queries = len(query_log)

    # print(f"\nTotal queries: {total_queries}")

    # print("✅ TEST COMPLETED SUCCESSFULLY")
    # print("=" * 70)


if __name__ == "__main__":
    print("\n🚀 SEMANTIC CACHING - CACHE MANAGER TEST\n")

    try:
        # Run main test
        test_cache_manager()

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure to install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\nMake sure config.yaml exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        sys.exit(1)
