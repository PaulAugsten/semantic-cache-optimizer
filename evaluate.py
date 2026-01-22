"""
Evaluation script for semantic caching strategies.

Compares different similarity evaluation strategies using the QQP dataset.
"""

import os
import time
import shutil
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

import yaml
import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from gptcache import cache

from src.cache_manager import CacheManager


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    strategy: str
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_queries: int = 0
    cache_hit_latencies: List[float] = field(default_factory=list)
    cache_miss_latencies: List[float] = field(default_factory=list)
    avg_tokens_per_call: int = 50

    @property
    def precision(self) -> float:
        """Percentage of cache hits that are correct."""
        if self.cache_hits == 0:
            return 0.0
        return self.true_positives / self.cache_hits

    @property
    def recall(self) -> float:
        """Percentage of actual duplicates found."""
        total_positives = self.true_positives + self.false_negatives
        if total_positives == 0:
            return 0.0
        return self.true_positives / total_positives

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def false_positive_rate(self) -> float:
        """How often system incorrectly matches different queries."""
        total_negatives = self.false_positives + self.true_negatives
        if total_negatives == 0:
            return 0.0
        return self.false_positives / total_negatives

    @property
    def cache_hit_ratio(self) -> float:
        """Percentage of requests served from cache."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def avg_cache_hit_latency(self) -> float:
        """Average latency for cache hits (ms)."""
        if not self.cache_hit_latencies:
            return 0.0
        return np.mean(self.cache_hit_latencies)

    @property
    def avg_cache_miss_latency(self) -> float:
        """Average latency for cache misses (ms)."""
        if not self.cache_miss_latencies:
            return 0.0
        return np.mean(self.cache_miss_latencies)

    @property
    def token_savings(self) -> int:
        """Estimated tokens saved by cache hits."""
        return self.cache_hits * self.avg_tokens_per_call

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "strategy": self.strategy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "false_positive_rate": self.false_positive_rate,
            "cache_hit_ratio": self.cache_hit_ratio,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "avg_cache_hit_latency_ms": self.avg_cache_hit_latency,
            "avg_cache_miss_latency_ms": self.avg_cache_miss_latency,
            "std_cache_hit_latency_ms": (
                np.std(self.cache_hit_latencies) if self.cache_hit_latencies else 0.0
            ),
            "std_cache_miss_latency_ms": (
                np.std(self.cache_miss_latencies) if self.cache_miss_latencies else 0.0
            ),
            "token_savings": self.token_savings,
            "total_queries": self.total_queries,
        }


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_qqp_dataset(max_samples: int = 10000, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load QQP dataset for evaluation.

    Args:
        max_samples: Maximum number of samples to use.
        seed: Random seed for deterministic sampling.

    Returns:
        List of question pair dictionaries.
    """
    logger.info("Loading QQP dataset...")
    dataset = load_dataset("nyu-mll/glue", "qqp")["train"]

    if len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    pairs = []
    for item in dataset:
        pairs.append(
            {
                "idx": item["idx"],
                "question1": item["question1"],
                "question2": item["question2"],
                "label": item["label"],
            }
        )

    logger.info(f"Loaded {len(pairs)} question pairs")
    return pairs


def is_cache_hit(response: str) -> bool:
    """Check if response is a cache hit (numeric ID)."""
    try:
        int(str(response).strip())
        return True
    except (ValueError, TypeError):
        return False


def reset_cache() -> None:
    """Reset gptcache singleton between strategy evaluations."""
    # Reset the cache singleton state
    cache.cache_enable_func = None
    cache.pre_embedding_func = None
    cache.embedding_func = None
    cache.data_manager = None
    cache.similarity_evaluation = None
    cache.post_process_messages_func = None
    cache.config = None
    cache.next_cache = None

    # Clean up default cache files
    for f in ["sqlite.db", "faiss.index"]:
        if os.path.exists(f):
            os.remove(f)


def evaluate_strategy(
    config: Dict[str, Any],
    strategy: str,
    pairs: List[Dict[str, Any]],
) -> EvaluationResult:
    """
    Evaluate a single caching strategy.

    Args:
        config: Configuration dictionary.
        strategy: Strategy name.
        pairs: List of question pairs.

    Returns:
        EvaluationResult with computed metrics.
    """
    logger.info(f"Evaluating strategy: {strategy}")
    result = EvaluationResult(strategy=strategy)

    # Reset gptcache singleton for clean evaluation
    reset_cache()

    # Initialize cache manager
    manager = CacheManager(config, strategy=strategy)
    manager.initialize()

    # Import first questions into cache
    questions = [pair["question1"] for pair in pairs]
    answers = [str(pair["idx"]) for pair in pairs]

    logger.info(f"Importing {len(questions)} questions into cache...")
    cache.import_data(questions=questions, answers=answers)

    # Query with second questions
    logger.info("Querying cache with second questions...")
    for pair in tqdm(pairs, desc=f"Evaluating {strategy}"):
        question2 = pair["question2"]
        expected_id = str(pair["idx"])
        label = pair["label"]

        response, metadata = manager.query(question2)
        latency = metadata.get("latency_ms", 0)
        result.total_queries += 1

        response_str = str(response).strip()
        hit = is_cache_hit(response_str)

        if hit:
            result.cache_hits += 1
            result.cache_hit_latencies.append(latency)

            if response_str == expected_id:
                # Cache hit with matching ID
                if label == 1:
                    result.true_positives += 1
                else:
                    result.false_positives += 1
            else:
                # Cache hit with wrong ID - always a false positive
                # try:
                got_idx = int(response_str)
                row = pairs.filter(lambda x: x["idx"] == got_idx)
                print(
                    "Cache hit mismatch:",
                    question2,
                    "expected", expected_id,
                    "got_idx", got_idx,
                    "question_got1", row["question1"],
                    "question_got2", row["question2"],
                )
                # except Exception as e:
                #     print("Cache hit mismatch:", question2, "expected", expected_id, "got_idx", got_idx, "error", e)
                result.false_positives += 1
        else:
            result.cache_misses += 1
            result.cache_miss_latencies.append(latency)

            if label == 1:
                result.false_negatives += 1
            else:
                result.true_negatives += 1

    return result


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for a list of values."""
    if not values or len(values) < 2:
        return (0.0, 0.0)

    from scipy import stats

    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def print_results(results: List[EvaluationResult]) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 120)
    print("EVALUATION RESULTS")
    print("=" * 120)

    headers = [
        "Strategy",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "Hit Ratio",
        "Hits",
        "Misses",
        "Tokens Saved",
    ]
    print(
        f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} "
        f"{headers[4]:<10} {headers[5]:<10} {headers[6]:<8} {headers[7]:<8} {headers[8]:<12}"
    )
    print("-" * 120)

    for r in results:
        print(
            f"{r.strategy:<15} {r.precision:.4f}     {r.recall:.4f}     "
            f"{r.f1_score:.4f}     {r.false_positive_rate:.4f}     "
            f"{r.cache_hit_ratio:.4f}     {r.cache_hits:<8} {r.cache_misses:<8} "
            f"{r.token_savings:<12}"
        )

    print("=" * 120)

    # Detailed breakdown per strategy
    for r in results:
        print(f"\n--- {r.strategy.upper()} DETAILED METRICS ---")
        print(f"  True Positives:  {r.true_positives}")
        print(f"  False Positives: {r.false_positives}")
        print(f"  True Negatives:  {r.true_negatives}")
        print(f"  False Negatives: {r.false_negatives}")

        if r.cache_hit_latencies:
            ci_low, ci_high = compute_confidence_interval(r.cache_hit_latencies)
            print(
                f"  Cache Hit Latency:  {r.avg_cache_hit_latency:.2f} ms "
                f"(std: {np.std(r.cache_hit_latencies):.2f}, "
                f"95% CI: [{ci_low:.2f}, {ci_high:.2f}])"
            )

        if r.cache_miss_latencies:
            ci_low, ci_high = compute_confidence_interval(r.cache_miss_latencies)
            print(
                f"  Cache Miss Latency: {r.avg_cache_miss_latency:.2f} ms "
                f"(std: {np.std(r.cache_miss_latencies):.2f}, "
                f"95% CI: [{ci_low:.2f}, {ci_high:.2f}])"
            )


def main() -> None:
    """Run evaluation for all strategies."""
    config = load_config()

    seed = config.get("seed", 42)
    max_samples = config.get("max_eval_samples", 10000)

    pairs = load_qqp_dataset(max_samples=max_samples, seed=1) # seed)

    # print(pairs)

    strategies = ["fixed", "length_based", "density_based", "score_gap", "adaptive"]
    strategies = ["fixed"]
    results = []

    for strategy in strategies[:1]:
        # TODO: später wieder aktivieren
        # try:
        result = evaluate_strategy(config, strategy, pairs)
        results.append(result)
        logger.info(f"Completed evaluation for {strategy}")
        # except Exception as e:
        #     logger.error(f"Failed to evaluate {strategy}: {e}")
        #     continue

    print_results(results)

    # Save results to file
    import json

    results_dict = [r.to_dict() for r in results]
    with open("evaluation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
