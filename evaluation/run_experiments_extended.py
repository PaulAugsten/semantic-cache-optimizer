"""
Extended Experiment Tracking for Adaptive Threshold Configurations.

This script runs multiple configurations with:
- Larger sample sizes (5000-10000+)
- More configurations based on previous analysis
- Timestamped output directories
- Comprehensive result tracking
"""

import yaml
import json
import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score, average_precision_score

from gptcache.embedding import Huggingface
from src.similarity_evaluators.adaptive_threshold import (
    AdaptiveSimilarityEvaluation,
    QueryCategory,
)


@dataclass
class CategoryMetrics:
    """Metrics for a specific category."""
    
    category: str
    count: int
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_positive_rate: float
    false_negative_rate: float  # FN / (FN + TP) - missed duplicates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "count": self.count,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
        }


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    
    name: str
    description: str
    thresholds: Dict[str, Dict[str, float]]  # category -> {base_threshold, length_adjustment}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "thresholds": self.thresholds,
        }


@dataclass
class ExperimentResult:
    """Results from an experiment."""
    
    config: ExperimentConfig
    num_samples: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    cache_hit_ratio: float  # Total cache usage (TP + FP) / total
    clean_hit_ratio: float  # Only correct hits (TP) / total
    roc_auc: float
    pr_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    category_metrics: List[CategoryMetrics]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "num_samples": self.num_samples,
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "false_positive_rate": self.false_positive_rate,
                "cache_hit_ratio": self.cache_hit_ratio,
                "clean_hit_ratio": self.clean_hit_ratio,
                "roc_auc": self.roc_auc,
                "pr_auc": self.pr_auc,
            },
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
            },
            "category_metrics": [cm.to_dict() for cm in self.category_metrics],
            "timestamp": self.timestamp,
        }


def create_experiment_configs() -> List[ExperimentConfig]:
    """
    Create experiment configurations for adaptive threshold evaluation.
    
    This function defines 5 distinct threshold configurations:
    1. fixed_baseline: Standard GPTCache threshold (0.80) for all categories
    2. precision_focused: Higher thresholds to minimize false positives
    3. adaptive_category_tuned: Moderate thresholds for balanced performance
    4. balanced_hybrid: Optimized for equal precision and recall
    5. length_weighted: Adjusted thresholds based on query length
    
    Each configuration specifies base thresholds and length adjustments per category.
    The UNKNOWN category maintains 0.80 threshold for GPTCache compatibility.
    """
    
    configs = []
    
    # 1. Fixed Baseline - Reference configuration
    configs.append(ExperimentConfig(
        name="fixed_baseline",
        description="Fixed threshold of 0.80 for all queries (GPTCache standard baseline)",
        thresholds={
            "FACTUAL": {"base": 0.80, "adj": 0.0},
            "SUBJECTIVE": {"base": 0.80, "adj": 0.0},
            "COMPARISON": {"base": 0.80, "adj": 0.0},
            "MATHEMATICAL": {"base": 0.80, "adj": 0.0},
            "CREATIVE": {"base": 0.80, "adj": 0.0},
            "CODE": {"base": 0.80, "adj": 0.0},
            "UNKNOWN": {"base": 0.80, "adj": 0.0},
        }
    ))
    
    # 2. Precision Focused - Minimize false positives
    configs.append(ExperimentConfig(
        name="precision_focused",
        description="Precision-optimized: Higher thresholds to reduce false positives",
        thresholds={
            "FACTUAL": {"base": 0.90, "adj": -0.003},
            "SUBJECTIVE": {"base": 0.89, "adj": -0.005},
            "COMPARISON": {"base": 0.91, "adj": -0.003},
            "MATHEMATICAL": {"base": 0.89, "adj": 0.0},
            "CREATIVE": {"base": 0.85, "adj": -0.005},
            "CODE": {"base": 0.92, "adj": -0.003},
            "UNKNOWN": {"base": 0.80, "adj": 0.0},
        }
    ))
    
    # 3. Adaptive Category Tuned - Balanced approach
    configs.append(ExperimentConfig(
        name="adaptive_category_tuned",
        description="Balanced thresholds with moderate length adjustments",
        thresholds={
            "FACTUAL": {"base": 0.85, "adj": -0.020},
            "SUBJECTIVE": {"base": 0.80, "adj": -0.035},
            "COMPARISON": {"base": 0.85, "adj": -0.025},
            "MATHEMATICAL": {"base": 0.87, "adj": -0.015},
            "CREATIVE": {"base": 0.55, "adj": -0.035},
            "CODE": {"base": 0.87, "adj": -0.020},
            "UNKNOWN": {"base": 0.80, "adj": 0.0},
        }
    ))
    
    # 4. Balanced Hybrid - Optimize F1 Score
    configs.append(ExperimentConfig(
        name="balanced_hybrid",
        description="Balanced approach optimizing for equal precision and recall",
        thresholds={
            "FACTUAL": {"base": 0.87, "adj": -0.008},
            "SUBJECTIVE": {"base": 0.85, "adj": -0.015},
            "COMPARISON": {"base": 0.88, "adj": -0.010},
            "MATHEMATICAL": {"base": 0.91, "adj": -0.005},
            "CREATIVE": {"base": 0.60, "adj": -0.025},
            "CODE": {"base": 0.91, "adj": -0.007},
            "UNKNOWN": {"base": 0.80, "adj": 0.0},
        }
    ))
    
    # 5. length_weighted - Length-based adjustments
    configs.append(ExperimentConfig(
        name="length_weighted",
        description="Length-based adjustments (longer queries receive more lenient thresholds)",
        thresholds={
            "FACTUAL": {"base": 0.88, "adj": -0.015},
            "SUBJECTIVE": {"base": 0.86, "adj": -0.025},
            "COMPARISON": {"base": 0.88, "adj": -0.020},
            "MATHEMATICAL": {"base": 0.91, "adj": -0.010},
            "CREATIVE": {"base": 0.62, "adj": -0.030},
            "CODE": {"base": 0.92, "adj": -0.012},
            "UNKNOWN": {"base": 0.80, "adj": 0.0},
        }
    ))
    
    return configs


def evaluate_configuration(
    config: ExperimentConfig,
    pairs: List[Dict[str, Any]],
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    base_config: Dict[str, Any],
) -> ExperimentResult:
    """
    Evaluate a specific threshold configuration with per-category metrics.
    
    Args:
        config: Experiment configuration.
        pairs: Question pairs.
        embeddings1: Embeddings for question1.
        embeddings2: Embeddings for question2.
        base_config: Base configuration dictionary.
        
    Returns:
        Experiment result with overall and per-category metrics.
    """
    logger.info(f"Evaluating configuration: {config.name}")
    
    # Create evaluator with threshold overrides (uses default config + overrides)
    evaluator = AdaptiveSimilarityEvaluation(
        config=base_config,
        threshold_overrides=config.thresholds,
    )
    
    # Track overall metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    cache_hits = 0
    
    # Track per-category metrics
    category_stats = {}
    for cat in QueryCategory:
        category_stats[cat.name] = {
            "count": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    
    # Store predictions and ground truth for AUC calculation
    all_similarities = []
    all_labels = []
    all_predictions = []
    
    # Evaluate each pair
    for i, pair in enumerate(tqdm(pairs, desc=f"Evaluating {config.name}", ncols=100)):
        query_emb = embeddings1[i]
        cache_emb = embeddings2[i]
        is_duplicate = pair["is_duplicate"]
        
        # Compute cosine similarity
        similarity = float(np.dot(query_emb, cache_emb))
        
        # Get adaptive threshold for this query
        query_text = pair["question1"]
        
        # Classify query
        category = evaluator._classify_query(query_text)
        
        # Get adaptive threshold from evaluator
        adaptive_threshold = evaluator._calculate_adaptive_threshold(query_text, category)
        
        # Check if similar
        is_similar = similarity >= adaptive_threshold
        
        # Store for AUC calculation
        all_similarities.append(similarity)
        all_labels.append(1 if is_duplicate else 0)
        all_predictions.append(1 if is_similar else 0)
        
        # Update per-category stats
        cat_name = category.name
        category_stats[cat_name]["count"] += 1
        
        # Update confusion matrices (overall and per-category)
        if is_duplicate:
            if is_similar:
                true_positives += 1
                cache_hits += 1
                category_stats[cat_name]["tp"] += 1
            else:
                false_negatives += 1
                category_stats[cat_name]["fn"] += 1
        else:
            if is_similar:
                false_positives += 1
                cache_hits += 1
                category_stats[cat_name]["fp"] += 1
            else:
                true_negatives += 1
                category_stats[cat_name]["tn"] += 1
    
    # Calculate overall metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    cache_hit_ratio = cache_hits / len(pairs) if len(pairs) > 0 else 0  # Total usage (TP + FP)
    clean_hit_ratio = true_positives / len(pairs) if len(pairs) > 0 else 0  # Only correct hits (TP)
    
    # Calculate AUC scores using predictions (not raw similarities)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = roc_auc_score(all_labels, all_predictions)
        pr_auc = average_precision_score(all_labels, all_predictions)
    except Exception as e:
        logger.warning(f"Could not calculate AUC scores: {e}")
        roc_auc = 0.0
        pr_auc = 0.0
    
    # Calculate per-category metrics
    category_metrics = []
    for cat_name, stats in category_stats.items():
        if stats["count"] == 0:
            continue
            
        cat_tp = stats["tp"]
        cat_fp = stats["fp"]
        cat_tn = stats["tn"]
        cat_fn = stats["fn"]
        
        cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
        cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
        cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
        cat_fpr = cat_fp / (cat_fp + cat_tn) if (cat_fp + cat_tn) > 0 else 0
        cat_fnr = cat_fn / (cat_fn + cat_tp) if (cat_fn + cat_tp) > 0 else 0  # False Negative Rate
        
        category_metrics.append(CategoryMetrics(
            category=cat_name,
            count=stats["count"],
            precision=cat_precision,
            recall=cat_recall,
            f1_score=cat_f1,
            true_positives=cat_tp,
            false_positives=cat_fp,
            true_negatives=cat_tn,
            false_negatives=cat_fn,
            false_positive_rate=cat_fpr,
            false_negative_rate=cat_fnr,
        ))
    
    return ExperimentResult(
        config=config,
        num_samples=len(pairs),
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        false_positive_rate=fpr,
        cache_hit_ratio=cache_hit_ratio,
        clean_hit_ratio=clean_hit_ratio,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        category_metrics=category_metrics,
        timestamp=datetime.now().isoformat(),
    )


def print_results_table(results: List[ExperimentResult]) -> None:
    """Print formatted results table with all metrics."""
    
    # Sort by F1 score
    sorted_results = sorted(results, key=lambda r: r.f1_score, reverse=True)
    
    print("\n" + "=" * 140)
    print(" " * 55 + "EXPERIMENT RESULTS SUMMARY")
    print("=" * 140)
    print()
    
    # Main metrics table
    print("Overall Metrics:")
    print("-" * 140)
    
    # Column widths
    rank_w = 6
    name_w = 30
    f1_w = 8
    prec_w = 8
    rec_w = 8
    fpr_w = 8
    hit_w = 9
    clean_w = 9
    roc_w = 10
    pr_w = 10
    
    # Header with proper alignment
    header = (
        f"{'Rank':<{rank_w}}"
        f"{'Configuration':<{name_w}}"
        f"{'F1':>{f1_w}}"
        f"{'Prec':>{prec_w}}"
        f"{'Recall':>{rec_w}}"
        f"{'FPR':>{fpr_w}}"
        f"{'Hit%':>{hit_w}}"
        f"{'Clean%':>{clean_w}}"
        f"{'ROC-AUC':>{roc_w}}"
        f"{'PR-AUC':>{pr_w}}"
    )
    print(header)
    print("-" * 150)
    
    for i, result in enumerate(sorted_results, 1):
        rank_str = f"{i}"
        name = result.config.name[:28]
        hit_pct = f"{result.cache_hit_ratio*100:.1f}%"
        clean_pct = f"{result.clean_hit_ratio*100:.1f}%"
        
        row = (
            f"{rank_str:<{rank_w}}"
            f"{name:<{name_w}}"
            f"{result.f1_score:>{f1_w}.4f}"
            f"{result.precision:>{prec_w}.4f}"
            f"{result.recall:>{rec_w}.4f}"
            f"{result.false_positive_rate:>{fpr_w}.4f}"
            f"{hit_pct:>{hit_w}}"
            f"{clean_pct:>{clean_w}}"
            f"{result.roc_auc:>{roc_w}.4f}"
            f"{result.pr_auc:>{pr_w}.4f}"
        )
        print(row)
    
    print("=" * 150)
    print()
    
    # Find baseline for comparison
    baseline = next((r for r in sorted_results if r.config.name == "fixed_baseline"), None)
    
    # Sort by PRECISION (FP are more costly than FN)
    sorted_by_precision = sorted(sorted_results, key=lambda r: r.precision, reverse=True)
    
    # Best configurations summary (sorted by precision)
    print("Best Configurations (Sorted by Precision - FP errors are more costly):")
    print("-" * 150)
    
    # Show all non-baseline configs (should be 4 configs)
    non_baseline = [r for r in sorted_by_precision if r.config.name != "fixed_baseline"]
    
    for i, result in enumerate(non_baseline, 1):
        if baseline:
            prec_delta = ((result.precision - baseline.precision) / baseline.precision) * 100
            f1_delta = ((result.f1_score - baseline.f1_score) / baseline.f1_score) * 100
            recall_delta = ((result.recall - baseline.recall) / baseline.recall) * 100
            
            print(f"  #{i} {result.config.name:<28} "
                  f"Prec={result.precision:.4f} ({prec_delta:+.1f}%), "
                  f"F1={result.f1_score:.4f} ({f1_delta:+.1f}%), "
                  f"Recall={result.recall:.4f} ({recall_delta:+.1f}%)")
        else:
            print(f"  #{i} {result.config.name:<28} "
                  f"Prec={result.precision:.4f}, "
                  f"F1={result.f1_score:.4f}, "
                  f"Recall={result.recall:.4f}")
    
    if baseline:
        print(f"\n  📊 Baseline (fixed_baseline):     "
              f"Prec={baseline.precision:.4f}, "
              f"F1={baseline.f1_score:.4f}, "
              f"Recall={baseline.recall:.4f}")
    
    print("-" * 150)
    print()


def print_category_breakdown(result: ExperimentResult) -> None:
    """Print per-category metrics for a configuration."""
    
    print(f"\nCategory Breakdown for: {result.config.name}")
    print("=" * 150)
    
    # Sort categories by count (descending)
    sorted_cats = sorted(result.category_metrics, key=lambda c: c.count, reverse=True)
    
    # Column widths
    cat_w = 20
    count_w = 16
    f1_w = 8
    prec_w = 8
    rec_w = 8
    fpr_w = 8
    fnr_w = 8
    tp_w = 7
    fp_w = 7
    tn_w = 7
    fn_w = 7
    
    header = (
        f"{'Category':<{cat_w}}"
        f"{'Count':>{count_w}}"
        f"{'F1':>{f1_w}}"
        f"{'Prec':>{prec_w}}"
        f"{'Recall':>{rec_w}}"
        f"{'FPR':>{fpr_w}}"
        f"{'FNR':>{fnr_w}}"
        f"{'TP':>{tp_w}}"
        f"{'FP':>{fp_w}}"
        f"{'TN':>{tn_w}}"
        f"{'FN':>{fn_w}}"
    )
    print(header)
    print("-" * 150)
    
    for cat_metric in sorted_cats:
        if cat_metric.count == 0:
            continue
            
        pct = (cat_metric.count / result.num_samples) * 100
        count_str = f"{cat_metric.count:,} ({pct:.1f}%)"
        
        row = (
            f"{cat_metric.category:<{cat_w}}"
            f"{count_str:>{count_w}}"
            f"{cat_metric.f1_score:>{f1_w}.4f}"
            f"{cat_metric.precision:>{prec_w}.4f}"
            f"{cat_metric.recall:>{rec_w}.4f}"
            f"{cat_metric.false_positive_rate:>{fpr_w}.4f}"
            f"{cat_metric.false_negative_rate:>{fnr_w}.4f}"
            f"{cat_metric.true_positives:>{tp_w}}"
            f"{cat_metric.false_positives:>{fp_w}}"
            f"{cat_metric.true_negatives:>{tn_w}}"
            f"{cat_metric.false_negatives:>{fn_w}}"
        )
        print(row)
    
    print("=" * 150)
    print()


def print_error_analysis(result: ExperimentResult) -> None:
    """Print detailed error analysis with FP and FN distribution."""
    
    print(f"\n{'='*140}")
    print(f"ERROR ANALYSIS for: {result.config.name}")
    print(f"{'='*140}\n")
    
    # Overall error summary
    total_errors = result.false_positives + result.false_negatives
    error_rate = total_errors / result.num_samples if result.num_samples > 0 else 0
    
    print(f"Overall Error Summary:")
    print(f"  Total Errors:        {total_errors:,} ({error_rate*100:.2f}% of {result.num_samples:,} samples)")
    print(f"  False Positives:     {result.false_positives:,} ({result.false_positive_rate*100:.2f}% FPR)")
    print(f"  False Negatives:     {result.false_negatives:,} (missed {result.false_negatives} duplicates)")
    print(f"  True Positives:      {result.true_positives:,} (correctly cached)")
    print(f"  True Negatives:      {result.true_negatives:,} (correctly rejected)")
    print(f"\n  Cache Hit Ratio:     {result.cache_hit_ratio*100:.2f}% (TP + FP)")
    print(f"  Clean Hit Ratio:     {result.clean_hit_ratio*100:.2f}% (TP only)")
    print(f"  Contamination:       {((result.cache_hit_ratio - result.clean_hit_ratio)*100):.2f}% (FP in cache hits)")
    print()
    
    # Sort categories by count for better overview
    sorted_cats = sorted(result.category_metrics, key=lambda c: c.count, reverse=True)
    
    # Filter out categories with no samples
    active_cats = [c for c in sorted_cats if c.count > 0]
    
    # FP Analysis
    print(f"FALSE POSITIVE Analysis (Wrong Cache Hits - Returned incorrect answer):")
    print(f"{'-'*140}")
    
    # Calculate total FP
    total_fp = sum(c.false_positives for c in active_cats)
    
    if total_fp > 0:
        print(f"{'Category':<20} {'Count':>12} {'FP':>8} {'FP%':>8} {'FPR':>8} {'Contrib':>10} {'Impact'}")
        print(f"{'-'*140}")
        
        # Sort by FP count descending
        fp_sorted = sorted(active_cats, key=lambda c: c.false_positives, reverse=True)
        
        for cat in fp_sorted:
            if cat.false_positives > 0:
                pct_of_cat = (cat.false_positives / cat.count) * 100 if cat.count > 0 else 0
                contrib = (cat.false_positives / total_fp) * 100 if total_fp > 0 else 0
                fpr_pct = cat.false_positive_rate * 100
                
                # Impact assessment
                if fpr_pct > 20:
                    impact = "🔴 CRITICAL"
                elif fpr_pct > 10:
                    impact = "🟡 HIGH"
                elif fpr_pct > 5:
                    impact = "🟢 MODERATE"
                else:
                    impact = "⚪ LOW"
                
                print(f"{cat.category:<20} {cat.count:>12,} {cat.false_positives:>8} "
                      f"{pct_of_cat:>7.1f}% {fpr_pct:>7.2f}% {contrib:>9.1f}% {impact}")
    else:
        print("  No false positives found!")
    
    print()
    
    # FN Analysis
    print(f"FALSE NEGATIVE Analysis (Missed Cache Hits - Missed duplicates):")
    print(f"{'-'*140}")
    
    # Calculate total FN
    total_fn = sum(c.false_negatives for c in active_cats)
    
    if total_fn > 0:
        print(f"{'Category':<20} {'Count':>12} {'FN':>8} {'FN%':>8} {'FNR':>8} {'Contrib':>10} {'Impact'}")
        print(f"{'-'*140}")
        
        # Sort by FN count descending
        fn_sorted = sorted(active_cats, key=lambda c: c.false_negatives, reverse=True)
        
        for cat in fn_sorted:
            if cat.false_negatives > 0:
                pct_of_cat = (cat.false_negatives / cat.count) * 100 if cat.count > 0 else 0
                contrib = (cat.false_negatives / total_fn) * 100 if total_fn > 0 else 0
                fnr_pct = cat.false_negative_rate * 100
                
                # Impact assessment
                if fnr_pct > 20:
                    impact = "🔴 CRITICAL"
                elif fnr_pct > 10:
                    impact = "🟡 HIGH"
                elif fnr_pct > 5:
                    impact = "🟢 MODERATE"
                else:
                    impact = "⚪ LOW"
                
                print(f"{cat.category:<20} {cat.count:>12,} {cat.false_negatives:>8} "
                      f"{pct_of_cat:>7.1f}% {fnr_pct:>7.2f}% {contrib:>9.1f}% {impact}")
    else:
        print("  No false negatives found!")
    
    print()
    
    # Key Insights
    print(f"KEY INSIGHTS:")
    print(f"{'-'*140}")
    
    # Find problematic categories
    high_fpr_cats = [c for c in active_cats if c.false_positive_rate > 0.10]
    high_fnr_cats = [c for c in active_cats if c.false_negative_rate > 0.10]
    
    if high_fpr_cats:
        print(f"  ⚠️  HIGH FP RATE (>10%):")
        for c in high_fpr_cats:
            print(f"     - {c.category}: {c.false_positive_rate*100:.1f}% FPR → Consider RAISING threshold")
    
    if high_fnr_cats:
        print(f"  ⚠️  HIGH FN RATE (>10%):")
        for c in high_fnr_cats:
            print(f"     - {c.category}: {c.false_negative_rate*100:.1f}% FNR → Consider LOWERING threshold")
    
    if not high_fpr_cats and not high_fnr_cats:
        print(f"  ✅ No categories with critical error rates (all <10%)")
    
    print(f"{'='*140}\n")


def save_results(
    results: List[ExperimentResult],
    output_dir: Path,
    num_samples: int,
) -> None:
    """
    Save experiment results to both JSON (for R analysis) and Markdown (for quick overview).
    
    Args:
        results: List of experiment results.
        output_dir: Output directory.
        num_samples: Number of samples used.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # detailed JSON for analysis
    json_filename = f"experiments_detailed_{num_samples}samples_{timestamp}.json"
    json_filepath = output_dir / json_filename
    
    # Create comprehensive JSON structure
    json_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "num_configs": len(results),
            "dataset": "QQP (Quora Question Pairs)",
            "split": "validation",
        },
        "configurations": []
    }
    
    # Add each configuration with full details
    for result in results:
        config_data = {
            "name": result.config.name,
            "description": result.config.description,
            
            # Overall metrics
            "metrics": {
                "f1_score": float(result.f1_score),
                "precision": float(result.precision),
                "recall": float(result.recall),
                "cache_hit_ratio": float(result.cache_hit_ratio),
                "false_positive_rate": float(result.false_positive_rate),
                "roc_auc": float(result.roc_auc),
                "pr_auc": float(result.pr_auc),
            },
            
            # Confusion matrix
            "confusion_matrix": {
                "true_positives": int(result.true_positives),
                "false_positives": int(result.false_positives),
                "true_negatives": int(result.true_negatives),
                "false_negatives": int(result.false_negatives),
            },
            
            # Threshold configuration
            "thresholds": {
                category: {
                    "base": float(params["base"]),
                    "adjustment": float(params["adj"])
                }
                for category, params in result.config.thresholds.items()
            },
            
            # Per-category metrics (for detailed R analysis)
            "category_metrics": [
                {
                    "category": cat.category,
                    "count": int(cat.count),
                    "percentage": float((cat.count / result.num_samples) * 100),
                    "f1_score": float(cat.f1_score),
                    "precision": float(cat.precision),
                    "recall": float(cat.recall),
                    "false_positive_rate": float(cat.false_positive_rate),
                    "true_positives": int(cat.true_positives),
                    "false_positives": int(cat.false_positives),
                    "true_negatives": int(cat.true_negatives),
                    "false_negatives": int(cat.false_negatives),
                }
                for cat in result.category_metrics
            ]
        }
        
        json_data["configurations"].append(config_data)
    
    # Save JSON
    with open(json_filepath, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Detailed JSON saved to: {json_filepath}")
    
    # ========== 2. MARKDOWN SUMMARY FOR QUICK OVERVIEW ==========
    md_filename = f"SUMMARY_{num_samples}samples_{timestamp}.md"
    md_filepath = output_dir / md_filename
    
    with open(md_filepath, "w") as f:
        f.write("# Experiment Results Summary\n\n")
        f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Number of Samples:** {num_samples:,}\n")
        f.write(f"**Number of Configurations:** {len(results)}\n")
        f.write(f"**Dataset:** QQP (Quora Question Pairs)\n\n")
        
        f.write("## Overall Rankings (by F1 Score)\n\n")
        f.write("| Rank | Configuration | F1 Score | Precision | Recall | Cache Hit % | ROC-AUC | PR-AUC |\n")
        f.write("|------|---------------|----------|-----------|--------|-------------|---------|--------|\n")
        
        for rank, result in enumerate(sorted(results, key=lambda r: r.f1_score, reverse=True), 1):
            f.write(
                f"| {rank} "
                f"| {result.config.name:<30} "
                f"| {result.f1_score:.4f} "
                f"| {result.precision:.4f} "
                f"| {result.recall:.4f} "
                f"| {result.cache_hit_ratio*100:.1f}% "
                f"| {result.roc_auc:.4f} "
                f"| {result.pr_auc:.4f} |\n"
            )
        
        f.write("\n## Best Configurations\n\n")
        
        # Best by different metrics
        best_f1 = max(results, key=lambda r: r.f1_score)
        best_precision = max(results, key=lambda r: r.precision)
        best_recall = max(results, key=lambda r: r.recall)
        best_cache = max(results, key=lambda r: r.cache_hit_ratio)
        
        f.write(f"### Best F1 Score: **{best_f1.config.name}**\n")
        f.write(f"- F1: {best_f1.f1_score:.4f}\n")
        f.write(f"- Precision: {best_f1.precision:.4f}\n")
        f.write(f"- Recall: {best_f1.recall:.4f}\n")
        f.write(f"- Cache Hit: {best_f1.cache_hit_ratio*100:.1f}%\n")
        f.write(f"- ROC-AUC: {best_f1.roc_auc:.4f}\n\n")
        
        f.write(f"### Best Precision: **{best_precision.config.name}**\n")
        f.write(f"- Precision: {best_precision.precision:.4f}\n")
        f.write(f"- F1: {best_precision.f1_score:.4f}\n")
        f.write(f"- Recall: {best_precision.recall:.4f}\n\n")
        
        f.write(f"### Best Recall: **{best_recall.config.name}**\n")
        f.write(f"- Recall: {best_recall.recall:.4f}\n")
        f.write(f"- F1: {best_recall.f1_score:.4f}\n")
        f.write(f"- Precision: {best_recall.precision:.4f}\n\n")
        
        f.write(f"### Best Cache Hit Rate: **{best_cache.config.name}**\n")
        f.write(f"- Cache Hit: {best_cache.cache_hit_ratio*100:.1f}%\n")
        f.write(f"- F1: {best_cache.f1_score:.4f}\n\n")
        
        # Top 3 detailed comparison
        f.write("## Top 3 Configurations - Detailed Comparison\n\n")
        
        top_3 = sorted(results, key=lambda r: r.f1_score, reverse=True)[:3]
        
        for i, result in enumerate(top_3, 1):
            f.write(f"### #{i}: {result.config.name}\n\n")
            f.write(f"**Description:** {result.config.description}\n\n")
            
            f.write("**Overall Metrics:**\n")
            f.write(f"- F1 Score: {result.f1_score:.4f}\n")
            f.write(f"- Precision: {result.precision:.4f}\n")
            f.write(f"- Recall: {result.recall:.4f}\n")
            f.write(f"- Cache Hit Rate: {result.cache_hit_ratio*100:.1f}%\n")
            f.write(f"- ROC-AUC: {result.roc_auc:.4f}\n")
            f.write(f"- PR-AUC: {result.pr_auc:.4f}\n")
            f.write(f"- False Positive Rate: {result.false_positive_rate:.4f}\n\n")
            
            f.write("**Confusion Matrix:**\n")
            f.write(f"- True Positives: {result.true_positives}\n")
            f.write(f"- False Positives: {result.false_positives}\n")
            f.write(f"- True Negatives: {result.true_negatives}\n")
            f.write(f"- False Negatives: {result.false_negatives}\n\n")
            
            f.write("**Per-Category Performance:**\n\n")
            f.write("| Category | Count (%) | F1 | Precision | Recall | FPR |\n")
            f.write("|----------|-----------|-----|-----------|--------|-----|\n")
            
            for cat in sorted(result.category_metrics, key=lambda c: c.count, reverse=True):
                if cat.count > 0:
                    pct = (cat.count / result.num_samples) * 100
                    f.write(
                        f"| {cat.category:<15} "
                        f"| {cat.count} ({pct:.1f}%) "
                        f"| {cat.f1_score:.4f} "
                        f"| {cat.precision:.4f} "
                        f"| {cat.recall:.4f} "
                        f"| {cat.false_positive_rate:.4f} |\n"
                    )
            f.write("\n")
            
            # Add threshold configuration
            f.write("**Threshold Configuration:**\n\n")
            f.write("| Category | Base Threshold | Length Adjustment |\n")
            f.write("|----------|----------------|-------------------|\n")
            for category, params in sorted(result.config.thresholds.items()):
                f.write(f"| {category:<15} | {params['base']:.3f} | {params['adj']:.3f} |\n")
            f.write("\n---\n\n")
    
    logger.info(f"Markdown summary saved to: {md_filepath}")
    logger.info(f"Detailed JSON saved to: {json_filepath}")



def main():
    """Run all experiments."""
    
    # Get script directory and go to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load configuration from project root
    config_path = project_root / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    num_samples = config["max_eval_samples"]
    logger.info(f"Running experiments with {num_samples} samples")
    
    # Create output directory with timestamp (relative to script location)
    output_dir = script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading QQP dataset...")
    dataset = load_dataset("nyu-mll/glue", "qqp", split="validation")
    pairs = []
    for item in dataset.select(range(num_samples)):
        pairs.append({
            "question1": item["question1"],
            "question2": item["question2"],
            "is_duplicate": bool(item["label"]),
        })
    
    # Generate embeddings once
    logger.info("Generating embeddings...")
    embedding_model = Huggingface(
        model=config["embedding"]["model"],
    )
    
    questions1 = [p["question1"] for p in pairs]
    questions2 = [p["question2"] for p in pairs]
    
    embeddings1 = np.array([embedding_model.to_embeddings(q) for q in tqdm(questions1, desc="Embedding Q1")])
    embeddings2 = np.array([embedding_model.to_embeddings(q) for q in tqdm(questions2, desc="Embedding Q2")])
    
    # Normalize embeddings
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Create experiment configurations
    logger.info("Creating experiment configurations...")
    configs = create_experiment_configs()
    logger.info(f"Testing {len(configs)} configurations")
    
    # Run experiments
    results = []
    for config_obj in configs:
        result = evaluate_configuration(
            config=config_obj,
            pairs=pairs,
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            base_config=config,
        )
        results.append(result)
        
        # Quick summary after each config
        logger.info(
            f"{config_obj.name:30s} | "
            f"F1={result.f1_score:.4f} | "
            f"Prec={result.precision:.4f} | "
            f"Recall={result.recall:.4f} | "
            f"Hit={result.cache_hit_ratio*100:5.1f}% | "
            f"ROC-AUC={result.roc_auc:.4f}"
        )
    
    # Save results
    save_results(results, output_dir, num_samples)
    
    # Print detailed results table
    print_results_table(results)
    
    # Print category breakdown for all configurations
    print("\n" + "=" * 150)
    print(" " * 50 + "DETAILED CATEGORY BREAKDOWN - ALL CONFIGURATIONS")
    print("=" * 150)
    
    # Sort by precision (most important metric) and show all
    sorted_by_precision = sorted(results, key=lambda r: r.precision, reverse=True)
    
    for i, result in enumerate(sorted_by_precision, 1):
        print_category_breakdown(result)
        print_error_analysis(result)


if __name__ == "__main__":
    main()
