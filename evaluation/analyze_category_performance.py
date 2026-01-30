"""
Analyze category-specific performance across different threshold configurations.

Compares performance metrics (Precision, FPR, Hit Rate, F1) across all 
configurations for each query category and identifies the best performers.

Usage:
    poetry run python evaluation/analyze_category_performance.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
import sys


def load_latest_results() -> Dict[str, Any]:
    """Load the most recent experiment results."""
    results_dir = Path(__file__).parent / "results"
    
    # Find latest results file
    result_files = sorted(results_dir.glob("experiments_detailed_*.json"), reverse=True)
    if not result_files:
        logger.error("No experiment results found!")
        sys.exit(1)
    
    latest = result_files[0]
    logger.info(f"Loading results from: {latest.name}")
    
    with open(latest) as f:
        return json.load(f)


def analyze_category_performance(results: Dict[str, Any]) -> None:
    """Analyze and compare category-specific metrics across configurations."""
    
    configurations = results["configurations"]
    
    # Categories to analyze (excluding UNKNOWN)
    categories = ["FACTUAL", "SUBJECTIVE", "COMPARISON", "MATHEMATICAL", "CREATIVE", "CODE"]

    logger.info("="*90)
    logger.info("CATEGORY-SPECIFIC PERFORMANCE ANALYSIS")
    logger.info("="*90)
    
    for category in categories:
        logger.info("")
        logger.info(f"{'='*90}")
        logger.info(f"CATEGORY: {category}")
        logger.info(f"{'='*90}")
        
        # Collect metrics for this category across all configs
        category_data = []
        
        for config in configurations:
            config_name = config["name"]
            
            # Check if category metrics exist
            if "category_metrics" not in config:
                continue
            
            # Find this category in the list
            category_metrics = None
            for cat_metric in config["category_metrics"]:
                if cat_metric["category"] == category:
                    category_metrics = cat_metric
                    break
            
            if not category_metrics or category_metrics["count"] == 0:
                continue
            
            # Get threshold info
            threshold_info = config["thresholds"][category]
            base_threshold = threshold_info["base"]
            length_adj = threshold_info["adjustment"]
            
            precision = category_metrics["precision"]
            recall = category_metrics["recall"]
            fpr = category_metrics["false_positive_rate"]
            f1 = category_metrics["f1_score"]
            
            # Calculate hit rate (cache hits / total pairs)
            hit_rate = (category_metrics["true_positives"] + category_metrics["false_positives"]) / category_metrics["count"]
            
            category_data.append({
                "config": config_name,
                "base_threshold": base_threshold,
                "length_adj": length_adj,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "f1": f1,
                "hit_rate": hit_rate,
                "total_samples": category_metrics["count"],
            })
        
        if not category_data:
            logger.warning(f"  No data for category {category}")
            continue
        
        # Sort by different criteria to find best performers
        by_precision = sorted(category_data, key=lambda x: x["precision"], reverse=True)
        by_fpr = sorted(category_data, key=lambda x: x["fpr"])
        by_hit_rate = sorted(category_data, key=lambda x: x["hit_rate"], reverse=True)
        by_f1 = sorted(category_data, key=lambda x: x["f1"], reverse=True)
        
        # Display all configurations for this category
        logger.info("  All configurations (sorted by Precision):")
        logger.info(f"  {'Config':<32} {'Thr':<7} {'Adj':<8} {'Prec':<7} {'FPR':<7} {'Hit':<7} {'F1':<8} {'N':<8}")
        logger.info(f"  {'-'*87}")
        
        for data in by_precision:
            logger.info(
                f"  {data['config']:<32} "
                f"{data['base_threshold']:<7.3f} "
                f"{data['length_adj']:<8.3f} "
                f"{data['precision']*100:>5.1f}% "
                f"{data['fpr']*100:>5.1f}% "
                f"{data['hit_rate']*100:>5.1f}% "
                f"{data['f1']:<8.4f} "
                f"{data['total_samples']:<8}"
            )
        
        # Best performers by metric
        logger.info("")
        logger.info("  Best performers:")
        logger.info(f"     Highest Prec: {by_precision[0]['config']:<32} (Prec: {by_precision[0]['precision']*100:.1f}%, FPR: {by_precision[0]['fpr']*100:.1f}%)")
        logger.info(f"     Lowest FPR:   {by_fpr[0]['config']:<32} (FPR: {by_fpr[0]['fpr']*100:.1f}%, Prec: {by_fpr[0]['precision']*100:.1f}%)")
        logger.info(f"     Highest Hit:  {by_hit_rate[0]['config']:<32} (Hit: {by_hit_rate[0]['hit_rate']*100:.1f}%, FPR: {by_hit_rate[0]['fpr']*100:.1f}%)")
        logger.info(f"     Best F1:      {by_f1[0]['config']:<32} (F1: {by_f1[0]['f1']:.4f}, Prec: {by_f1[0]['precision']*100:.1f}%)")


def main():
    """Main analysis function."""
    results = load_latest_results()
    analyze_category_performance(results)


if __name__ == "__main__":
    main()
