#!/usr/bin/env python3
"""
Dataset Category Analysis Script

This script analyzes the entire dataset to understand the distribution of
query categories and their characteristics.

Purpose:
- Understand category distribution across the dataset
- Identify which categories are meaningful for threshold adaptation
- Provide statistics for category-based strategy design

Usage:
    poetry run python evaluation/analyze_dataset_categories.py

Output:
- JSON with detailed category statistics
- Markdown report with category breakdown
- Insights on category-specific duplicate rates
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import json
import numpy as np
from datetime import datetime
from typing import List
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass, asdict
from collections import defaultdict

from datasets import load_dataset

# Import from gptcache and src
from src.similarity_evaluators.adaptive_threshold import (
    AdaptiveSimilarityEvaluation,
    QueryCategory,
)


@dataclass
class CategoryStats:
    """Statistics for a single category"""
    category: str
    total_count: int
    percentage: float
    duplicate_count: int
    non_duplicate_count: int
    duplicate_rate: float
    avg_q1_length: float
    avg_q2_length: float
    avg_combined_length: float


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def analyze_categories(
    questions1: List[str],
    questions2: List[str],
    labels: List[int],
    evaluator: AdaptiveSimilarityEvaluation
) -> List[CategoryStats]:
    """
    Analyze category distribution and statistics
    
    Args:
        questions1: First questions
        questions2: Second questions
        labels: Ground truth labels (1=duplicate, 0=not duplicate)
        evaluator: AdaptiveSimilarityEvaluation instance for classification
    
    Returns:
        List of CategoryStats for each category
    """
    # Track statistics per category
    category_data = defaultdict(lambda: {
        'count': 0,
        'duplicates': 0,
        'q1_lengths': [],
        'q2_lengths': []
    })
    
    # Classify all questions
    logger.info("Classifying questions by category...")
    for q1, q2, label in tqdm(zip(questions1, questions2, labels), total=len(labels)):
        # Classify based on first question (primary query)
        category = evaluator._classify_query(q1)
        category_name = category.name
        
        # Update statistics
        data = category_data[category_name]
        data['count'] += 1
        if label == 1:
            data['duplicates'] += 1
        data['q1_lengths'].append(len(q1.split()))
        data['q2_lengths'].append(len(q2.split()))
    
    # Calculate statistics for each category
    total_samples = len(labels)
    results = []
    
    for category, data in category_data.items():
        count = data['count']
        dup_count = data['duplicates']
        non_dup_count = count - dup_count
        
        stats = CategoryStats(
            category=category,
            total_count=count,
            percentage=count / total_samples * 100,
            duplicate_count=dup_count,
            non_duplicate_count=non_dup_count,
            duplicate_rate=dup_count / count * 100 if count > 0 else 0.0,
            avg_q1_length=float(np.mean(data['q1_lengths'])) if data['q1_lengths'] else 0.0,
            avg_q2_length=float(np.mean(data['q2_lengths'])) if data['q2_lengths'] else 0.0,
            avg_combined_length=float((np.mean(data['q1_lengths']) + np.mean(data['q2_lengths'])) / 2) 
                               if data['q1_lengths'] else 0.0
        )
        results.append(stats)
    
    # Sort by count (descending)
    results.sort(key=lambda x: x.total_count, reverse=True)
    
    return results


def save_results(stats: List[CategoryStats], output_dir: Path, num_samples: int):
    """Save results to JSON and Markdown files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"category_analysis_{num_samples}samples_{timestamp}.json"
    json_data = {
        "metadata": {
            "timestamp": timestamp,
            "num_samples": num_samples,
            "num_categories": len(stats)
        },
        "categories": [asdict(s) for s in stats]
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"JSON results saved to: {json_path}")
    
    # Create Markdown report
    md_path = output_dir / f"CATEGORY_ANALYSIS_{num_samples}samples_{timestamp}.md"
    
    with open(md_path, 'w') as f:
        f.write("# Dataset Category Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Samples:** {num_samples:,}\n")
        f.write(f"**Categories Found:** {len(stats)}\n\n")
        
        # Overall statistics
        total_duplicates = sum(s.duplicate_count for s in stats)
        total_non_duplicates = sum(s.non_duplicate_count for s in stats)
        overall_dup_rate = total_duplicates / num_samples * 100
        
        f.write("## Overall Dataset Statistics\n\n")
        f.write(f"- **Total Pairs:** {num_samples:,}\n")
        f.write(f"- **Duplicates:** {total_duplicates:,} ({overall_dup_rate:.1f}%)\n")
        f.write(f"- **Non-Duplicates:** {total_non_duplicates:,} ({100-overall_dup_rate:.1f}%)\n\n")
        
        # Category breakdown
        f.write("## Category Distribution\n\n")
        f.write("| Rank | Category | Count | % of Dataset | Duplicates | Non-Dup | Dup Rate | Avg Length |\n")
        f.write("|------|----------|-------|--------------|------------|---------|----------|------------|\n")
        
        for i, stat in enumerate(stats, 1):
            f.write(f"| {i} | **{stat.category}** | {stat.total_count:,} | "
                    f"{stat.percentage:.1f}% | {stat.duplicate_count:,} | {stat.non_duplicate_count:,} | "
                    f"{stat.duplicate_rate:.1f}% | {stat.avg_combined_length:.1f} words |\n")
        
        f.write("\n**Legend:**\n")
        f.write("- Major category (>=10% of dataset)\n")
        f.write("- Significant category (5-10% of dataset)\n\n")
        
        # Insights
        f.write("## Key Insights\n\n")
        
        # Top 3 categories
        top3 = stats[:3]
        top3_percentage = sum(s.percentage for s in top3)
        f.write(f"### Top 3 Categories Cover {top3_percentage:.1f}% of Dataset\n\n")
        for i, stat in enumerate(top3, 1):
            f.write(f"{i}. **{stat.category}**: {stat.total_count:,} samples ({stat.percentage:.1f}%)\n")
            f.write(f"   - Duplicate Rate: {stat.duplicate_rate:.1f}%\n")
            f.write(f"   - Avg Length: {stat.avg_combined_length:.1f} words\n\n")
        
        # Categories with unusual duplicate rates
        avg_dup_rate = overall_dup_rate
        high_dup_cats = [s for s in stats if s.duplicate_rate > avg_dup_rate * 1.2 and s.total_count >= 10]
        low_dup_cats = [s for s in stats if s.duplicate_rate < avg_dup_rate * 0.8 and s.total_count >= 10]
        
        if high_dup_cats:
            f.write(f"### Categories with High Duplicate Rate (>{avg_dup_rate*1.2:.1f}%)\n\n")
            f.write("These categories might benefit from **lower thresholds** (more aggressive caching):\n\n")
            for stat in high_dup_cats:
                f.write(f"- **{stat.category}**: {stat.duplicate_rate:.1f}% duplicates "
                        f"({stat.total_count} samples)\n")
            f.write("\n")
        
        if low_dup_cats:
            f.write(f"### Categories with Low Duplicate Rate (<{avg_dup_rate*0.8:.1f}%)\n\n")
            f.write("These categories might benefit from **higher thresholds** (more conservative caching):\n\n")
            for stat in low_dup_cats:
                f.write(f"- **{stat.category}**: {stat.duplicate_rate:.1f}% duplicates "
                        f"({stat.total_count} samples)\n")
            f.write("\n")
        
        # Rare categories
        rare_cats = [s for s in stats if s.percentage < 1.0]
        if rare_cats:
            rare_percentage = sum(s.percentage for s in rare_cats)
            f.write(f"### Rare Categories (<1% of dataset)\n\n")
            f.write(f"**{len(rare_cats)} rare categories** representing {rare_percentage:.1f}% of data:\n\n")
            for stat in rare_cats:
                f.write(f"- {stat.category}: {stat.total_count} samples ({stat.percentage:.2f}%)\n")
            f.write("\n")
            f.write("**Note:** These categories may not have enough samples for reliable threshold tuning.\n\n")
        
        # Recommendations
        f.write("## Recommendations for Adaptive Thresholding\n\n")
        
        major_cats = [s for s in stats if s.percentage >= 5.0]
        f.write(f"### Focus on Major Categories ({len(major_cats)} categories, "
                f"{sum(s.percentage for s in major_cats):.1f}% of data)\n\n")
        
        for stat in major_cats:
            if stat.duplicate_rate > avg_dup_rate * 1.1:
                recommendation = f"**Lower threshold** (more caching, high duplicate rate: {stat.duplicate_rate:.1f}%)"
            elif stat.duplicate_rate < avg_dup_rate * 0.9:
                recommendation = f"**Higher threshold** (less caching, low duplicate rate: {stat.duplicate_rate:.1f}%)"
            else:
                recommendation = f"**Standard threshold** (balanced, duplicate rate: {stat.duplicate_rate:.1f}%)"
            
            f.write(f"- **{stat.category}** ({stat.percentage:.1f}% of dataset): {recommendation}\n")
        
        f.write("\n### Handle Rare Categories\n\n")
        f.write("For categories with <1% of dataset:\n")
        f.write("- Use **UNKNOWN** category as fallback\n")
        f.write("- Apply conservative (higher) thresholds to avoid false positives\n")
        f.write("- Consider merging similar rare categories\n")
    
    logger.info(f"Markdown report saved to: {md_path}")


def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("Dataset Category Analysis")
    logger.info("=" * 80)
    
    # Load config
    config = load_config()
    seed = config.get('seed', 42)
    max_samples = config.get('max_eval_samples', 10000)
    model_name = config['embedding']['model']
    
    logger.info("Configuration:")
    logger.info(f"  - Max samples: {max_samples}")
    logger.info(f"  - Seed: {seed}")
    logger.info(f"  - Model: {model_name}")
    
    # Load dataset - use TRAIN split for full dataset analysis
    logger.info("\nLoading QQP dataset (TRAIN split for complete analysis)...")
    dataset = load_dataset("nyu-mll/glue", "qqp", split="train")
    
    # Don't limit samples - use full dataset for comprehensive analysis
    logger.info(f"Full dataset loaded: {len(dataset)} pairs")
    
    questions1 = [q for q in dataset['question1']]
    questions2 = [q for q in dataset['question2']]
    labels = [int(label) for label in dataset['label']]
    
    logger.info(f"Dataset loaded: {len(labels)} pairs")
    
    # Initialize evaluator (we only need the classifier, not embeddings)
    logger.info("\nInitializing category classifier...")
    base_config = {
        'similarity_threshold': 0.80,
        'embedding': {'model': model_name}
    }
    evaluator = AdaptiveSimilarityEvaluation(
        config=base_config,
        threshold_rules={}  # No rules needed, just for classification
    )
    
    # Analyze categories
    stats = analyze_categories(questions1, questions2, labels, evaluator)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    save_results(stats, output_dir, len(labels))
    
    # Print summary with clean logging
    logger.info("")
    logger.info("=" * 100)
    logger.info("CATEGORY ANALYSIS SUMMARY")
    logger.info("=" * 100)
    logger.info("")
    logger.info(f"Total Samples Analyzed: {len(labels):,}")
    logger.info(f"Total Categories Found: {len(stats)}")
    logger.info("")
    
    # Calculate overall stats
    total_duplicates = sum(s.duplicate_count for s in stats)
    total_non_duplicates = sum(s.non_duplicate_count for s in stats)
    overall_dup_rate = total_duplicates / len(labels) * 100
    
    logger.info("Overall Dataset Statistics:")
    logger.info(f"  • Duplicates:     {total_duplicates:>7,} ({overall_dup_rate:>5.1f}%)")
    logger.info(f"  • Non-Duplicates: {total_non_duplicates:>7,} ({100-overall_dup_rate:>5.1f}%)")
    logger.info("")
    
    # Print all categories
    logger.info("All Categories (sorted by frequency):")
    logger.info("-" * 100)
    
    # Column widths
    marker_w = 5
    rank_w = 4
    cat_w = 18
    count_w = 11
    pct_w = 10
    dup_w = 12
    rate_w = 10
    len_w = 10
    
    header = (
        f"{'':>{marker_w}}"
        f"{'Rank':<{rank_w}}"
        f"{'Category':<{cat_w}}"
        f"{'Count':>{count_w}}"
        f"{'% Dataset':>{pct_w}}"
        f"{'Duplicates':>{dup_w}}"
        f"{'Dup Rate':>{rate_w}}"
        f"{'Avg Len':>{len_w}}"
    )
    logger.info(header)
    logger.info("-" * 100)
    
    for i, stat in enumerate(stats, 1):
        # Mark major categories
        marker = "[M]" if stat.percentage >= 10.0 else ("[S]" if stat.percentage >= 5.0 else "   ")
        
        count_str = f"{stat.total_count:,}"
        pct_str = f"{stat.percentage:.1f}%"
        dup_str = f"{stat.duplicate_count:,}"
        rate_str = f"{stat.duplicate_rate:.1f}%"
        len_str = f"{stat.avg_combined_length:.1f}w"
        
        row = (
            f"{marker:>{marker_w}}"
            f"{i:<{rank_w}}"
            f"{stat.category:<{cat_w}}"
            f"{count_str:>{count_w}}"
            f"{pct_str:>{pct_w}}"
            f"{dup_str:>{dup_w}}"
            f"{rate_str:>{rate_w}}"
            f"{len_str:>{len_w}}"
        )
        logger.info(row)
    
    logger.info("-" * 100)
    logger.info("")
    
    # Key insights
    significant_cats = [s for s in stats if 1.0 <= s.percentage < 5.0]
    rare_cats = [s for s in stats if s.percentage < 1.0]
    
    logger.info("Category Groups:")
    logger.info(f"  [M] Major (>=10%):      {len([s for s in stats if s.percentage >= 10.0])} categories, "
               f"covering {sum(s.percentage for s in stats if s.percentage >= 10.0):.1f}% of data")
    logger.info(f"  [S] Significant (5-10%): {len([s for s in stats if 5.0 <= s.percentage < 10.0])} categories, "
               f"covering {sum(s.percentage for s in stats if 5.0 <= s.percentage < 10.0):.1f}% of data")
    logger.info(f"      Minor (1-5%):      {len(significant_cats)} categories, "
               f"covering {sum(s.percentage for s in significant_cats):.1f}% of data")
    logger.info(f"      Rare (<1%):        {len(rare_cats)} categories, "
               f"covering {sum(s.percentage for s in rare_cats):.1f}% of data")
    logger.info("")
    
    # Duplicate rate analysis
    high_dup_cats = [s for s in stats if s.duplicate_rate > overall_dup_rate * 1.15 and s.percentage >= 1.0]
    low_dup_cats = [s for s in stats if s.duplicate_rate < overall_dup_rate * 0.85 and s.percentage >= 1.0]
    
    if high_dup_cats:
        logger.info(f"High Duplicate Rate (>{overall_dup_rate*1.15:.1f}%):")
        for stat in high_dup_cats:
            logger.info(f"  • {stat.category:<18} {stat.duplicate_rate:>5.1f}% duplicates ({stat.total_count:,} samples)")
        logger.info("")
    
    if low_dup_cats:
        logger.info(f"Low Duplicate Rate (<{overall_dup_rate*0.85:.1f}%):")
        for stat in low_dup_cats:
            logger.info(f"  • {stat.category:<18} {stat.duplicate_rate:>5.1f}% duplicates ({stat.total_count:,} samples)")
        logger.info("")
    
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
