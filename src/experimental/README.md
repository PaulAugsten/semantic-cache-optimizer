# Experimental Code

This directory contains code that was developed during the research phase but is **not part of the main evaluation**.

## Contents

### `cache_manager.py`

Unified cache manager supporting multiple threshold strategies. This was designed to allow easy switching between different strategies, but in our evaluation we focus exclusively on the adaptive threshold approach.

### `preprocessing.py`

Text preprocessing utilities (word counting, etc.) used by some experimental strategies.

### Alternative Threshold Strategies

Alternative threshold strategies that were explored:

- **`fixed_threshold.py`**: Simple fixed threshold (e.g., 0.9 for all queries)
- **`length_based.py`**: Adjusts threshold based on query length
- **`density_based.py`**: Uses embedding space density for threshold adjustment
- **`score_gap.py`**: Uses gap between top-2 scores for decision making

## Why Keep This?

These experimental strategies are preserved for:

1. **Future research**: Potential comparative studies
2. **Baseline comparisons**: Understanding why adaptive approach is better
3. **Educational purposes**: Showing the evolution of the approach
4. **Reproducibility**: Complete record of development process

## Status

**Note:** These modules are not tested or evaluated in the current project.

The main evaluation focuses solely on the adaptive threshold strategy in `src/adaptive_threshold.py`.

For the evaluated approach, see:

- `src/adaptive_threshold.py`
- `evaluation/run_experiments_extended.py`
- `evaluation/analyze_category_performance.py`

