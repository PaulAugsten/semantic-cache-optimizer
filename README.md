# Adaptive Threshold Semantic Caching

A Python implementation of semantic caching with **adaptive, category-based thresholds** using GPTCache. This system intelligently decides when to return cached responses versus calling an LLM by classifying queries into categories (factual, subjective, comparison, etc.) and applying optimized similarity thresholds per category.

## Objectives

- **Cost Reduction**: Minimize LLM API calls through intelligent caching
- **Latency Optimization**: Fast cache lookups (ms) vs. slow LLM generation (seconds)
- **High Precision**: Prevent incorrect cached responses through category-aware thresholds
- **Adaptive Strategy**: Different thresholds for different query types (factual questions need higher precision than creative tasks)

## Project Structure

```
.
├── src/                              # Main implementation
│   ├── adaptive_threshold.py         # Core adaptive threshold logic
│   ├── __init__.py                   # Package exports
│   └── ADAPTIVE_THRESHOLD.md         # Technical documentation
│
├── examples/                         # Demo & test scripts
│   ├── demo.py                       # Interactive demonstration
│   └── test_adaptive_threshold.py    # Test suite (4 tests)
│
├── evaluation/                       # Benchmark & analysis
│   ├── run_experiments_extended.py   # Run 9 threshold configurations
│   ├── analyze_category_performance.py
│   ├── analyze_dataset_categories.py
│   └── results/                      # Evaluation outputs (JSON + Markdown)
│
├── config.yaml                       # Configuration file
├── pyproject.toml                    # Poetry dependencies
└── README.md                         # This file
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Poetry (recommended) or pip

### 2. Installation

```bash
# Clone repository
git clone https://github.com/PaulAugsten/semantic-cache-optimizer.git
cd implementation

# Install dependencies with Poetry
poetry install

# OR with pip
pip install -r requirements.txt
```

### 3. Run Demo

```bash
# Interactive demonstration
poetry run python examples/demo.py

# Run test suite
poetry run python examples/test_adaptive_threshold.py
```

### 4. Run Full Evaluation

```bash
# Run all 9 threshold configurations (40,430 samples)
poetry run python evaluation/run_experiments_extended.py

# With verbose error analysis (optional)
poetry run python evaluation/run_experiments_extended.py --verbose-errors

# Analyze category performance
poetry run python evaluation/analyze_category_performance.py
```

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
cache:
    backend: faiss # Vector store (faiss or sqlite)
    similarity_threshold: 0.8 # Base threshold
    top_k: 5 # Number of candidates to retrieve

embedding:
    model: sentence-transformers/all-MiniLM-L6-v2
    device: cpu # or "cuda" for GPU
    normalize: true
    pooling: mean

# Adaptive thresholds per category
adaptive_threshold:
    factual:
        base_threshold: 0.85 # High precision for facts
        length_adjustment: 0.003
        min_threshold: 0.70
        max_threshold: 0.95
    subjective:
        base_threshold: 0.82 # More lenient for opinions
        length_adjustment: 0.0015
        min_threshold: 0.65
        max_threshold: 0.90
    creative:
        base_threshold: 0.56 # Very lenient for creative tasks
        length_adjustment: 0.001
        min_threshold: 0.45
        max_threshold: 0.70
    # ... (see config.yaml for all 7 categories)
```

## Adaptive Threshold Strategy

### Query Categories

The system classifies queries into 7 categories with optimized thresholds:

| Category         | Base Threshold | Use Case                          | Example Query                          |
| ---------------- | -------------- | --------------------------------- | -------------------------------------- |
| **FACTUAL**      | 0.85           | Pure information seeking          | "What is the capital of France?"       |
| **SUBJECTIVE**   | 0.82           | Opinions, advice, recommendations | "How do I learn Python?"               |
| **COMPARISON**   | 0.84           | Direct comparisons                | "Python vs Java differences?"          |
| **MATHEMATICAL** | 0.88           | Calculations, formulas            | "What is 25 times 4?"                  |
| **CREATIVE**     | 0.56           | Story/poem generation             | "Write a poem about coding"            |
| **CODE**         | 0.86           | Implementation questions          | "Write a function to reverse a string" |
| **UNKNOWN**      | 0.80           | Fallback for unclassified queries | Generic questions                      |

### How It Works

1. **Query Classification**: Regex patterns identify query type
    - "What is..." → FACTUAL
    - "Do you think..." → SUBJECTIVE
    - "What's the best..." → SUBJECTIVE
    - "X vs Y" → COMPARISON
    - Math keywords → MATHEMATICAL

2. **Length Adjustment**: Longer queries get slightly lower thresholds

    ```python
    threshold = base_threshold + (length // 10) * length_adjustment
    threshold = clamp(threshold, min_threshold, max_threshold)
    ```

3. **Similarity Evaluation**: Compare query embeddings and apply category-specific threshold

### Why Category-Based Thresholds?

- **Factual questions** need high precision (wrong facts are harmful)
- **Subjective/creative** queries can be more lenient (slight variations acceptable)
- **Better F1 score** than fixed thresholds (see evaluation results)

## Evaluation Results

Tested on **40,430 Quora question pairs** (semantic duplicates):

| Configuration       | Precision | Recall    | F1 Score  | Cache Hit % |
| ------------------- | --------- | --------- | --------- | ----------- |
| Fixed 0.80          | 84.2%     | 58.3%     | 68.9%     | 14.2%       |
| **Adaptive (ours)** | **86.5%** | **62.1%** | **72.3%** | **15.8%**   |

_Full results in `evaluation/results/`_

### Key Findings

**+3.4% F1 improvement** over fixed threshold (0.80)  
**Better recall** without sacrificing precision  
**Category-aware**: CREATIVE queries cached more aggressively, FACTUAL more conservatively  
**Length adaptation** improves handling of long queries

## API Usage

```python
from src import AdaptiveSimilarityEvaluation, QueryCategory

# Initialize with config
config = {
    'cache': {'similarity_threshold': 0.80},
    'embedding': {'model': 'sentence-transformers/all-MiniLM-L6-v2'}
}

evaluator = AdaptiveSimilarityEvaluation(config)

# Classify a query
category = evaluator._classify_query("What is the capital of France?")
print(category)  # QueryCategory.FACTUAL

# Get adaptive threshold
threshold = evaluator.get_threshold_for_query("What is the capital of France?")
print(threshold)  # 0.85 (high precision for factual)

# Custom threshold overrides
custom_rules = {
    QueryCategory.FACTUAL: ThresholdRule(
        base_threshold=0.95,  # Even stricter
        length_adjustment=0.003,
        min_threshold=0.80,
        max_threshold=0.98
    )
}
evaluator = AdaptiveSimilarityEvaluation(config, threshold_overrides=custom_rules)
```

## Testing

```bash
# Run test suite (4 tests)
poetry run python examples/test_adaptive_threshold.py

# Tests cover:
# - Category classification (18/19 accuracy = 94.7%)
# - Threshold calculation with length adjustment
# - Similarity evaluation and cache decisions
# - Custom threshold overrides
```

## Metrics

- **Precision**: % of cache hits that are correct (avoid false positives)
- **Recall**: % of semantic duplicates successfully found
- **F1 Score**: Harmonic mean of precision and recall
- **Cache Hit Ratio**: % of requests served from cache
- **Latency**: Cache lookup (ms) vs. LLM generation (seconds)

## API Tokens

For HuggingFace models (if using LLM mode), create a `.env` file:

```bash
HUGGINGFACE_API_TOKEN=your_token_here
```

## Development

### Running Experiments

```bash
# Full evaluation with 9 configurations
poetry run python evaluation/run_experiments_extended.py

# Analyze per-category performance
poetry run python evaluation/analyze_category_performance.py

# Analyze dataset category distribution
poetry run python evaluation/analyze_dataset_categories.py
```

### Code Structure

- `src/adaptive_threshold.py`: Core implementation (420 lines)
    - `AdaptiveSimilarityEvaluation`: Main evaluator class
    - `QueryCategory`: Enum with 7 categories
    - `ThresholdRule`: Dataclass for threshold config
    - Pattern-based classification logic
    - Length-adjusted threshold calculation

- `examples/`: Standalone demos and tests
- `evaluation/`: Benchmarking scripts and results

## Authors

Catharina Dümmen & Paul Augsten
