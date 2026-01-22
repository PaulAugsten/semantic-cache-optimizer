# Semantic Caching MVP

A Python implementation of semantic caching using `gptcache` for a university project. This system decides when to return cached responses versus calling an LLM based on similarity thresholds, reducing costs and latency while maintaining answer correctness.

## 🎯 Objectives

- **Cost Reduction**: Minimize LLM API calls through intelligent caching
- **Latency Optimization**: Fast cache lookups vs. slow LLM generation
- **High Precision**: Ensure cached answers are correct (prevent wrong responses)

## 📁 Project Structure

Todo

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Install uv (Windows / PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your terminal, then verify installation
uv --version

# Create environment
uv venv --python 3.11

# Activate virtual environment
.venv\Scripts\activate

# If activating doesn't work initialy, execute this:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Run Demo

```powershell
python demo.py
```

### 3. Run Full Evaluation

```powershell
# Evaluate strategies
python evaluate.py

# Test Cache Manager
python test_cache_manager.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
seed: 42
max_eval_samples: 10000

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda"

cache:
  backend: "faiss"
  similarity_threshold: 0.85

llm:
  mode: "hf"
```

## Threshold Strategies

Todo

## Metrics

- **Precision**: Cache hits that are correct
- **Recall**: Duplicates successfully found  
- **False Positive Rate**: Incorrect matches (danger!)
- **Cache Hit Ratio**: Requests served from cache
- **Latency**: ms for cache vs. s for LLM

## API Tokens

For HuggingFace models, create a `.env` file:

```
HUGGINGFACE_API_TOKEN=your_token_here
```

## Autoren

Paul Augsten & Catharina Dümmen, WS 2025/26
