# GPTCache Extensions - Software Quality Project

Experimentelle Cache-Strategien für GPTCache mit Benchmark-Framework.

## Projekt-Übersicht

Vergleich von vier Cache-Strategien:

1. **Baseline** - Standard GPTCache (Threshold 0.8)
2. **Adaptive Threshold** - Dynamische Thresholds je nach Query-Typ
3. **Cache Aging** - Zeitbasiertes Decay für Einträge
4. **Partitioned Cache** - Separate Caches für verschiedene Query-Typen

## Struktur

```
implementation/
├── experiments/           # Cache-Strategien
├── evaluation/            # Benchmark & Metriken
│   ├── cache_benchmark.py
│   └── data/              # Datasets
└── GPTCache/              # Original GPTCache
```

## Quick Start

```bash
# Installation
poetry install

# Benchmark ausführen
poetry run python evaluation/cache_benchmark.py
```

Konfiguration in `evaluation/cache_benchmark.py`:

-   `MAX_CONVERSATIONS` - Anzahl Queries

Details siehe `evaluation/README.md` und `experiments/README.md`

## Autoren

Paul Augsten & Catharina Dümmen, WS 2025/26
