# Keigo Master: Japanese Formality Conversion Corpus

This repository directory contains the parallel dataset designed for training, validating, and evaluating the **Keigo Master** multi-agent Japanese formality conversion pipeline. The dataset bridges the gap between informal vernacular/casual Japanese and context-appropriate formal Japanese (Keigo).

---

## Directory Structure

```text
corpus-dataset/
├── raw/                      # Raw, uncleaned casual-formal sentence pairs
│   └── raw_casual_pairs.csv
├── processed/                # Unified, schema-validated full dataset
│   └── full_keigo_corpus.jsonl
├── splits/                   # Split files for model pipelines and evaluation
│   ├── train.jsonl           # Training set (~70%)
│   ├── val.jsonl             # Validation set (~15%)
│   └── test.jsonl            # Frozen gold test set (~15%)
├── scripts/                  # Data engineering and validation utilities
│   ├── preprocess.py         # Text cleaning and UTF-8 / Kanji normalization
│   ├── validate_schema.py    # Schema verification script
│   └── split_dataset.py      # Stratified train/val/test data splitter
├── annotation_guidelines.md  # Detailed tagging guidelines for annotators
└── README.md                 # Dataset documentation and taxonomy