# Workbench

A Tkinter GUI for inspecting, slicing, and filtering ML training datasets. Designed to sit alongside fine-tuning tools — check your data quality, measure token distributions, and carve out clean subsets before training.

## Features

- **Inspect** — load any dataset (Parquet, JSONL, JSON, CSV), view columns, compute token length statistics (mean, percentiles, distribution bins), measure math/LaTeX density, and browse random full examples.
- **Slice** — reservoir-sample a uniform random subset of any size from large datasets and export to Parquet. Streaming row-group processing means you can slice multi-GB Parquet files without loading them into memory.
- **Token Filter** — filter rows by token count using any HuggingFace tokenizer. Keep only examples ≤ N tokens, preview retention stats, and export the filtered result. Supports dry-run mode.

All tabs share a single input file path and auto-populate output directories. Settings (tokenizer path, slice size, filter params) persist to `~/.assay.json` across sessions.

## Requirements

- Python 3.10+
- `pandas`, `numpy`
- `pyarrow` (recommended — enables streaming Parquet processing)
- `transformers` (required for tokenizer-based features)
- `tabulate` (optional — prettier console tables)
- `sv-ttk` (optional — Sun Valley theme for a modern look)

```bash
pip install pandas numpy pyarrow transformers tabulate
```

## Usage

```bash
python workbench.py
```

The GUI opens with three tabs. Browse to a dataset file in any tab — the path is shared across all tabs.

### Inspect Tab

Select a dataset and tokenizer, then click **Run Analysis**. The console shows column names, token length statistics (p25/p50/p75/p90/p95/p99), a distribution histogram, math density score, and a compact row preview. Use "Random examples" to print N full examples for manual review.

### Slice Tab

Set a slice size (number of rows) and click **Start**. The tool reservoir-samples that many rows uniformly at random and writes a Parquet file. Useful for creating manageable subsets of large datasets.

### Token Filter Tab

Set a tokenizer and a max token threshold, then click **Run Filter**. Rows whose combined text exceeds the threshold are dropped. The filtered dataset is written to the output directory with the token limit appended to the filename (e.g., `dataset_750.parquet`).

## Column Detection

All tabs auto-detect common column pairs (prompt/output, instruction/response, question/answer, etc.) and single-column datasets (text, output, etc.). You can override detection by specifying column names manually.

## License

MIT
