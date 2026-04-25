#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assay — Tkinter all-in-one dataset inspection GUI
Tabs:
  1) Inspect  (columns, random examples, token stats, math density)
  2) Slice    (uniform random slice → Parquet)
  3) Token Filter (token-length filter with local tokenizer)

Requires: pandas, numpy, (optional) pyarrow, (optional) tabulate, transformers
"""

import os
import sys
import re
import io
import math
import json
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import numpy as np

# Optional deps
try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except Exception:
    HAVE_PYARROW = False

try:
    from tabulate import tabulate
    HAVE_TABULATE = True
except Exception:
    HAVE_TABULATE = False

try:
    from transformers import AutoTokenizer
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False


# =========================
# Settings persistence
# =========================

SETTINGS_PATH = Path.home() / ".assay.json"

def load_settings() -> dict:
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_settings(data: dict):
    try:
        existing = load_settings()
        existing.update(data)
        with SETTINGS_PATH.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass


# =========================
# Tooltip helper
# =========================

class Tooltip:
    """Simple hover tooltip for any Tk widget."""
    def __init__(self, widget: tk.Widget, text: str, delay: int = 600):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._job = None
        self._tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._cancel)
        widget.bind("<ButtonPress>", self._cancel)

    def _schedule(self, _event=None):
        self._cancel()
        self._job = self.widget.after(self.delay, self._show)

    def _cancel(self, _event=None):
        if self._job:
            self.widget.after_cancel(self._job)
            self._job = None
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def _show(self):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=self.text, justify="left",
                       background="#ffffe0", relief="solid", borderwidth=1,
                       font=("Segoe UI", 9), wraplength=320, padx=6, pady=4)
        lbl.pack()



def load_dataframe(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".jsonl",):
        return pd.read_json(path, lines=True)
    if ext in (".json",):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data if isinstance(data, list) else [])
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {ext}. Use .parquet/.jsonl/.json/.csv")

def safe_tabulate(data, headers="keys", tablefmt="github", showindex=False, stralign="left"):
    if HAVE_TABULATE:
        return tabulate(data, headers=headers, tablefmt=tablefmt, showindex=showindex, stralign=stralign)
    # minimal fallback — handles list-of-dicts only
    if not data:
        return ""
    # Normalise: if rows are not dicts, fall back to a simple repr
    if not isinstance(data[0], dict):
        return "\n".join(str(row) for row in data)
    if isinstance(headers, str) and headers == "keys":
        cols = list(data[0].keys())
    elif isinstance(headers, (list, tuple)):
        cols = list(headers)
    else:
        cols = []
    lines = []
    if cols:
        lines.append(" | ".join(cols))
        lines.append("-+-".join("-"*len(c) for c in cols))
    for row in data:
        lines.append(" | ".join(str(row.get(c, "")) for c in cols))
    return "\n".join(lines)

class TextLogger:
    """Thread-safe logger to a read-only Tk Text widget.

    The widget is kept in state='disabled' so users cannot accidentally type
    into the console.  write() briefly sets state='normal', inserts text, then
    re-disables the widget — all from the main-thread via widget.after().
    """
    def __init__(self, widget: tk.Text):
        self.widget = widget
        self.lock = threading.Lock()

    def write(self, s: str):
        if not s:
            return
        with self.lock:
            def _insert():
                self.widget.config(state="normal")
                self.widget.insert("end", s)
                self.widget.see("end")
                self.widget.config(state="disabled")
            self.widget.after(0, _insert)

    def println(self, s: str = ""):
        self.write(s + "\n")

# ---------- tokenizer helpers ----------
def try_load_tokenizer(path: str, local_only: bool = True):
    if not HAVE_TRANSFORMERS:
        raise RuntimeError("transformers is not installed. pip install transformers")

    if not path:
        raise RuntimeError("Please set a tokenizer path (e.g., /path/to/model or 'microsoft/phi-2').")

    if local_only:
        return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True), f"{path} (local_only)"
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True), f"{path} (local)"
    except Exception:
        return AutoTokenizer.from_pretrained(path, use_fast=True), f"{path} (fallback online)"

# ---------- column detection ----------
COMMON_PAIRS: List[Tuple[str, str]] = [
    ("prompt", "output"), ("instruction", "response"), ("instruction", "output"),
    ("input", "output"), ("document", "summary"), ("text", "summary"),
    ("article", "title"), ("dialogue", "summary"), ("post", "tldr"),
    ("content", "headline"), ("question", "answer"), ("source", "target"),
    ("body", "title"), ("question", "generated_solution"), ("query", "answer"),
    ("problem", "solution"), ("question", "long_answer"),
]
SINGLE_COL_CANDIDATES = ["text", "title", "headline", "output", "response", "summary"]

def detect_columns(df: pd.DataFrame,
                   input_col: Optional[str],
                   output_col: Optional[str]) -> Tuple[Optional[str], Optional[str], bool]:
    cols_lower = {c.lower(): c for c in df.columns}
    if input_col and output_col:
        if input_col in df.columns and output_col in df.columns:
            return input_col, output_col, True
        if input_col.lower() in cols_lower and output_col.lower() in cols_lower:
            return cols_lower[input_col.lower()], cols_lower[output_col.lower()], True
    colset = set(map(str.lower, df.columns))
    for a, b in COMMON_PAIRS:
        if a in colset and b in colset:
            return cols_lower[a], cols_lower[b], True
    for s in SINGLE_COL_CANDIDATES:
        if s in colset:
            return cols_lower[s], None, False
    return None, None, False

def combine_text_row(row, in_col: Optional[str], out_col: Optional[str], sep: str) -> str:
    if in_col and out_col:
        a = "" if pd.isna(row[in_col]) else str(row[in_col])
        b = "" if pd.isna(row[out_col]) else str(row[out_col])
        return (a + (sep if (a and b) else "") + b).strip()
    if in_col:
        a = "" if pd.isna(row[in_col]) else str(row[in_col])
        return a.strip()
    return ""


# =========================
# Slice tab
# =========================

class RandomX:
    def __init__(self, seed_int: Optional[int] = None):
        import random
        if seed_int is None:
            seed_int = int.from_bytes(os.urandom(8), "big", signed=False)
        self.seed = seed_int
        self._r = random.Random(seed_int)
    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

class Reservoir:
    def __init__(self, k: int, rng: RandomX):
        self.k = k
        self.rng = rng
        self.sample: List[Dict] = []
        self.n_seen = 0
    def consider(self, item: Dict):
        self.n_seen += 1
        if len(self.sample) < self.k:
            self.sample.append(item)
        else:
            # Knuth Algorithm R: pick a 1-based random index.
            # If it falls within [1, k] replace that slot.  This is correct;
            # the 1-based arithmetic is intentional, not an off-by-one error.
            j = self.rng.randint(1, self.n_seen)
            if j <= self.k:
                self.sample[j - 1] = item

def stream_parquet(path: Path, limit: int, rng: RandomX, stop_flag_ref=None) -> pd.DataFrame:
    if not HAVE_PYARROW:
        df = pd.read_parquet(path)
        return df.sample(n=min(limit, len(df)), random_state=rng.seed).reset_index(drop=True)
    pf = pq.ParquetFile(str(path))
    res = Reservoir(limit, rng)
    for rg_idx in range(pf.num_row_groups):
        # Check stop flag between row groups so "Stop" works on large files
        if stop_flag_ref is not None and stop_flag_ref[0]:
            break
        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        for rec in df.to_dict(orient="records"):
            res.consider(rec)
    return pd.DataFrame(res.sample)

def stream_jsonl(path: Path, limit: int, rng: RandomX) -> pd.DataFrame:
    res = Reservoir(limit, rng)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    res.consider(obj)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(res.sample)

def load_json_array(path: Path, limit: int, rng: RandomX) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data if isinstance(data, list) else [])
    if len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=rng.seed).reset_index(drop=True)

def stream_csv(path: Path, limit: int, rng: RandomX, chunksize: int = 100_000) -> pd.DataFrame:
    res = Reservoir(limit, rng)
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for rec in chunk.to_dict(orient="records"):
            res.consider(rec)
    return pd.DataFrame(res.sample)

def infer_slice_out(in_path: Path, out_dir: Optional[Path], slice_size: int) -> Path:
    out_dir = out_dir if out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only use the short 'Nk' form for exact multiples of 1000 — otherwise use the raw number
    # to avoid misleading filenames like '1500k' implying 1.5 million rows.
    if slice_size % 1000 == 0:
        suffix = f"{slice_size // 1000}k"
    else:
        suffix = str(slice_size)
    return out_dir / f"{in_path.stem}_slice{suffix}.parquet"

def save_parquet(df: pd.DataFrame, out_path: Path):
    if HAVE_PYARROW:
        df.to_parquet(out_path, index=False, engine="pyarrow")
    else:
        df.to_parquet(out_path, index=False)

class SliceTab(ttk.Frame):
    def __init__(self, master, logger: TextLogger, shared_in_var: tk.StringVar):
        super().__init__(master, padding=12)
        self.logger = logger
        self.in_path = shared_in_var          # <-- shared across tabs
        self.out_dir = tk.StringVar()
        self.slice_size = tk.IntVar(value=20_000)
        self._tab_index = None  # set by Workbench after adding to notebook

        # Restore persisted settings
        _s = load_settings()
        if _s.get("slice_out_dir"):
            self.out_dir.set(_s["slice_out_dir"])
        if _s.get("slice_size"):
            try:
                self.slice_size.set(int(_s["slice_size"]))
            except Exception:
                pass

        self._build()
        self._worker = None
        self._stop_flag = [False]  # mutable container so stream_parquet can read it

        # Disable Stop at startup
        self.btn_quit.config(state="disabled")

        # Auto-populate out_dir when shared in_path changes
        shared_in_var.trace_add("write", self._on_in_path_changed)

    def _on_in_path_changed(self, *_):
        p = self.in_path.get().strip()
        if p and not self.out_dir.get().strip():
            self.out_dir.set(str(Path(p).parent))

    def _build(self):
        r1 = ttk.Frame(self); r1.pack(fill="x", pady=(0,8))
        ttk.Label(r1, text="Input file (.parquet/.jsonl/.json/.csv):").pack(side="left")
        ttk.Entry(r1, textvariable=self.in_path).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r1, text="Browse…", command=self._browse_in).pack(side="left")

        r2 = ttk.Frame(self); r2.pack(fill="x", pady=(0,8))
        ttk.Label(r2, text="Output directory (optional):").pack(side="left")
        ttk.Entry(r2, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r2, text="Browse…", command=self._browse_out).pack(side="left")

        r3 = ttk.Frame(self); r3.pack(fill="x", pady=(0,8))
        ttk.Label(r3, text="Slice size (rows):").pack(side="left")
        sz_entry = ttk.Entry(r3, textvariable=self.slice_size, width=10); sz_entry.pack(side="left", padx=8)
        Tooltip(sz_entry, "Number of rows to randomly sample from the input file and write to the output Parquet slice.")
        self.btn_start = ttk.Button(r3, text="Start", command=self._start, style="Primary.TButton"); self.btn_start.pack(side="left")
        self.btn_quit = ttk.Button(r3, text="Stop", command=self._stop); self.btn_quit.pack(side="left", padx=8)
        self.prog = ttk.Progressbar(r3, mode="indeterminate"); self.prog.pack(side="right", fill="x", expand=True)

    def _browse_in(self):
        path = filedialog.askopenfilename(title="Select input file",
                                          filetypes=[("Supported","*.parquet *.jsonl *.json *.csv"),
                                                     ("Parquet","*.parquet"),("JSONL","*.jsonl"),
                                                     ("JSON","*.json"),("CSV","*.csv"),("All","*.*")])
        if path:
            self.in_path.set(path)
            if not self.out_dir.get().strip():
                self.out_dir.set(str(Path(path).parent))

    def _browse_out(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.out_dir.set(path)

    def _set_running(self, running: bool):
        state = "disabled" if running else "normal"
        self.btn_start.config(state=state)
        self.btn_quit.config(state=("normal" if running else "disabled"))
        if running:
            self.prog.config(mode="indeterminate"); self.prog.start(12)
        else:
            self.prog.stop(); self.prog.config(mode="determinate", value=100)
        # Update notebook tab label to show running indicator
        self._set_tab_running(running)

    def _set_tab_running(self, running: bool):
        try:
            nb = self.master
            idx = nb.index(self)
            current = nb.tab(idx, "text")
            if running and not current.startswith("⏳"):
                nb.tab(idx, text="⏳ " + current)
            elif not running and current.startswith("⏳ "):
                nb.tab(idx, text=current[3:])
        except Exception:
            pass

    def _stop(self):
        self._stop_flag[0] = True
        self.logger.println("[Slice] Stop requested…")

    def _start(self):
        path = self.in_path.get().strip()
        if not path:
            messagebox.showerror("Missing input", "Choose an input file."); return
        try:
            k = int(self.slice_size.get())
            if k <= 0: raise ValueError
        except Exception:
            messagebox.showerror("Invalid slice size", "Slice size must be a positive integer."); return

        self._stop_flag[0] = False
        self._set_running(True)
        self.logger.println("[Slice] Starting…")
        # Persist settings
        save_settings({"slice_out_dir": self.out_dir.get().strip(), "slice_size": k})

        def worker():
            try:
                in_path = Path(path)
                if not in_path.exists():
                    raise FileNotFoundError(in_path)
                rng = RandomX()
                self.logger.println(f"[Slice] RNG seed: {rng.seed}")
                ext = in_path.suffix.lower()
                if ext == ".parquet":
                    df = stream_parquet(in_path, k, rng, stop_flag_ref=self._stop_flag)
                elif ext == ".jsonl":
                    df = stream_jsonl(in_path, k, rng)
                elif ext == ".json":
                    self.logger.println("[Slice] Warning: .json arrays load fully; prefer JSONL/Parquet for huge files.")
                    df = load_json_array(in_path, k, rng)
                elif ext == ".csv":
                    df = stream_csv(in_path, k, rng)
                else:
                    raise ValueError(f"Unsupported input type: {ext}")

                if self._stop_flag[0]:
                    self.logger.println("[Slice] Cancelled."); return

                if len(df) > k:
                    df = df.sample(n=k, random_state=rng.seed).reset_index(drop=True)
                else:
                    df = df.reset_index(drop=True)

                out_dir = Path(self.out_dir.get().strip()) if self.out_dir.get().strip() else None
                out_path = infer_slice_out(in_path, out_dir, k)
                save_parquet(df, out_path)
                msg = f"Wrote {len(df):,} rows → {out_path}"
                self.logger.println(f"[Slice] {msg}")
                self.logger.println("[Slice] Done.")
                # Update status bar from main thread
                self.after(0, lambda: self.winfo_toplevel()._set_status(f"✓ Slice: {msg}"))
            except Exception as e:
                self.logger.println(f"[Slice][ERROR] {e}")
                self.after(0, lambda: self.winfo_toplevel()._set_status(f"✗ Slice error: {e}"))
            finally:
                self._set_running(False)

        threading.Thread(target=worker, daemon=True).start()


# =========================
# Filter tab
# =========================

def batched_token_lengths(texts: List[str], tokenizer, batch_size: int, progress_cb=None) -> List[int]:
    lengths = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
        if progress_cb:
            progress_cb(min(i + len(batch), total), total)
    return lengths

class FilterTab(ttk.Frame):
    def __init__(self, master, logger: TextLogger, shared_in_var: tk.StringVar):
        super().__init__(master, padding=12)
        self.logger = logger

        self.in_path = shared_in_var       # <-- shared across tabs
        self.out_dir = tk.StringVar()

        # Restore persisted settings
        _s = load_settings()
        self.tokenizer_path = tk.StringVar(value=_s.get("tokenizer_path", ""))
        self.local_only = tk.BooleanVar(value=True)

        self.max_tokens = tk.IntVar(value=int(_s.get("filter_max_tokens", 750)))
        self.input_col = tk.StringVar()
        self.output_col = tk.StringVar()
        # Store sep as the literal two characters \n so the Entry widget round-trips correctly;
        # the worker unescapes it to a real newline before use.
        self.sep = tk.StringVar(value=_s.get("filter_sep", "\\n"))
        self.batch_size = tk.IntVar(value=int(_s.get("filter_batch_size", 1024)))
        self.sample_limit = tk.IntVar(value=0)
        self.dry_run = tk.BooleanVar(value=False)

        self._build()
        self._worker = None
        self._stop_flag = False

        # Disable Stop at startup
        self.btn_stop.config(state="disabled")

        # Auto-populate out_dir when shared in_path changes
        shared_in_var.trace_add("write", self._on_in_path_changed)

    def _on_in_path_changed(self, *_):
        p = self.in_path.get().strip()
        if p and not self.out_dir.get().strip():
            self.out_dir.set(str(Path(p).parent))

    def _build(self):
        top = ttk.Frame(self); top.pack(fill="x", pady=(0,8))
        ttk.Label(top, text="Input file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.in_path).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(top, text="Browse…", command=self._browse_in).grid(row=0, column=2, sticky="w")
        ttk.Label(top, text="Output folder:").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Entry(top, textvariable=self.out_dir).grid(row=1, column=1, sticky="we", padx=6, pady=(6,0))
        ttk.Button(top, text="Browse…", command=self._browse_out).grid(row=1, column=2, sticky="w", pady=(6,0))
        top.columnconfigure(1, weight=1)

        tok = ttk.Labelframe(self, text="Tokenizer", padding=8); tok.pack(fill="x", pady=(0,8))
        ttk.Label(tok, text="Path or name:").grid(row=0, column=0, sticky="w")
        tok_entry = ttk.Entry(tok, textvariable=self.tokenizer_path, width=52)
        tok_entry.grid(row=0, column=1, sticky="we", padx=6)
        Tooltip(tok_entry, "Local directory of a HuggingFace tokenizer or a Hub model name (e.g. 'meta-llama/Llama-3.1-8B', 'microsoft/phi-2').")
        ttk.Button(tok, text="Browse…", command=self._browse_tok).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(tok, text="Local files only", variable=self.local_only).grid(row=0, column=3, sticky="w", padx=(8,0))
        tok.columnconfigure(1, weight=1)

        opt = ttk.Labelframe(self, text="Filter options", padding=8); opt.pack(fill="x", pady=(0,8))
        ttk.Label(opt, text="max_tokens ≤").grid(row=0, column=0, sticky="w")
        mt_entry = ttk.Entry(opt, textvariable=self.max_tokens, width=8); mt_entry.grid(row=0, column=1, sticky="w", padx=4)
        Tooltip(mt_entry, "Keep only rows whose combined token length is ≤ this value.")
        ttk.Label(opt, text="batch_size").grid(row=0, column=2, sticky="w")
        bs_entry = ttk.Entry(opt, textvariable=self.batch_size, width=8); bs_entry.grid(row=0, column=3, sticky="w", padx=4)
        Tooltip(bs_entry, "Number of examples to tokenize in one batch. Larger values are faster but use more RAM.")
        ttk.Label(opt, text="sample_limit (0=all)").grid(row=0, column=4, sticky="w")
        sl_entry = ttk.Entry(opt, textvariable=self.sample_limit, width=8); sl_entry.grid(row=0, column=5, sticky="w", padx=4)
        Tooltip(sl_entry, "Randomly subsample this many rows before filtering. Set to 0 to process the entire dataset.")
        ttk.Checkbutton(opt, text="dry_run (no write)", variable=self.dry_run).grid(row=0, column=6, sticky="w", padx=(8,0))

        cols = ttk.Labelframe(self, text="Columns", padding=8); cols.pack(fill="x", pady=(0,8))
        ttk.Label(cols, text="input_col").grid(row=0, column=0, sticky="w")
        ic_entry = ttk.Entry(cols, textvariable=self.input_col, width=18); ic_entry.grid(row=0, column=1, sticky="w", padx=4)
        Tooltip(ic_entry, "Column name for the input/prompt side. Leave blank to auto-detect.")
        ttk.Label(cols, text="output_col").grid(row=0, column=2, sticky="w")
        oc_entry = ttk.Entry(cols, textvariable=self.output_col, width=18); oc_entry.grid(row=0, column=3, sticky="w", padx=4)
        Tooltip(oc_entry, "Column name for the output/response side. Leave blank to auto-detect.")
        ttk.Label(cols, text="sep").grid(row=0, column=4, sticky="w")
        sep_entry = ttk.Entry(cols, textvariable=self.sep, width=10); sep_entry.grid(row=0, column=5, sticky="w", padx=4)
        Tooltip(sep_entry, r"String inserted between input and output when concatenating for tokenization. Use \n for a newline, \t for a tab.")

        act = ttk.Frame(self); act.pack(fill="x", pady=(0,8))
        self.btn_run = ttk.Button(act, text="Run Filter", command=self._start, style="Primary.TButton"); self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(act, text="Stop", command=self._stop); self.btn_stop.pack(side="left", padx=8)
        self.pbar = ttk.Progressbar(act, mode="determinate"); self.pbar.pack(side="right", fill="x", expand=True)

    def _browse_in(self):
        path = filedialog.askopenfilename(title="Select input dataset",
                                          filetypes=[("Supported","*.parquet *.jsonl *.json *.csv"),
                                                     ("Parquet","*.parquet"),("JSONL","*.jsonl"),
                                                     ("JSON","*.json"),("CSV","*.csv"),("All","*.*")])
        if path:
            self.in_path.set(path)
            self.out_dir.set(os.path.dirname(path))

    def _browse_out(self):
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.out_dir.set(path)

    def _browse_tok(self):
        path = filedialog.askdirectory(title="Select tokenizer directory (local)")
        if path:
            self.tokenizer_path.set(path)

    def _stop(self):
        self._stop_flag = True
        self.logger.println("[Filter] Stop requested…")

    def _set_progress(self, v, m=100):
        try:
            self.pbar.configure(maximum=m, value=v)
        except Exception:
            pass

    def _set_running(self, running: bool):
        self.btn_run.config(state=("disabled" if running else "normal"))
        self.btn_stop.config(state=("normal" if running else "disabled"))
        if running:
            self.pbar.configure(value=0, maximum=100)
        else:
            self.pbar.configure(value=100, maximum=100)
        self._set_tab_running(running)

    def _set_tab_running(self, running: bool):
        try:
            nb = self.master
            idx = nb.index(self)
            current = nb.tab(idx, "text")
            if running and not current.startswith("⏳"):
                nb.tab(idx, text="⏳ " + current)
            elif not running and current.startswith("⏳ "):
                nb.tab(idx, text=current[3:])
        except Exception:
            pass

    def _start(self):
        if hasattr(self, "_worker") and self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running."); return

        in_path = self.in_path.get().strip()
        out_dir = self.out_dir.get().strip()
        tok_path = self.tokenizer_path.get().strip()
        if not in_path:
            messagebox.showwarning("Input", "Please select an input file."); return
        if not os.path.isfile(in_path):
            messagebox.showerror("Not found", in_path); return
        if not out_dir:
            messagebox.showwarning("Output folder", "Choose an output folder."); return
        os.makedirs(out_dir, exist_ok=True)
        if not tok_path:
            messagebox.showwarning("Tokenizer", "Set your tokenizer path."); return

        self._stop_flag = False
        self._set_running(True)
        self.logger.println("[Filter] Starting…")

        # Persist settings
        save_settings({
            "tokenizer_path": tok_path,
            "filter_max_tokens": self.max_tokens.get(),
            "filter_batch_size": self.batch_size.get(),
            "filter_sep": self.sep.get(),
        })

        def worker():
            try:
                max_tokens = int(self.max_tokens.get())
                in_col = self.input_col.get().strip() or None
                out_col = self.output_col.get().strip() or None
                # Unescape sep: user types \n or \t as literal characters in the Entry widget
                raw_sep = self.sep.get()
                sep = raw_sep.replace("\\n", "\n").replace("\\t", "\t")
                batch_size = int(self.batch_size.get())
                sample_limit = int(self.sample_limit.get())
                dry_run = bool(self.dry_run.get())

                t0 = time.time()
                self.logger.println(f"[Filter] Loading tokenizer: {tok_path} (local_only={self.local_only.get()})")
                tokenizer, label = try_load_tokenizer(tok_path, local_only=self.local_only.get())
                self.logger.println(f"[Filter] Using tokenizer: {label}")

                if self._stop_flag: self.logger.println("[Filter] Cancelled."); return

                self.logger.println(f"[Filter] Reading dataset: {in_path}")
                df = load_dataframe(in_path)
                if sample_limit > 0:
                    self.logger.println(f"[Filter] sample_limit={sample_limit} → random sample of {sample_limit} rows")
                    df = df.sample(n=min(sample_limit, len(df)), random_state=42).reset_index(drop=True).copy()
                if df.empty:
                    self.logger.println("[Filter] Empty dataframe; nothing to do."); return

                det_in, det_out, is_pair = detect_columns(df, in_col, out_col)
                if det_in is None and det_out is None:
                    self.logger.println("[Filter][ERROR] Could not detect columns. Provide input/output explicitly.")
                    self.logger.println(f"Available columns: {list(df.columns)}")
                    return
                self.logger.println(f"[Filter] Mode: {'pair' if is_pair else 'single'} | input={det_in}" + (f" | output={det_out}" if det_out else ""))

                self.logger.println("[Filter] Composing texts…")
                combined = df.apply(lambda r: combine_text_row(r, det_in, det_out, sep), axis=1)

                self.logger.println("[Filter] Tokenizing (batched)…")
                self._set_progress(0, 100)
                def on_prog(done, total):
                    pct = int(100 * (done / max(total,1)))
                    self._set_progress(pct, 100)
                lengths = batched_token_lengths(combined.tolist(), tokenizer, batch_size, progress_cb=on_prog)
                df["_tokens"] = lengths

                if self._stop_flag: self.logger.println("[Filter] Cancelled."); return

                kept_mask = df["_tokens"] <= max_tokens
                kept = df[kept_mask].drop(columns=["_tokens"])
                total = len(df); kept_n = len(kept); dropped_n = total - kept_n
                pct = 100.0 * kept_n / max(total, 1)
                self.logger.println(f"\n[Filter] Retained {kept_n} / {total} rows (≤ {max_tokens}). Retention={pct:.2f}% | Dropped={dropped_n}")

                if dry_run:
                    self.logger.println("\n[Filter][dry_run] Skipping write. Head of kept:")
                    self.logger.println(str(kept.head(min(5, kept_n))))
                    self._set_progress(100,100); return

                base, ext = os.path.splitext(os.path.basename(in_path))
                ext = ext or ".parquet"
                out_file = f"{base}_{max_tokens}{ext}"
                out_path = os.path.join(out_dir, out_file)
                self.logger.println(f"[Filter] Writing: {out_path}")
                if ext == ".parquet":
                    kept.to_parquet(out_path, index=False)
                elif ext in (".jsonl", ".json"):
                    kept.to_json(out_path, orient="records", lines=True, force_ascii=False)
                elif ext == ".csv":
                    kept.to_csv(out_path, index=False)
                else:
                    kept.to_parquet(out_path, index=False)

                dt = time.time() - t0
                self._set_progress(100,100)
                msg = f"Retained {kept_n:,}/{total:,} rows → {out_path}  ({dt:.2f}s)"
                self.logger.println(f"[Filter] Done in {dt:.2f}s.")
                self.after(0, lambda: self.winfo_toplevel()._set_status(f"✓ Token Filter: {msg}"))
            except Exception as e:
                self.logger.println(f"[Filter][ERROR] {e}")
                self.after(0, lambda: self.winfo_toplevel()._set_status(f"✗ Token Filter error: {e}"))
            finally:
                self._set_running(False)

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()


# =========================
# Inspect tab
# =========================

def _pick_first_nonempty(row, cols):
    for c in cols:
        if c in row and pd.notna(row[c]):
            s = str(row[c]).strip()
            if s:
                return s
    return ""

def _clean_and_trunc(s, width):
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > width:
        return s[:width - 1] + "…"
    return s

def compose_example_text(row, columns):
    colset = set(columns)
    def get(names):
        for n in names:
            if n in colset:
                v = row.get(n)
                if pd.isna(v):
                    continue
                if isinstance(v, (list, dict)):
                    v = str(v)
                s = str(v).strip()
                if s:
                    return s
        return ""
    src = get(["prompt","question","query","input","instruction"])
    tgt = get(["output","chosen","answer","expected_answer","predicted_answer","generated_solution","long_answer","response","text"])
    if src or tgt:
        return f"{src}\n{tgt}".strip()
    single = get(["text","output","chosen","answer","expected_answer","predicted_answer","generated_solution","long_answer","prompt","input","question"])
    if single:
        return single
    parts = []
    for c in columns:
        v = row.get(c)
        if isinstance(v, (list, dict)):
            v = str(v)
        if pd.notna(v):
            s = str(v).strip()
            if s:
                parts.append(s)
        if len(parts) >= 4:
            break
    return "\n".join(parts).strip()

def count_tokens_any(text, tokenizer):
    try:
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
        if callable(tokenizer):
            out = tokenizer(text)
            if isinstance(out, dict) and "input_ids" in out:
                return len(out["input_ids"])
            if isinstance(out, (list, tuple)):
                return len(out)
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    except Exception:
        return 0

def quantiles(arr, qs=(0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    if len(arr) == 0:
        return {q: float("nan") for q in qs}
    vals = np.quantile(arr, qs)
    return {q: float(v) for q, v in zip(qs, vals)}

def percent(n, d):
    return 0.0 if d == 0 else 100.0 * (n / d)

class InspectTab(ttk.Frame):
    def __init__(self, master, logger: TextLogger, shared_in_var: tk.StringVar):
        super().__init__(master, padding=12)
        self.logger = logger

        self.in_path = shared_in_var     # <-- shared across tabs

        # Restore persisted settings
        _s = load_settings()
        self.tokenizer_path = tk.StringVar(value=_s.get("tokenizer_path", ""))
        self.local_only = tk.BooleanVar(value=True)
        self.preview_rows = tk.IntVar(value=100)
        self.random_examples = tk.IntVar(value=10)
        self.threshold = tk.IntVar(value=1000)
        self.analyze_rows = tk.IntVar(value=0)

        self._build()
        self._worker = None
        self._stop_flag = False

        # Disable Stop at startup
        self.btn_stop.config(state="disabled")

    def _build(self):
        top = ttk.Frame(self); top.pack(fill="x", pady=(0,8))
        ttk.Label(top, text="Input file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.in_path).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(top, text="Browse…", command=self._browse_in).grid(row=0, column=2, sticky="w")

        tok = ttk.Labelframe(self, text="Tokenizer", padding=8); tok.pack(fill="x", pady=(0,8))
        ttk.Label(tok, text="Path or name:").grid(row=0, column=0, sticky="w")
        tok_entry = ttk.Entry(tok, textvariable=self.tokenizer_path)
        tok_entry.grid(row=0, column=1, sticky="we", padx=6)
        Tooltip(tok_entry, "Local directory of a HuggingFace tokenizer or a Hub model name (e.g. 'meta-llama/Llama-3.1-8B', 'microsoft/phi-2').")
        ttk.Button(tok, text="Browse…", command=self._browse_tok).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(tok, text="Local files only", variable=self.local_only).grid(row=0, column=3, sticky="w", padx=(8,0))
        tok.columnconfigure(1, weight=1)

        opts = ttk.Labelframe(self, text="Analysis options", padding=8); opts.pack(fill="x", pady=(0,8))
        ttk.Label(opts, text="Preview rows").grid(row=0, column=0, sticky="w")
        pr_entry = ttk.Entry(opts, textvariable=self.preview_rows, width=8); pr_entry.grid(row=0, column=1, sticky="w", padx=4)
        Tooltip(pr_entry, "Number of rows to show in the compact preview table.")
        ttk.Label(opts, text="Random examples").grid(row=0, column=2, sticky="w")
        re_entry = ttk.Entry(opts, textvariable=self.random_examples, width=8); re_entry.grid(row=0, column=3, sticky="w", padx=4)
        Tooltip(re_entry, "Number of randomly chosen full examples to print to the console for manual review.")
        ttk.Label(opts, text="Token threshold (≤)").grid(row=0, column=4, sticky="w")
        thr_entry = ttk.Entry(opts, textvariable=self.threshold, width=8); thr_entry.grid(row=0, column=5, sticky="w", padx=4)
        Tooltip(thr_entry, "Report what percentage of examples and tokens fall at or below this length threshold.")
        ttk.Label(opts, text="Analyze rows (0=all)").grid(row=0, column=6, sticky="w")
        ar_entry = ttk.Entry(opts, textvariable=self.analyze_rows, width=8); ar_entry.grid(row=0, column=7, sticky="w", padx=4)
        Tooltip(ar_entry, "Limit analysis to the first N rows. Set to 0 to analyze the full dataset.")

        act = ttk.Frame(self); act.pack(fill="x", pady=(0,8))
        self.btn_run = ttk.Button(act, text="Run Analysis", command=self._start, style="Primary.TButton"); self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(act, text="Stop", command=self._stop); self.btn_stop.pack(side="left", padx=8)
        self.pbar = ttk.Progressbar(act, mode="indeterminate"); self.pbar.pack(side="right", fill="x", expand=True)

    def _browse_in(self):
        path = filedialog.askopenfilename(title="Select dataset",
                                          filetypes=[("Supported","*.parquet *.jsonl *.json *.csv"),
                                                     ("Parquet","*.parquet"),("JSONL","*.jsonl"),
                                                     ("JSON","*.json"),("CSV","*.csv"),("All","*.*")])
        if path:
            self.in_path.set(path)

    def _browse_tok(self):
        path = filedialog.askdirectory(title="Tokenizer directory (local)")
        if path:
            self.tokenizer_path.set(path)

    def _stop(self):
        self._stop_flag = True
        self.logger.println("[Inspect] Stop requested…")

    def _set_running(self, running: bool):
        self.btn_run.config(state=("disabled" if running else "normal"))
        self.btn_stop.config(state=("normal" if running else "disabled"))
        if running:
            self.pbar.config(mode="indeterminate"); self.pbar.start(12)
        else:
            self.pbar.stop(); self.pbar.config(mode="determinate", value=100)
        self._set_tab_running(running)

    def _set_tab_running(self, running: bool):
        try:
            nb = self.master
            idx = nb.index(self)
            current = nb.tab(idx, "text")
            if running and not current.startswith("⏳"):
                nb.tab(idx, text="⏳ " + current)
            elif not running and current.startswith("⏳ "):
                nb.tab(idx, text=current[3:])
        except Exception:
            pass

    def _start(self):
        if hasattr(self, "_worker") and self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Analysis already running."); return
        path = self.in_path.get().strip()
        if not path:
            messagebox.showerror("Missing input", "Choose a file to inspect."); return
        tok = self.tokenizer_path.get().strip()
        if not tok:
            messagebox.showerror("Tokenizer", "Set a tokenizer path/name."); return

        self._stop_flag = False
        self._set_running(True)
        self.logger.println("[Inspect] Starting…")

        # Persist tokenizer path so other tabs pick it up too
        save_settings({"tokenizer_path": tok})

        def worker():
            try:
                df = load_dataframe(path)
                if df.empty:
                    self.logger.println("[Inspect] Empty dataframe."); return

                df.columns = [str(c).strip() for c in df.columns]

                text_blob = " ".join(map(str, df.values.flatten()))
                # Matches: $...$ inline math, \latexcmd, and common math unicode symbols
                math_matches = re.findall(r"(\$.*?\$|\\[a-zA-Z]+|[∑π∞√≤≥^_])", text_blob)
                math_density = len(math_matches) / max(len(text_blob), 1)
                math_percent = math_density * 100.0

                self.logger.println("Columns in file:")
                self.logger.println(str(list(df.columns)) + "\\n")

                tokenizer, label = try_load_tokenizer(tok, local_only=self.local_only.get())
                self.logger.println(f"Using tokenizer: {label}")

                analyze_rows = int(self.analyze_rows.get())
                if analyze_rows > 0 and len(df) > analyze_rows:
                    df = df.head(analyze_rows).copy()
                    self.logger.println(f"[Inspect] analyze_rows={analyze_rows} → limiting to head({analyze_rows})")

                def _compose_and_len(row):
                    txt = compose_example_text(row, df.columns)
                    return count_tokens_any(txt, tokenizer)

                self.logger.println("[Inspect] Computing token lengths…")
                token_lengths = df.apply(_compose_and_len, axis=1).astype(int)
                n_examples = int(token_lengths.shape[0])
                total_tokens = int(token_lengths.sum())
                mean_len = float(token_lengths.mean()) if n_examples else float("nan")
                std_len = float(token_lengths.std(ddof=0)) if n_examples else float("nan")
                min_len = int(token_lengths.min()) if n_examples else 0
                max_len = int(token_lengths.max()) if n_examples else 0
                qs = quantiles(token_lengths.values)

                thr = int(self.threshold.get())
                n_le_thr = int((token_lengths <= thr).sum())
                pct_examples_le_thr = percent(n_le_thr, n_examples)
                tokens_le_thr = int(token_lengths[token_lengths <= thr].sum())
                pct_tokens_le_thr = percent(tokens_le_thr, total_tokens)

                bins = [
                    (0, 256), (257, 512), (513, 750), (751, 1024), (1025, 1200),
                    (1201, 1536), (1537, 1700), (1701, 2048), (2049, 4096), (4097, math.inf),
                ]
                bin_rows = []
                for lo, hi in bins:
                    if math.isinf(hi):
                        mask = token_lengths >= lo
                        name = f"{lo}+"
                    else:
                        mask = (token_lengths >= lo) & (token_lengths <= hi)
                        name = f"{lo}-{hi}"
                    cnt = int(mask.sum()); pct = percent(cnt, n_examples)
                    tok_sum = int(token_lengths[mask].sum()); tok_pct = percent(tok_sum, total_tokens)
                    bin_rows.append({"bin (tokens)": name, "rows": cnt, "rows %": f"{pct:.2f}%", "tokens": tok_sum, "tokens %": f"{tok_pct:.2f}%"})

                preview_rows = max(1, int(self.preview_rows.get()))
                src_cols = ["prompt","question","query","input","instruction","text"]
                tgt_cols = ["output","chosen","answer","expected_answer","predicted_answer",
                            "generated_solution","long_answer","response","best_answer"]
                meta_cols = [c for c in ["is_correct","generation_type","dataset","error_message"] if c in df.columns]

                preview = []
                for i, (_, row) in enumerate(df.head(preview_rows).iterrows(), 1):
                    src = _pick_first_nonempty(row, src_cols)
                    tgt = _pick_first_nonempty(row, tgt_cols)
                    rec = {"#": i, "src": _clean_and_trunc(src, 120), "tgt": _clean_and_trunc(tgt, 120)}
                    for m in meta_cols:
                        rec[m] = _clean_and_trunc(row.get(m, ""), 40)
                    preview.append(rec)

                self.logger.println("\\n--------------------------")
                self.logger.println(f"Math density: {math_percent:.2f}%")
                self.logger.println("--------------------------\\n")

                stats_table = [
                    {"metric": "examples", "value": n_examples},
                    {"metric": "total_tokens", "value": total_tokens},
                    {"metric": "mean", "value": f"{mean_len:.2f}"},
                    {"metric": "std", "value": f"{std_len:.2f}"},
                    {"metric": "min", "value": min_len},
                    {"metric": "p25", "value": f"{qs.get(0.25, float('nan')):.2f}"},
                    {"metric": "p50 (median)", "value": f"{qs.get(0.5, float('nan')):.2f}"},
                    {"metric": "p75", "value": f"{qs.get(0.75, float('nan')):.2f}"},
                    {"metric": "p90", "value": f"{qs.get(0.9, float('nan')):.2f}"},
                    {"metric": "p95", "value": f"{qs.get(0.95, float('nan')):.2f}"},
                    {"metric": "p99", "value": f"{qs.get(0.99, float('nan')):.2f}"},
                    {"metric": "max", "value": max_len},
                ]
                self.logger.println("Token length breakdown")
                self.logger.println(safe_tabulate(stats_table, headers="keys", tablefmt="github", showindex=False, stralign="left") + "\\n")

                self.logger.println(f"Examples ≤ {thr}: {pct_examples_le_thr:.2f}%  ({n_le_thr}/{n_examples})")
                self.logger.println(f"TOTAL TOKENS from examples ≤ {thr}: {pct_tokens_le_thr:.2f}%  ({tokens_le_thr}/{total_tokens})\\n")

                self.logger.println(safe_tabulate(bin_rows, headers="keys", tablefmt="github", showindex=False, stralign="left") + "\\n")

                if preview:
                    self.logger.println("Compact preview:\\n")
                    self.logger.println(safe_tabulate(preview, headers="keys", tablefmt="github", showindex=False, stralign="left") + "\\n")

                # Optional: random full examples (kept; shows in log panel, not CLI)
                rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
                k = min(int(self.random_examples.get()), n_examples)
                if n_examples and k > 0:
                    random_indices = rng.choice(n_examples, size=k, replace=False).tolist()
                    self.logger.println("Random full examples:\\n")
                    for j, i in enumerate(random_indices, 1):
                        row = df.iloc[int(i)]
                        text = compose_example_text(row, df.columns)
                        tok_len = count_tokens_any(text, tokenizer)
                        header = f"Example {j}/{len(random_indices)} — row {int(i)} — {tok_len} tokens"
                        self.logger.println(header + "\\n" + "-" * len(header))
                        self.logger.println(text + "\\n")

                self.logger.println("[Inspect] Done.")
                self.after(0, lambda: self.winfo_toplevel()._set_status(
                    f"✓ Inspect: {n_examples:,} rows, {total_tokens:,} tokens, mean {mean_len:.1f} tok/row"))
            except Exception as e:
                self.logger.println(f"[Inspect][ERROR] {e}")
                self.after(0, lambda: self.winfo_toplevel()._set_status(f"✗ Inspect error: {e}"))
            finally:
                self._set_running(False)

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()


# =========================
# Main App
# =========================

class Workbench(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Assay — Inspect • Slice • Token Filter")
        self.geometry("1200x860")
        self.minsize(1000, 680)

        self._apply_theme()

        # Shared input path across tabs
        self.shared_in_path = tk.StringVar()

        # Layout: PanedWindow fills window; status bar pinned at bottom
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)   # paned window expands
        self.rowconfigure(1, weight=0)   # status bar fixed

        # PanedWindow — vertical split lets user drag the notebook/console divider
        paned = tk.PanedWindow(self, orient="vertical", sashrelief="raised", sashwidth=6)
        paned.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 0))

        # ---- Notebook (top pane) ----
        self.nb = ttk.Notebook(paned)
        paned.add(self.nb, minsize=200)

        # ---- Console (bottom pane) ----
        console_outer = ttk.Labelframe(paned, text="Console", padding=(8, 6, 8, 8))
        paned.add(console_outer, minsize=120)

        # Tabs (order: Inspect first)
        self.tab_inspect = InspectTab(self.nb, self._get_logger_proxy(), self.shared_in_path)
        self.tab_slice   = SliceTab(self.nb,   self._get_logger_proxy(), self.shared_in_path)
        self.tab_filter  = FilterTab(self.nb,  self._get_logger_proxy(), self.shared_in_path)
        self.nb.add(self.tab_inspect, text="Inspect")
        self.nb.add(self.tab_slice,   text="Slice")
        self.nb.add(self.tab_filter,  text="Token Filter")   # renamed from "Filter"

        # Console layout
        console_outer.columnconfigure(0, weight=1)
        console_outer.rowconfigure(1, weight=1)

        # Console toolbar (Copy/Clear on the right)
        toolbar = ttk.Frame(console_outer)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="e", pady=(0,6))
        ttk.Button(toolbar, text="Copy", command=self._copy_console).pack(side="right", padx=(6,0))
        ttk.Button(toolbar, text="Clear", command=self._clear_console).pack(side="right")

        # Read-only Text area with scroll
        self.log = tk.Text(console_outer, wrap="word", height=14, font=("Consolas", 10),
                           state="disabled")
        self.log.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(console_outer, command=self.log.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.log.config(yscrollcommand=scroll.set)

        # Status bar
        self._status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self._status_var, anchor="w",
                               relief="sunken", padding=(8, 2))
        status_bar.grid(row=1, column=0, sticky="we", padx=0, pady=0)

        # Now the real logger is ready — tabs will use it via the proxy
        self.logger = TextLogger(self.log)
        self.nb.select(self.tab_inspect)  # ensure Inspect shows first

    def _set_status(self, msg: str):
        """Update the status bar label (call from any thread via .after())."""
        self._status_var.set(msg)

    # --- helpers ---
    def _get_logger_proxy(self) -> TextLogger:
        """Provide tabs with a proxy that forwards to the real logger once it's created.

        Implements both .write() and .println() to match the full TextLogger interface.
        """
        class _Proxy:
            def write(_self, s: str):
                try:
                    self.logger.write(s)
                except Exception:
                    pass
            def println(_self, s: str = ""):
                try:
                    self.logger.println(s)
                except Exception:
                    pass
        return _Proxy()

    def _copy_console(self):
        try:
            # Temporarily enable to allow get()
            self.log.config(state="normal")
            text = self.log.get("1.0", "end-1c")
            self.log.config(state="disabled")
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            pass

    def _clear_console(self):
        try:
            self.log.config(state="normal")
            self.log.delete("1.0", "end")
            self.log.config(state="disabled")
        except Exception:
            pass

    def _apply_theme(self):
        """Try Sun Valley (sv-ttk) if present; otherwise configure 'clam' with nicer spacing."""
        try:
            import sv_ttk  # type: ignore
            sv_ttk.set_theme("light")
            self.option_add("*Font", ("Segoe UI", 10))
            style = ttk.Style(self)
            style.configure("TNotebook.Tab", padding=(16, 10))
            style.configure("TLabelframe", padding=10)
            style.configure("TFrame", padding=6)
            style.configure("Primary.TButton", padding=(14, 8))
            return
        except Exception:
            pass

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Global font
        self.option_add("*Font", ("Segoe UI", 10))

        # Spacing & simple accent for primary actions
        style.configure("TNotebook.Tab", padding=(16, 10))
        style.configure("TLabelframe", padding=10)
        style.configure("TFrame", padding=6)
        style.configure("Primary.TButton", padding=(14, 8))

        # On 'clam', background/foreground tweaks (kept subtle to avoid platform issues)
        style.map(
            "Primary.TButton",
            background=[("!disabled", "#2563EB"), ("active", "#1D4ED8")],
            foreground=[("!disabled", "white")],
        )

def main():
    app = Workbench()
    app.mainloop()

if __name__ == "__main__":
    main()