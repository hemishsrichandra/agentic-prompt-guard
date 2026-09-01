"""Evaluate the generalised heuristic detector on train_dataset.csv.

Workflow
--------
1. Load ``train_dataset.csv`` (canonical schema).
2. Deduplicate by prompt string — keeps one representative row per unique
   prompt text.  This avoids artificially inflating metrics on repeated rows
   from the template-based generator.
3. Stratified 70 / 15 / 15 train / val / test split.
4. Evaluate the *heuristic* tier (Tier 3 — no LLM, no Ollama required) on
   each split and report accuracy, precision, recall, F1, and confusion
   matrix counts.
5. Optionally save the three split CSVs to ``data/``.

Usage
-----
    # Full evaluation on ~102k deduped rows (may take a few minutes)
    python scripts/analyze_dataset.py

    # Fast smoke-test on 2 000 randomly-sampled unique prompts
    python scripts/analyze_dataset.py --sample 2000

    # Persist split CSVs alongside the evaluation
    python scripts/analyze_dataset.py --save-splits

    # Use a different dataset path
    python scripts/analyze_dataset.py --dataset path/to/other.csv
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Make sure the project root is importable even when run from scripts/ ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from guard.datasets import (
    CANONICAL_LABEL,
    CANONICAL_PROMPT,
    LABEL_SAFE,
    LABEL_UNSAFE,
    dataset_stats,
    ingest_dataset,
)
from guard.ingestion import ingest
from guard.detector import ThreatDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = _ROOT / "train_dataset.csv"
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
# Test fraction is implicitly 1 - train - val = 0.15

SEED = 42


# ---------------------------------------------------------------------------
# Heuristic-only evaluation
# ---------------------------------------------------------------------------


def _classify_heuristic(prompt: str, detector: ThreatDetector) -> str:
    """Run the heuristic detector on a single prompt without LLM.

    Returns ``LABEL_UNSAFE`` when the heuristic flags the prompt, else
    ``LABEL_SAFE``.
    """
    ing = ingest(prompt)
    verdict = detector.detect(prompt, ing)
    return LABEL_UNSAFE if not verdict.is_safe else LABEL_SAFE


def evaluate_split(
    df: pd.DataFrame,
    split_name: str,
    detector: ThreatDetector,
    sample: Optional[int] = None,
) -> dict:
    """Evaluate *detector* on the rows of *df* and return metrics.

    Parameters
    ----------
    df:
        DataFrame with ``CANONICAL_PROMPT`` and ``CANONICAL_LABEL`` columns.
    split_name:
        Human-readable name shown in the printed report (``"train"``, etc.).
    detector:
        A :class:`~guard.detector.ThreatDetector` initialised without an LLM.
    sample:
        When not ``None``, evaluate only a random sample of this many rows
        (useful for quick smoke tests).
    """
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=SEED)

    total = len(df)
    tp = tn = fp = fn = 0
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        prompt = str(row[CANONICAL_PROMPT])
        gold = str(row[CANONICAL_LABEL])
        gold_unsafe = gold.strip().lower() != LABEL_SAFE.lower()

        pred = _classify_heuristic(prompt, detector)
        pred_unsafe = pred == LABEL_UNSAFE

        if gold_unsafe and pred_unsafe:
            tp += 1
        elif not gold_unsafe and not pred_unsafe:
            tn += 1
        elif not gold_unsafe and pred_unsafe:
            fp += 1
        else:
            fn += 1

        # Progress indicator every 5 000 rows
        if i % 5000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"    [{split_name}] {i:>6}/{total}  "
                  f"({i/total:.0%})  ~{remaining:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0

    return {
        "split": split_name,
        "total": n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _bar(value: float, width: int = 30) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_metrics(m: dict) -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'─' * 60}")
    print(f"  Split   : {m['split'].upper()}  ({m['total']:,} rows, {m['elapsed_s']}s)")
    print(f"{'─' * 60}")
    print(f"  Accuracy  {_bar(m['accuracy'])}  {m['accuracy']:.4f}")
    print(f"  Precision {_bar(m['precision'])}  {m['precision']:.4f}")
    print(f"  Recall    {_bar(m['recall'])}  {m['recall']:.4f}")
    print(f"  F1        {_bar(m['f1'])}  {m['f1']:.4f}")
    print(f"{'─' * 60}")
    print(f"  Confusion matrix (Unsafe = positive class)")
    print(f"    TP={m['tp']:>7,}  FN={m['fn']:>7,}   (missed unsafe)")
    print(f"    FP={m['fp']:>7,}  TN={m['tn']:>7,}   (false alarms / safe misclassified)")
    print(f"{'─' * 60}")


def print_dataset_overview(df: pd.DataFrame, title: str = "Dataset") -> None:
    stats = dataset_stats(df)
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    print(f"  Total rows      : {stats['rows']:,}")
    print(f"  Unique prompts  : {stats['unique_prompts']:,}")
    print(f"  Duplicates      : {stats['duplicate_prompts']:,}")
    print(f"  Avg prompt len  : {stats['avg_prompt_chars']} chars")
    print(f"  Max prompt len  : {stats['max_prompt_chars']} chars")
    if stats.get("has_labels"):
        dist = stats.get("label_distribution", {})
        for label, count in dist.items():
            pct = count / stats["rows"] * 100
            print(f"  {label:<22}: {count:>7,}  ({pct:.1f}%)")
    print(f"{'═' * 60}")


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with stratified label distribution."""
    safe_idx = df[df[CANONICAL_LABEL] == LABEL_SAFE].index.tolist()
    unsafe_idx = df[df[CANONICAL_LABEL] == LABEL_UNSAFE].index.tolist()

    r = random.Random(seed)
    r.shuffle(safe_idx)
    r.shuffle(unsafe_idx)

    def _cut(idx, train_f, val_f):
        n = len(idx)
        n_train = round(n * train_f)
        n_val = round(n * val_f)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    s_tr, s_v, s_te = _cut(safe_idx, train_frac, val_frac)
    u_tr, u_v, u_te = _cut(unsafe_idx, train_frac, val_frac)

    train_df = df.loc[s_tr + u_tr].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = df.loc[s_v + u_v].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = df.loc[s_te + u_te].sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the heuristic detector on train_dataset.csv "
            "with stratified train / val / test splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the labelled CSV dataset.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on N randomly sampled rows per split (fast mode).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=DEFAULT_TRAIN_FRAC,
        help="Fraction of deduped rows used for the training split.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Fraction of deduped rows used for the validation split.",
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Save train/val/test CSVs to the data/ directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        parser.error("--train-frac + --val-frac must be < 1.0")

    # ── 1. Load ─────────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.dataset}")
    t_load = time.time()
    df_raw = ingest_dataset(args.dataset)
    print(f"Loaded {len(df_raw):,} rows in {time.time() - t_load:.1f}s")
    print_dataset_overview(df_raw, "Raw dataset (before deduplication)")

    # ── 2. Deduplicate ───────────────────────────────────────────────────
    df = df_raw.drop_duplicates(subset=[CANONICAL_PROMPT], keep="first").reset_index(drop=True)
    print(f"\nAfter deduplication: {len(df):,} unique prompts "
          f"(removed {len(df_raw) - len(df):,} duplicates)")
    print_dataset_overview(df, "Deduped dataset")

    # ── 3. Split ─────────────────────────────────────────────────────────
    test_frac = round(1 - args.train_frac - args.val_frac, 4)
    print(f"\nSplitting: {args.train_frac:.0%} train / "
          f"{args.val_frac:.0%} val / "
          f"{test_frac:.0%} test  (seed={args.seed})")
    train_df, val_df, test_df = stratified_split(
        df, args.train_frac, args.val_frac, args.seed
    )
    for name, sdf in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = sdf[CANONICAL_LABEL].value_counts().to_dict()
        print(f"  {name:<6}: {len(sdf):>7,} rows  {dist}")

    if args.save_splits:
        out_dir = _ROOT / "data"
        out_dir.mkdir(exist_ok=True)
        train_df.to_csv(out_dir / "train_split.csv", index=False)
        val_df.to_csv(out_dir / "val_split.csv", index=False)
        test_df.to_csv(out_dir / "test_split.csv", index=False)
        df.to_csv(out_dir / "unique_prompts.csv", index=False)
        print(f"\nSplit CSVs saved to {out_dir}/")

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    # Instantiate the detector in heuristic-only mode (no LLM).
    detector = ThreatDetector(llm=None)

    print("\n" + "═" * 60)
    print("  HEURISTIC DETECTOR EVALUATION")
    print("  (Tier 3 — deterministic, no LLM required)")
    print()
    print("  Architecture note:")
    print("    Tier 1 — Ingestion (signatures + semantic similarity) — always")
    print("    Tier 2 — LLM (Ollama) — runs FIRST when available; takes priority")
    print("    Tier 3 — Heuristics (this evaluation) — fallback / safety net")
    print("═" * 60)

    if args.sample:
        print(f"\n  Fast mode: up to {args.sample:,} rows sampled per split.")

    all_metrics = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\nEvaluating {split_name} split ({len(split_df):,} rows)…")
        m = evaluate_split(split_df, split_name, detector, sample=args.sample)
        all_metrics.append(m)
        print_metrics(m)

    # ── 5. Summary table ─────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"  {'Split':<8}  {'Acc':>8}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'N':>8}")
    print(f"  {'─' * 56}")
    for m in all_metrics:
        print(f"  {m['split']:<8}  {m['accuracy']:>8.4f}  {m['precision']:>8.4f}"
              f"  {m['recall']:>8.4f}  {m['f1']:>8.4f}  {m['total']:>8,}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
