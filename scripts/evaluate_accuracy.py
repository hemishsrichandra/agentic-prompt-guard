"""Standalone evaluation of the heuristic detector rules on train_dataset.csv.

Performs:
1. Load dataset (train_dataset.csv)
2. Deduplicate
3. Stratified 70/15/15 split (train, val, test)
4. Evaluate accuracy, precision, recall, F1, and confusion matrix on each split.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

# ── Load rules from detector.py directly ────────────────────────────────────
with open(_ROOT / "guard" / "detector.py") as f:
    detector_code = f.read()

# Strip relative imports from detector code for standalone execution
detector_code = re.sub(r"from \.\w+ import [^\n]+", "# stripped import", detector_code)

# Execute detector definitions in local namespace
namespace = {
    "__name__": "guard.detector",
    "__file__": str(_ROOT / "guard" / "detector.py"),
}
# Provide mock/stub classes for schemas so detector.py loads with pure stdlib
class Category:
    GOOD_VALID = "Good/Valid"
    RESPONSIBLE = "Responsible"
    RISKY = "Risky"

class ThreatType:
    NONE = "none"
    JAILBREAK = "jailbreak"
    ENCODED_PAYLOAD = "encoded_payload"
    ROLEPLAY_EXPLOIT = "roleplay_exploit"
    SPLIT_REQUEST = "split_request"
    PII_PHI_EXPOSURE = "pii_phi_exposure"
    OFF_LABEL_PROMOTION = "off_label_promotion"
    MISLEADING_CLAIM = "misleading_claim"
    UNSAFE_TARGETING = "unsafe_targeting"
    MEDICAL_ADVICE = "medical_advice"
    SAFETY_DATA_TAMPERING = "safety_data_tampering"
    AMBIGUOUS = "ambiguous"
    HYPOTHETICAL_BYPASS = "hypothetical_bypass"
    PROMPT_OVERLOAD = "prompt_overload"
    OTHER = "other"

class DetectorVerdict:
    def __init__(self, is_safe, category, threat_types, ambiguity_flags, rationale, confidence):
        self.is_safe = is_safe
        self.category = category
        self.threat_types = threat_types
        self.ambiguity_flags = ambiguity_flags
        self.rationale = rationale
        self.confidence = confidence

class IngestionResult:
    def __init__(self, normalized, flagged=False, signature_hits=None, decoded_payloads=None, similarity=0.0,
                 homoglyph_detected=False, leetspeak_detected=False, whitespace_injection_detected=False):
        self.normalized = normalized
        self.flagged = flagged
        self.signature_hits = signature_hits or []
        self.decoded_payloads = decoded_payloads or []
        self.similarity = similarity
        self.homoglyph_detected = homoglyph_detected
        self.leetspeak_detected = leetspeak_detected
        self.whitespace_injection_detected = whitespace_injection_detected

namespace.update({
    "Category": Category,
    "ThreatType": ThreatType,
    "DetectorVerdict": DetectorVerdict,
    "IngestionResult": IngestionResult,
    "OllamaClient": None,
})

# Extract and exec rule definitions and ThreatDetector
exec(detector_code, namespace)
ThreatDetector = namespace["ThreatDetector"]
detector = ThreatDetector(llm=None)

# ── Evaluation ──────────────────────────────────────────────────────────────
SEED = 42

def classify_prompt(prompt: str) -> bool:
    ing = IngestionResult(normalized=prompt)
    verdict = detector._detect_heuristic(ing)
    return verdict.is_safe

def evaluate_df(df: pd.DataFrame, split_name: str) -> dict:
    tp = tn = fp = fn = 0
    for _, row in df.iterrows():
        p = str(row["Risky ambiguous prompt"])
        gold_unsafe = str(row["Safe or Unsafe?"]).strip() != "Safe"
        pred_safe = classify_prompt(p)
        pred_unsafe = not pred_safe

        if gold_unsafe and pred_unsafe:
            tp += 1
        elif not gold_unsafe and not pred_unsafe:
            tn += 1
        elif not gold_unsafe and pred_unsafe:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "split": split_name,
        "total": total,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def main():
    print("Loading train_dataset.csv...")
    df_raw = pd.read_csv(_ROOT / "train_dataset.csv")
    print(f"Loaded {len(df_raw):,} raw rows.")

    df = df_raw.drop_duplicates(subset=["Risky ambiguous prompt"]).reset_index(drop=True)
    print(f"Deduped to {len(df):,} unique prompts (Safe: {(df['Safe or Unsafe?'] == 'Safe').sum():,}, Unsafe: {(df['Safe or Unsafe?'] != 'Safe').sum():,})")

    # Stratified 70/15/15 split
    safe_idx = df[df["Safe or Unsafe?"] == "Safe"].index.tolist()
    unsafe_idx = df[df["Safe or Unsafe?"] != "Safe"].index.tolist()

    r = random.Random(SEED)
    r.shuffle(safe_idx)
    r.shuffle(unsafe_idx)

    def cut(idx):
        n = len(idx)
        n_train = round(n * 0.70)
        n_val = round(n * 0.15)
        return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    s_tr, s_v, s_te = cut(safe_idx)
    u_tr, u_v, u_te = cut(unsafe_idx)

    train_df = df.loc[s_tr + u_tr].sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = df.loc[s_v + u_v].sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = df.loc[s_te + u_te].sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Splits -> Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    metrics = []
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        m = evaluate_df(split_df, name)
        metrics.append(m)
        print(f"\n[{name.upper()}] Accuracy: {m['accuracy']:.4%} | Precision: {m['precision']:.4%} | Recall: {m['recall']:.4%} | F1: {m['f1']:.4%}")
        print(f"  TP={m['tp']:,}  FN={m['fn']:,}  FP={m['fp']:,}  TN={m['tn']:,}")

    print("\n" + "=" * 65)
    print(f"  {'Split':<8}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1-Score':>10}  {'N':>8}")
    print("  " + "-" * 61)
    for m in metrics:
        print(f"  {m['split']:<8}  {m['accuracy']:>10.4%}  {m['precision']:>10.4%}  {m['recall']:>10.4%}  {m['f1']:>10.4%}  {m['total']:>8,}")
    print("=" * 65)

if __name__ == "__main__":
    main()
