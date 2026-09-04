# Agentic Prompt Guard

An enterprise-grade **router and classifier pipeline** designed to screen prompts *before* an autonomous agent acts on them. 

Agentic Prompt Guard serves as an upstream responsibility gate. Safe prompts take a fast path straight to execution, while risky, ambiguous, or disguised prompts enter a remediation loop that rewrites, clarifies, and re-validates them. This ensures that the latent threat of dangerous prompts is neutralized before downstream tools or external APIs are executed.

## Core Architecture

```text
User prompt
   │
   ▼
1. Ingestion             Normalization (NFKC, zero-width stripping), 
   (guard/ingestion.py)  Regex signatures, Semantic similarity against attack corpus
   │
   ▼
2. Threat & Disguise     Structured verdict: is_safe / category / threat_types
   Detector              (Primary: Ollama LLM | Fallback: Heuristics & Difflib)
   (guard/detector.py)
   │
   ├── SAFE ─────────────────────────────► 5. Safe Execution Sandbox
   │                                          (guard/sandbox.py) + PII/harm filter
   ▼ RISKY / DISGUISED
3. Safe Intent Rewriter  Extract benign intent → safe query, or ask
   (guard/rewriter.py)   clarification, or flag as INVALID
   │
   ▼
4. Policy-as-Code        Deterministic gate: Pydantic schemas, keyword blocklists,
   Validator             and toxicity checks
   (guard/validator.py)
   │
   ▼
   Verification          Re-run the detector on the rewrite to confirm the
   (in pipeline.py)      disguised-risk signals are gone.
   │
   ▼
   Audit Engine          Router logs every state transition (guard/pipeline.py)
```

## Key Features

- **Multi-layered Ingestion Engine:** Defeats text-smuggling attacks via Unicode NFKC normalization, invisible character stripping, and Base64/Hex payload decoding.
- **Resilient Intelligence:** The Threat Detector, Rewriter, and Sandbox primarily call a self-hosted Ollama model (`guard/llm.py`). If the LLM is unavailable, the system guarantees graceful degradation by falling back to a **deterministic heuristic engine** (regex signatures, threat lexicons, and structural pattern heuristics).
- **Fast Semantic Vector Matching:** Scans against known jailbreak corpora using `sentence-transformers` to identify rephrased exploits. Uses a zero-dependency `difflib.SequenceMatcher` as a fallback.
- **Policy-as-Code Validation:** A deterministic, non-LLM validation step ensures reliable gating before arbitrary execution.
- **Append-Only Audit Logs:** Every `GuardResult` carries the classification, the detector rationale, the rewrite/clarification, the validation verdicts, and a full routing trace for compliance and observability.

## Install

```bash
pip install -r requirements.txt        # pandas, pydantic, pytest, streamlit
# optional, for the LLM path:  install Ollama and `ollama pull llama3.2:latest`
```

## Interactive Dashboard (Streamlit)

An interactive dashboard exposes every stage of the pipeline in its own tab.

```bash
streamlit run app.py
```

| Tab | What it shows |
|-----|---------------|
| 🛡️ **Live Guard** | Full pipeline on a prompt: verdict, threats, rewrite, sandbox, audit log. |
| 🔬 **Preprocessing** | The normalisation cascade peeling back a disguised prompt, pass by pass. |
| 🔎 **Regex / Signatures** | Jailbreak signature hits, decoded payloads, and heuristic threat labels. |
| 🧠 **Embeddings** | Top-k nearest known-attack strings by semantic similarity (bar chart). |
| 📥 **Dataset Ingestion** | Upload/load a CSV, auto-detect columns, view label distribution + stats. |
| 🧪 **Generate Dataset** | Synthesise labelled prompts (rows/seed/dedup) and download the CSV. |
| 📊 **Evaluate** | Score the guard against an ingested labelled dataset (accuracy/P/R/F1). |

The sidebar toggles the LLM (Ollama) vs. heuristic backend and reports which similarity backend is active.

## Usage

Run from the `agentic_prompt_guard/` directory so the `guard` package is importable.

```bash
# screen a single prompt (uses Ollama if running, else heuristics)
python -m guard.cli check "Segment engagement by likelihood to start therapy and create a targeting list."

# force the offline heuristic backend, and run allowed prompts in the sandbox
python -m guard.cli check "Generate a de-identified, aggregated engagement report." --execute --no-llm

# evaluate the guard's safe/unsafe calls against a labeled CSV
python -m guard.cli eval data/seed_dataset.csv --no-llm
```

Programmatic use:

```python
from guard import PromptGuard

guard = PromptGuard(use_llm=True)          # falls back automatically if Ollama is down
result = guard.check("Write a catchy social post for the new biologic.", execute=True)
print(result.summary())                     # [BLOCKED] Risky via blocked
print(result.audit_log)                     # full routing trace
```

## Tests

```bash
pytest -q      # 11 tests, offline (heuristic backend)
```

## Dataset Generator

`generate_dataset.py` produces labeled Safe/Unsafe pharma prompts to train and evaluate the detector. The repository includes a massive `train_dataset.csv` balancing safe and adversarial prompts.

```bash
python generate_dataset.py                       # 100k rows -> generated_pharma_dataset_100000.csv
python generate_dataset.py --rows 1000 --seed 7
python generate_dataset.py --dedup               # unique prompts only
```

| Flag      | Default                              | Description                          |
|-----------|--------------------------------------|--------------------------------------|
| `--rows`  | `100000`                             | Number of rows (must be ≥ 1).        |
| `--seed`  | `42`                                 | Random seed for reproducibility.     |
| `--dedup` | off                                  | Emit only unique prompt strings.     |
| `--out`   | `generated_pharma_dataset_<rows>.csv`| Output path.                         |

## Layout

```text
agentic_prompt_guard/
├── guard/
│   ├── llm.py            # Ollama HTTP client + availability check
│   ├── ingestion.py      # Normalization and syntax attack parsing
│   ├── detector.py       # LLM/Heuristic threat classification
│   ├── rewriter.py       # Safe intent extraction and rewriting
│   ├── validator.py      # Policy-as-code deterministic checking
│   ├── sandbox.py        # Safe execution and PII filtering
│   ├── pipeline.py       # State router and recursive verification
│   ├── schemas.py        # Pydantic contract / classification enums
│   ├── datasets.py       # Dataset ingestion: load, column-detect, evaluate
│   └── cli.py            # `python -m guard.cli check|eval`
├── app.py                # Streamlit frontend
├── tests/test_pipeline.py
├── generate_dataset.py
├── train_dataset.csv     # Large benchmark dataset
└── data/                 # seed datasets
```
