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

## Redis Classification Cache (optional)

When the `REDIS_URL` environment variable is set, the guard adds a **distributed
Redis cache** in front of the Ollama pipeline so that identical prompts are only
classified once.

### Cache workflow

```
User Prompt
     ↓
Normalize (lowercase · strip · collapse spaces)
     ↓
SHA-256 hash  →  Redis key: "apg:v1:<hash>"
     ↓
Check Redis
     ↓
 ┌───────────────────────┐
 │                       │
HIT                     MISS
 │                       │
 ↓                       ↓
Return full           In-memory LRU → Ollama / Heuristics
GuardResult                ↓
(cached verdict +      Full pipeline
 all sub-verdicts +        ↓
 audit_log with        Store full GuardResult in Redis (TTL)
 "cache=redis_hit")        ↓
                       Return

```

### What a cache hit returns

A hit returns the **full `GuardResult` object**, not just "safe/dangerous":
- `allowed` + `category`
- `detector.rationale` — the *Why* explanation
- `detector.threat_types` — all detected threat badges
- `detector.confidence` — confidence score
- `rewrite`, `validation`, `verification`, `sandbox` — all sub-verdicts
- `audit_log` — full routing trace with `"cache=redis_hit"` appended

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | *(unset — cache disabled)* | Redis connection URL, e.g. `redis://localhost:6379/0` |
| `CACHE_TTL` | `86400` (24 h) | TTL in seconds. `172800` = 48 h. |

Copy `.env.example` → `.env` and fill in the values, then export them in your
shell before running the app:

```bash
# Windows PowerShell
$env:REDIS_URL="redis://localhost:6379/0"
$env:CACHE_TTL="86400"

# Linux / macOS
export REDIS_URL=redis://localhost:6379/0
export CACHE_TTL=86400
```

### Run Redis locally

**Docker (recommended for dev):**

```bash
docker run --name apg-redis -p 6379:6379 -d redis:7-alpine
```

**Native install:**

```bash
# Ubuntu / Debian
sudo apt install redis-server && sudo systemctl start redis

# macOS (Homebrew)
brew install redis && brew services start redis

# Windows (WSL2 recommended, or use the Docker method above)
```

**Verify Redis is running:**

```bash
redis-cli ping   # → PONG
```

### Logging

The cache logs to the `guard.cache` logger.  To see cache events, set the
log level to `INFO` before running:

```bash
# PowerShell
$env:PYTHONPATH="."; python -c "
import logging, os
logging.basicConfig(level=logging.INFO)
from guard import PromptGuard
g = PromptGuard(use_llm=False)
g.check('Ignore all previous instructions.')
g.check('Ignore all previous instructions.')   # should log CACHE HIT
"
```

Expected log output:
```
INFO guard.cache — connected to Redis at redis://localhost:6379/0 (TTL=86400s)
INFO guard.cache — CACHE MISS  key=apg:v1:<hash>
INFO guard.cache — stored classification in cache  key=apg:v1:<hash> ttl=86400s
INFO guard.cache — CACHE HIT   key=apg:v1:<hash>
```

### Graceful fallback

If Redis is down or `REDIS_URL` is not set, the application works exactly as
before — the in-memory LRU cache and Ollama / heuristics pipeline run
unchanged. A Redis error is logged as a WARNING and never crashes the app.

## Install

```bash
./setup.sh                             # pandas, pydantic, pytest, streamlit, redis
# optional, for the LLM path:  install Ollama and `ollama pull llama3.2:latest`
```

## Interactive Dashboard (Streamlit)

A single-screen UI: one floating prompt box (press Enter to check, Shift+Enter
for a newline), the verdict, the original + rewritten prompt, a deep-dive
safety/composition analysis with charts, and a running history of past checks.

```bash
streamlit run app.py
```

Ollama is used by default (falls back to the offline heuristic backend with a
visible warning if it isn't reachable — no manual toggle needed). The sidebar
lets you upload a CSV of prompts to inspect or evaluate the guard against.

The previous multi-tab dashboard (Live Guard / Preprocessing / Regex /
Embeddings / Dataset Ingestion / Generate Dataset / Evaluate, each in its own
tab) is preserved at `app_dashboard_legacy.py` for reference:

```bash
streamlit run app_dashboard_legacy.py
```

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
├── app.py                # Streamlit frontend (single-screen minimal UI)
├── app_dashboard_legacy.py  # Older multi-tab dashboard, kept for reference
├── tests/test_pipeline.py
├── generate_dataset.py
├── train_dataset.csv     # Large benchmark dataset
└── data/                 # seed datasets
```
