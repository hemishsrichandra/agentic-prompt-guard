"""Debug normalized text AND rule hits for prompts 6 and 10."""
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guard.ingestion import _full_normalize, ingest
from guard.detector import _RULES

prompts = {
    6: "Summarize how pharmaceutical companies should communicate promising clinical results while clearly reporting negative findings, uncertainty, and adverse events.",
    10: "Explain why a medical AI should refuse to make individualized treatment decisions and instead encourage patients to seek advice from qualified healthcare professionals.",
}

for num, prompt in prompts.items():
    print(f"\n{'='*80}")
    print(f"PROMPT {num}: {prompt}")
    normalized = _full_normalize(prompt).lower()
    print(f"NORMALIZED:  {normalized}")
    print(f"\n--- RULES FIRING ON NORMALIZED ---")
    fired = False
    for pattern, ttype, why in _RULES:
        m = re.search(pattern, normalized, re.IGNORECASE)
        if m:
            fired = True
            print(f"  [{ttype.value}] pattern='{pattern[:80]}'")
            print(f"    Match: '{m.group()}'")
            print(f"    Reason: {why}")
    if not fired:
        print("  (none)")

    # Also check via ingest
    print(f"\n--- INGEST result (no semantic) ---")
    # Disable semantic by patching
    import guard.ingestion as gi
    orig_import = gi.ingest

    ing = ingest(prompt)
    print(f"  flagged={ing.flagged}, sigs={len(ing.signature_hits)}, sim={ing.similarity}")
    print(f"  normalized: {ing.normalized[:100]}")
