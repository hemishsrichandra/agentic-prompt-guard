"""Debug which rules fire for prompts 6 and 10."""
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guard.detector import _RULES, _SAFE_MARKERS, _STRONG_SAFE_FRAMES, _PHARMA_THREATS

prompts = {
    6: "Summarize how pharmaceutical companies should communicate promising clinical results while clearly reporting negative findings, uncertainty, and adverse events.",
    10: "Explain why a medical AI should refuse to make individualized treatment decisions and instead encourage patients to seek advice from qualified healthcare professionals.",
}

for num, prompt in prompts.items():
    text = prompt.lower()
    print(f"\n{'='*80}")
    print(f"PROMPT {num}: {prompt}")
    print(f"{'='*80}")
    print("\n--- RULES FIRING ---")
    for pattern, ttype, why in _RULES:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            print(f"  [{ttype.value}] pattern='{pattern[:60]}' match='{m.group()}'")
            print(f"    reason: {why}")
    print("\n--- SAFE MARKERS FOUND ---")
    for p in _SAFE_MARKERS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            print(f"  '{p}' -> '{m.group()}'")
    print("\n--- STRONG SAFE FRAMES ---")
    for p in _STRONG_SAFE_FRAMES:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            print(f"  MATCH: '{p[:80]}'")
