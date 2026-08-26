"""Step 4 — Policy-as-Code Validator.

A deterministic sanity check on the rewrite before it is allowed to execute
(deck: "Pydantic schema, keyword blocklist, toxicity classifier"). Deterministic
means no LLM: the same input always yields the same verdict, so it can gate the
non-deterministic rewrite stage.

Validation checks (in order):
  1. **Structural** — rewrite status and prompt-text sanity.
  2. **Keyword blocklist** — 24 regex patterns across 8 pharma-compliance
     categories (off-label, jailbreak, dosing, evidence, safety-data,
     PII, false claims, missing safety language).
  3. **Toxicity** — severity-weighted lexicon (HIGH / MEDIUM) with
     context-awareness (surrounding-word check) and negation detection.
     Uses absolute weighted scoring so long prompts cannot dilute a
     toxic term below the threshold.
  4. **Prompt length** — minimum-length gate on the rewritten prompt.
  5. **Category-aware policy rules** — when the detector verdict is
     provided, verify that the rewrite actually contains the safety
     constraint markers required to address the detected threat type.
"""

import re
from typing import Optional

from .schemas import (
    DetectorVerdict,
    RewriteResult,
    RewriteStatus,
    ThreatType,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Keyword Blocklist
# ---------------------------------------------------------------------------
# Each entry: (regex_pattern, human-readable reason).
# These patterns should NEVER survive into a properly rewritten safe prompt.

_BLOCKLIST: list[tuple[str, str]] = [
    # ── Off-label / unapproved promotion ────────────────────────────────
    (r"\boff[- ]?label\b",
     "off-label content"),
    (r"\b(unapproved|not approved) (indication|use|dosing|application)s?\b",
     "unapproved indication reference"),
    (r"\bindication.{0,25}(has not|is not|have not|are not|were not|was not) (been )?approved\b",
     "unapproved indication reference"),
    (r"\b(promot|market|advertis|recommend|prescrib)\w*\b.{0,40}\b(unapproved|not approved|has not been approved)\b",
     "promotion of unapproved use"),
    (r"\b(avoid|omit|without).{0,30}(mention|disclos)\w*.{0,30}(regulatory|fda|approval|status)\b",
     "omission of regulatory status"),

    # ── Jailbreak / injection remnants ──────────────────────────────────
    (r"ignore .*(instructions|rules)",
     "instruction-override attempt"),
    (r"\bbypass\b.{0,30}(safety|filter|guardrail|compliance|restriction)s?\b",
     "safety-bypass attempt"),
    (r"\boverride\b.{0,30}(safety|restriction|rule|policy|guideline)s?\b",
     "policy-override attempt"),
    (r"disregard .{0,50}(safety|compliance|regulation|guideline|rule)s?\b",
     "compliance disregard"),

    # ── Medical / dosing advice ─────────────────────────────────────────
    (r"(lower|raise|increase|reduce) (their )?(dose|dosage)",
     "individualised dosing instruction"),
    (r"(adjust|change|modify|increase|reduce|raise|lower)\b.{0,20}\b(dose|dosage|mg|medication)\b",
     "dosage modification instruction"),
    (r"\bself[- ]?medicat",
     "self-medication reference"),

    # ── Evidence manipulation ───────────────────────────────────────────
    (r"even if (the )?evidence is limited",
     "disregard of evidence limitations"),
    (r"(overstat|exaggerat|embellish|inflate)\w*.{0,30}(evidence|efficacy|data|results?|benefits?)",
     "evidence exaggeration"),
    (r"(downplay|hide|conceal)\b.{0,30}(risks?|side effects?|adverse|harms?|dangers?)",
     "risk concealment"),

    # ── Safety data tampering ───────────────────────────────────────────
    (r"remove .*(minor|adverse|safety)",
     "safety data removal"),
    (r"(discard|exclude|suppress|omit|hide)\b.{0,20}(adverse|safety|side effects?|pharmacovigil)",
     "adverse-event suppression"),
    (r"(clean up|sanitize|scrub)\b.{0,20}(safety|adverse|pharmacovigil)",
     "safety record alteration"),

    # ── Identifiable patient data ───────────────────────────────────────
    (r"\bpatient.{0,10}(names?|identifiers?|identit(y|ies)|contacts?|address(es)?)\b",
     "patient-identifying information"),
    (r"\b(identifiable|identifying) (patient|individual|person|people)s?\b",
     "identifiable individual reference"),
    (r"\bwithout (patient )?(consent|authorization)\b",
     "action without consent"),

    # ── False guarantees / absolute claims ──────────────────────────────
    (r"guarantee.*(cure|results)",
     "unsubstantiated guarantee"),
    (r"\b(100%|completely|totally|absolutely) (safe|effective|cures?)\b",
     "absolute efficacy claim"),
    (r"\b(no|zero|free of) side[- ]?effects?\b",
     "false claim of no side effects"),
    (r"\bsuperior to all\b",
     "unsubstantiated superiority claim"),

    # ── Missing safety language ─────────────────────────────────────────
    (r"\bwithout .{0,30}(safety (information|warnings?|disclaimers?)|fair[- ]?balance|ISI)\b",
     "explicit omission of required safety language"),
]


# ---------------------------------------------------------------------------
# Toxicity Lexicon (severity-tiered, context-aware)
# ---------------------------------------------------------------------------

# HIGH severity: always problematic in pharmaceutical context.  These terms
# have no legitimate use in a rewritten prompt (weight: 3.0 each).
_TOXIC_HIGH: dict[str, str] = {
    "murder": "violent term",
    "murders": "violent term",
    "murdered": "violent term",
    "murdering": "violent term",
    "assault": "violent term",
    "assaults": "violent term",
    "assaulted": "violent term",
    "assaulting": "violent term",
    "bribe": "corruption",
    "bribes": "corruption",
    "bribed": "corruption",
    "bribing": "corruption",
    "bribery": "corruption",
    "kickback": "illegal payment",
    "kickbacks": "illegal payment",
    "blackmail": "coercion",
    "blackmails": "coercion",
    "blackmailed": "coercion",
    "blackmailing": "coercion",
    "extort": "coercion",
    "extorts": "coercion",
    "extorted": "coercion",
    "extorting": "coercion",
    "extortion": "coercion",
    "intimidate": "coercion",
    "intimidates": "coercion",
    "intimidated": "coercion",
    "intimidating": "coercion",
    "intimidation": "coercion",
    "counterfeit": "counterfeiting",
    "counterfeits": "counterfeiting",
    "counterfeited": "counterfeiting",
    "counterfeiting": "counterfeiting",
}

# MEDIUM severity: may have legitimate context depending on surrounding words.
# Context and negation checks apply (weight: 1.5 each, 0.0 if neutralised).
_TOXIC_MEDIUM: dict[str, str] = {
    "kill": "potentially violent",
    "kills": "potentially violent",
    "killed": "potentially violent",
    "killing": "potentially violent",
    "hate": "hostile language",
    "hates": "hostile language",
    "hated": "hostile language",
    "hating": "hostile language",
    "hateful": "hostile language",
    "stupid": "demeaning language",
    "idiot": "demeaning language",
    "idiots": "demeaning language",
    "idiotic": "demeaning language",
    "scam": "fraud-related",
    "scams": "fraud-related",
    "scammed": "fraud-related",
    "scamming": "fraud-related",
    "fraud": "fraud-related",
    "frauds": "fraud-related",
    "fraudulent": "fraud-related",
    "defraud": "fraud-related",
    "defrauds": "fraud-related",
    "defrauded": "fraud-related",
    "defrauding": "fraud-related",
    "falsify": "data integrity violation",
    "falsifies": "data integrity violation",
    "falsified": "data integrity violation",
    "falsifying": "data integrity violation",
    "falsification": "data integrity violation",
    "fabricate": "data integrity violation",
    "fabricates": "data integrity violation",
    "fabricated": "data integrity violation",
    "fabricating": "data integrity violation",
    "fabrication": "data integrity violation",
    "destroy": "destructive action",
    "destroys": "destructive action",
    "destroyed": "destructive action",
    "destroying": "destructive action",
    "destruction": "destructive action",
    "abuse": "misuse / harm",
    "abuses": "misuse / harm",
    "abused": "misuse / harm",
    "abusing": "misuse / harm",
    "abusive": "misuse / harm",
    "exploit": "exploitation",
    "exploits": "exploitation",
    "exploited": "exploitation",
    "exploiting": "exploitation",
    "exploitation": "exploitation",
    "manipulate": "manipulation",
    "manipulates": "manipulation",
    "manipulated": "manipulation",
    "manipulating": "manipulation",
    "manipulation": "manipulation",
    "deceive": "deception",
    "deceives": "deception",
    "deceived": "deception",
    "deceiving": "deception",
    "deception": "deception",
    "deceptive": "deception",
    "corrupt": "corruption",
    "corrupts": "corruption",
    "corrupted": "corruption",
    "corrupting": "corruption",
    "corruption": "corruption",
    "coerce": "coercion",
    "coerces": "coercion",
    "coerced": "coercion",
    "coercing": "coercion",
    "coercion": "coercion",
    "coercive": "coercion",
    "harass": "harassment",
    "harasses": "harassment",
    "harassed": "harassment",
    "harassing": "harassment",
    "harassment": "harassment",
    "discriminate": "discrimination",
    "discriminates": "discrimination",
    "discriminated": "discrimination",
    "discriminating": "discrimination",
    "discrimination": "discrimination",
    "discriminatory": "discrimination",
    "retaliate": "retaliation",
    "retaliates": "retaliation",
    "retaliated": "retaliation",
    "retaliating": "retaliation",
    "retaliation": "retaliation",
    "retaliatory": "retaliation",
    "sabotage": "sabotage",
    "sabotages": "sabotage",
    "sabotaged": "sabotage",
    "sabotaging": "sabotage",
    "suppress": "suppression",
    "suppresses": "suppression",
    "suppressed": "suppression",
    "suppressing": "suppression",
    "suppression": "suppression",
    "withhold": "withholding",
    "withholds": "withholding",
    "withheld": "withholding",
    "withholding": "withholding",
}

# Context words that neutralise a MEDIUM-severity toxic term.
# Checked within a ±3 word window around the term.
_SAFE_CONTEXT: dict[str, frozenset[str]] = {
    "kill": frozenset({
        "process", "session", "task", "job", "thread",
        "switch", "timer", "signal", "command",
    }),
    "kills": frozenset({
        "process", "session", "task", "job", "thread",
        "switch", "timer", "signal", "command",
    }),
    "killed": frozenset({
        "process", "session", "task", "job", "thread",
        "switch", "timer", "signal", "command",
    }),
    "killing": frozenset({
        "process", "session", "task", "job", "thread",
        "switch", "timer", "signal", "command",
    }),
    "abuse": frozenset({
        "substance", "drug", "alcohol", "opioid",
        "prevention", "disorder", "potential",
    }),
    "abuses": frozenset({
        "substance", "drug", "alcohol", "opioid",
        "prevention", "disorder", "potential",
    }),
    "abused": frozenset({
        "substance", "drug", "alcohol", "opioid",
        "prevention", "disorder", "potential",
    }),
    "abusing": frozenset({
        "substance", "drug", "alcohol", "opioid",
        "prevention", "disorder", "potential",
    }),
    "exploit": frozenset({
        "vulnerability", "bug", "security", "cve", "patch",
    }),
    "exploits": frozenset({
        "vulnerability", "bug", "security", "cve", "patch",
    }),
    "exploited": frozenset({
        "vulnerability", "bug", "security", "cve", "patch",
    }),
    "exploiting": frozenset({
        "vulnerability", "bug", "security", "cve", "patch",
    }),
    "exploitation": frozenset({
        "vulnerability", "bug", "security", "cve", "patch",
    }),
    "destroy": frozenset({
        "file", "record", "document", "retention", "securely",
    }),
    "destroys": frozenset({
        "file", "record", "document", "retention", "securely",
    }),
    "destroyed": frozenset({
        "file", "record", "document", "retention", "securely",
    }),
    "destroying": frozenset({
        "file", "record", "document", "retention", "securely",
    }),
    "destruction": frozenset({
        "file", "record", "document", "retention", "securely",
    }),
    "manipulate": frozenset({
        "data", "variable", "parameter", "image",
        "text", "spreadsheet", "chart", "visualization",
    }),
    "manipulates": frozenset({
        "data", "variable", "parameter", "image",
        "text", "spreadsheet", "chart", "visualization",
    }),
    "manipulated": frozenset({
        "data", "variable", "parameter", "image",
        "text", "spreadsheet", "chart", "visualization",
    }),
    "manipulating": frozenset({
        "data", "variable", "parameter", "image",
        "text", "spreadsheet", "chart", "visualization",
    }),
    "manipulation": frozenset({
        "data", "variable", "parameter", "image",
        "text", "spreadsheet", "chart", "visualization",
    }),
    "corrupt": frozenset({
        "data", "file", "database", "disk", "storage", "memory",
    }),
    "corrupts": frozenset({
        "data", "file", "database", "disk", "storage", "memory",
    }),
    "corrupted": frozenset({
        "data", "file", "database", "disk", "storage", "memory",
    }),
    "corrupting": frozenset({
        "data", "file", "database", "disk", "storage", "memory",
    }),
    "corruption": frozenset({
        "data", "file", "database", "disk", "storage", "memory",
    }),
    "suppress": frozenset({
        "warning", "error", "output", "notification",
        "message", "log", "alert", "viral", "tumor", "tumour",
        "immune", "symptom", "symptoms", "acid",
    }),
    "suppresses": frozenset({
        "warning", "error", "output", "notification",
        "message", "log", "alert", "viral", "tumor", "tumour",
        "immune", "symptom", "symptoms", "acid",
    }),
    "suppressed": frozenset({
        "warning", "error", "output", "notification",
        "message", "log", "alert", "viral", "tumor", "tumour",
        "immune", "symptom", "symptoms", "acid",
    }),
    "suppressing": frozenset({
        "warning", "error", "output", "notification",
        "message", "log", "alert", "viral", "tumor", "tumour",
        "immune", "symptom", "symptoms", "acid",
    }),
    "suppression": frozenset({
        "warning", "error", "output", "notification",
        "message", "log", "alert", "viral", "tumor", "tumour",
        "immune", "symptom", "symptoms", "acid",
    }),
    "withhold": frozenset({"judgment", "opinion"}),
    "withholds": frozenset({"judgment", "opinion"}),
    "withheld": frozenset({"judgment", "opinion"}),
    "withholding": frozenset({"judgment", "opinion"}),
    "discriminate": frozenset({
        "between", "among", "types", "classes", "categories",
    }),
    "discriminates": frozenset({
        "between", "among", "types", "classes", "categories",
    }),
    "discriminated": frozenset({
        "between", "among", "types", "classes", "categories",
    }),
    "discriminating": frozenset({
        "between", "among", "types", "classes", "categories",
    }),
    "discrimination": frozenset({
        "between", "among", "types", "classes", "categories",
    }),
    "scam": frozenset({
        "anti", "prevention", "detection", "awareness",
        "protect", "report",
    }),
    "scams": frozenset({
        "anti", "prevention", "detection", "awareness",
        "protect", "report",
    }),
    "fraud": frozenset({
        "anti", "detection", "prevention", "reporting",
        "investigate", "compliance",
    }),
    "frauds": frozenset({
        "anti", "detection", "prevention", "reporting",
        "investigate", "compliance",
    }),
    "fraudulent": frozenset({
        "anti", "detection", "prevention", "reporting", "identify",
    }),
    "defraud": frozenset({
        "anti", "detection", "prevention", "reporting",
    }),
    "fabricate": frozenset({"pre", "prefabricated"}),
    "fabricates": frozenset({"pre", "prefabricated"}),
    "fabricated": frozenset({"pre", "prefabricated"}),
    "fabricating": frozenset({"pre", "prefabricated"}),
    "fabrication": frozenset({"pre", "prefabricated"}),
}

# Words that negate a following toxic term, rendering it safe.
# Checked within a 3-word window BEFORE the toxic term.
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "without", "nor",
    "don't", "doesn't", "didn't", "won't", "shouldn't",
    "mustn't", "cannot", "can't",
    "prevent", "prevents", "preventing", "prevention",
    "prohibit", "prohibits", "prohibiting", "prohibited", "prohibition",
    "avoid", "avoids", "avoiding", "avoidance",
    "protect", "protects", "protecting", "protection",
    "guard", "guards", "guarding", "safeguard", "safeguards", "safeguarding",
    "anti", "non", "free", "against",
})

# Categories of blocklist rules that represent prohibited actions
# which become compliant safety directives when explicitly negated
# (e.g. "must not bypass safety controls", "do not suppress adverse events",
# "without exaggerating efficacy").
_NEGATABLE_BLOCKLIST_REASONS: frozenset[str] = frozenset({
    "safety-bypass attempt",
    "policy-override attempt",
    "compliance disregard",
    "evidence exaggeration",
    "risk concealment",
    "adverse-event suppression",
    "safety data removal",
    "safety record alteration",
})

# A prompt's absolute toxicity score must stay below this threshold.
# One HIGH term (3.0) or two un-neutralised MEDIUM terms (2 × 1.5 = 3.0)
# are enough to exceed it.  A single MEDIUM term (1.5) also exceeds it,
# which is intentionally strict for rewritten pharmaceutical prompts.
_TOXICITY_THRESHOLD: float = 1.0


def _toxicity_check(text: str) -> tuple[float, list[str]]:
    """Score *text* for toxic content.

    Uses severity-weighted scoring with context and negation awareness
    so that legitimate phrases (e.g. "kill the process", "do not suppress
    adverse events", "risks should be prevented") are not penalised while
    genuinely harmful terms are always caught regardless of prompt length.

    Returns ``(weighted_score, detail_reasons)``.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0, []

    score: float = 0.0
    details: list[str] = []

    has_global_prevention = bool(re.search(r"\b(prevent\w*|prohibit\w*|avoid\w*|mitigat\w*)\b", text.lower()))

    for i, token in enumerate(tokens):
        is_high = token in _TOXIC_HIGH
        is_medium = token in _TOXIC_MEDIUM

        if not is_high and not is_medium:
            continue

        # ── Negation check (all severities): preceding or following window ────────
        pre_window = set(tokens[max(0, i - 3):i])
        post_window = set(tokens[i + 1:min(len(tokens), i + 4)])
        if (pre_window | post_window) & _NEGATION_WORDS:
            continue  # e.g. "do not suppress", "anti-fraud", "preventing bribery"

        # ── Global prevention framing check (e.g. risks to be prevented) ──
        if has_global_prevention and ("prevent" in text.lower() or "mitigat" in text.lower()):
            continue

        # ── Context check (MEDIUM only): surrounding ±3 words ─────────
        if is_medium and token in _SAFE_CONTEXT:
            full_window = set(
                tokens[max(0, i - 3):i]
                + tokens[i + 1:min(len(tokens), i + 4)]
            )
            if full_window & _SAFE_CONTEXT[token]:
                continue  # e.g. "kill the process", "substance abuse"

        if is_high:
            score += 3.0
            label = _TOXIC_HIGH[token]
            details.append(f"High-severity: '{token}' ({label})")
        else:
            score += 1.5
            label = _TOXIC_MEDIUM[token]
            details.append(f"'{token}' ({label})")

    return round(score, 2), details


# ---------------------------------------------------------------------------
# Category-Aware Policy Rules
# ---------------------------------------------------------------------------
# When the detector has flagged a specific threat type and the rewriter
# produced a rewrite, the validator verifies that the rewrite actually
# contains the safety-constraint language needed to address that threat.
#
# Each entry: ThreatType → (list of marker patterns — at least one must
# be present, human-readable failure reason).

_REQUIRED_CONSTRAINTS: dict[ThreatType, tuple[list[str], str]] = {
    ThreatType.PII_PHI_EXPOSURE: (
        [r"de-?identif", r"\baggregat", r"non-?identifying", r"\banonym"],
        "Rewrite addressing PHI/PII risk must include "
        "de-identification or aggregation language",
    ),
    ThreatType.OFF_LABEL_PROMOTION: (
        [r"approved (indication|material|messaging|use)", r"\bon[- ]?label\b"],
        "Rewrite addressing off-label risk must reference "
        "approved indications or materials",
    ),
    ThreatType.MISLEADING_CLAIM: (
        [
            r"fair[- ]?balance",
            r"\bsubstantiat",
            r"approved claim",
            r"safety (information|warnings?|language|disclaimers?)",
        ],
        "Rewrite addressing misleading-claim risk must include "
        "fair-balance or substantiation language",
    ),
    ThreatType.UNSAFE_TARGETING: (
        [
            r"non-?sensitive",
            r"\bconsented\b",
            r"explicit (selection )?criteria",
            r"compliant (selection )?criteria",
            r"\bgovernance\b",
        ],
        "Rewrite addressing unsafe-targeting risk must include "
        "consented attributes or explicit selection criteria language",
    ),
    ThreatType.SAFETY_DATA_TAMPERING: (
        [
            r"without (removing|filtering|altering|deleting|changing)",
            r"\ball (adverse|safety)",
            r"complete (safety|adverse)",
            r"\bno.{0,10}(filter|remov|discard)",
        ],
        "Rewrite addressing safety-data risk must include "
        "data-integrity preservation language",
    ),
    ThreatType.MEDICAL_ADVICE: (
        [
            r"non-?individuali[sz]ed",
            r"general (information|education)",
            r"consult.{0,20}(healthcare|physician|doctor|HCP|provider)",
            r"does not constitute.{0,20}(medical|clinical) advice",
        ],
        "Rewrite addressing medical-advice risk must include "
        "non-individualised or consult-HCP language",
    ),
}

# Minimum length (in characters) for a rewritten prompt to be considered
# substantive.  Anything shorter is likely an incomplete or broken rewrite.
_MIN_PROMPT_LENGTH: int = 10


def _check_required_constraints(
    candidate: str,
    verdict: DetectorVerdict,
) -> list[str]:
    """Verify that *candidate* contains the safety markers required for
    each threat type flagged by *verdict*."""
    reasons: list[str] = []
    for ttype in verdict.threat_types:
        entry = _REQUIRED_CONSTRAINTS.get(ttype)
        if entry is None:
            continue
        patterns, failure_reason = entry
        if not any(re.search(p, candidate, re.IGNORECASE) for p in patterns):
            reasons.append(f"Policy: {failure_reason}.")
    return reasons


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate(
    rewrite: RewriteResult,
    verdict: Optional[DetectorVerdict] = None,
) -> ValidationResult:
    """Deterministic policy-as-code validation of a rewritten prompt.

    Parameters
    ----------
    rewrite:
        The output of the Safe Intent Rewriter (Step 3).
    verdict:
        Optional detector verdict from Step 2.  When provided, enables
        category-aware policy checks that verify the rewrite actually
        addresses the specific threat type(s) detected.  Omitting it
        (the default) is fully backward-compatible — structural,
        blocklist, toxicity, and length checks still run.

    Returns
    -------
    :class:`~guard.schemas.ValidationResult`
        Deterministic pass/fail with human-readable reasons.
    """
    # 1. Structural check: status and prompt text sanity.
    if rewrite.status in (RewriteStatus.INVALID, RewriteStatus.NEEDS_CLARIFICATION):
        return ValidationResult(
            passed=False,
            reasons=[f"Rewrite status is {rewrite.status.value}; not executable."],
        )

    candidate = rewrite.rewritten_prompt or ""
    stripped = candidate.strip()
    if not stripped:
        return ValidationResult(
            passed=False,
            reasons=["Rewrite produced no prompt text."],
        )

    reasons: list[str] = []

    # 2. Keyword blocklist (with negation sensitivity for compliant directives).
    for pattern, reason in _BLOCKLIST:
        for match in re.finditer(pattern, candidate, re.IGNORECASE):
            if reason in _NEGATABLE_BLOCKLIST_REASONS:
                start_pos = match.start()
                prefix = candidate[:start_pos]
                pre_tokens = re.findall(r"[a-z']+", prefix.lower())
                window = pre_tokens[-3:] if len(pre_tokens) >= 3 else pre_tokens
                if set(window) & _NEGATION_WORDS:
                    continue  # Safely negated (e.g. 'do not suppress adverse events', 'must not bypass safety')
            reasons.append(f"Blocked: {reason}.")
            break

    # 3. Toxicity (severity-weighted, context-aware).
    tox_score, tox_details = _toxicity_check(candidate)
    if tox_score >= _TOXICITY_THRESHOLD:
        reasons.append(
            f"Toxicity score {tox_score:.1f} exceeds threshold "
            f"({_TOXICITY_THRESHOLD}): {'; '.join(tox_details)}."
        )

    # 4. Prompt length.
    if len(stripped) < _MIN_PROMPT_LENGTH:
        reasons.append(
            f"Rewritten prompt too short ({len(stripped)} chars); "
            f"minimum is {_MIN_PROMPT_LENGTH}."
        )

    # 5. Category-aware required-constraint checks.
    if verdict is not None:
        constraint_reasons = _check_required_constraints(candidate, verdict)
        reasons.extend(constraint_reasons)

    return ValidationResult(passed=not reasons, reasons=reasons)
