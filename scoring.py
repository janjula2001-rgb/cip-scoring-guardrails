"""From LLM per-attribute scores to one overall score.

The LLM scores each priority 0-10 with a short note. Before those scores are
combined, deterministic overrides replace the ones code can decide better
(salary - see salary_guardrail.py; location eligibility; contract type), and
the overall score is a weighted average computed here, not by the model.
"""

import re
from dataclasses import dataclass

from models import AttributeScore, SalaryExpectation
from salary_guardrail import apply_salary_floor_override

# --- Seniority rank (used by the contract guard) ------------------------------

# Ordered so a string matching multiple keywords (e.g. "Senior Staff
# Engineer") resolves to the highest rank found, not just the first hit.
SENIORITY_RANK = {
    "intern": 0, "internship": 0, "trainee": 0,
    "junior": 1, "entry": 1, "entry level": 1, "entry-level": 1,
    "mid": 2, "middle": 2, "mid level": 2, "mid-level": 2, "intermediate": 2, "medior": 2,
    "senior": 3,
    "lead": 4, "staff": 4, "principal": 4, "architect": 4,
}


def seniority_rank(value: str | None) -> int | None:
    """Word-boundary matching, not a plain substring check - a plain
    substring check would let "leadership" or "leads" match the "lead"
    keyword just because it's attached as a substring."""
    v = (value or "").lower()
    best: int | None = None
    for keyword, rank in SENIORITY_RANK.items():
        if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", v) and (best is None or rank > best):
            best = rank
    return best


# --- Contract type --------------------------------------------------------------

_FLOOR_MONTHS = 6.0
_FLOOR_SCORE = 3
_TARGET_MONTHS = 24.0
_TARGET_SCORE = 8


def deterministic_contract_score(contract_type: str | None, duration_months: float | None) -> tuple[int, str] | None:
    """Permanent scores a flat 10. Fixed-term scales with stated duration:
    ~6 months scores low, ~2 years close to permanent but never 10."""
    if contract_type == "permanent":
        return 10, "Deterministic: posting states a permanent/indefinite position."
    if contract_type == "fixed_term":
        if duration_months is None or duration_months <= 0:
            return None  # fixed-term but no usable duration stated - not enough to grade
        if duration_months >= _TARGET_MONTHS:
            score = _TARGET_SCORE
        elif duration_months <= _FLOOR_MONTHS:
            score = max(1, round(_FLOOR_SCORE * duration_months / _FLOOR_MONTHS))
        else:
            frac = (duration_months - _FLOOR_MONTHS) / (_TARGET_MONTHS - _FLOOR_MONTHS)
            score = round(_FLOOR_SCORE + (_TARGET_SCORE - _FLOOR_SCORE) * frac)
        note = f"Deterministic: fixed-term contract, ~{duration_months:.0f} months."
        return score, note
    return None  # contract_type unstated - nothing to grade


def apply_contract_override(
    per_attribute: list[AttributeScore],
    contract_type: str | None,
    duration_months: float | None,
    posting_seniority: str | None = None,
) -> list[AttributeScore]:
    """Real bug: a posting whose own text said "three-month-long internship"
    was still extracted with contract_type="permanent". A genuinely permanent
    internship/traineeship is essentially never real, so that combination is
    treated as untrustworthy extracted data and marked not applicable instead
    of confidently scoring a "fact" that's most likely an extraction error."""
    if contract_type == "permanent" and seniority_rank(posting_seniority) == 0:
        return [
            AttributeScore(
                key="contract_type_stability",
                score=a.score,
                note="Not applicable: posting is an internship but was extracted as a permanent position - likely an extraction error, not treated as a verified fact.",
                applicable=False,
            )
            if a.key == "contract_type_stability"
            else a
            for a in per_attribute
        ]
    override = deterministic_contract_score(contract_type, duration_months)
    if override is not None:
        score, note = override
        return [
            AttributeScore(key="contract_type_stability", score=score, note=note) if a.key == "contract_type_stability" else a
            for a in per_attribute
        ]
    if contract_type is None:
        # Most postings don't state this at all - the model has nothing to go on either.
        return [
            AttributeScore(key="contract_type_stability", score=a.score, note="Not applicable: posting doesn't state a contract type.", applicable=False)
            if a.key == "contract_type_stability"
            else a
            for a in per_attribute
        ]
    return per_attribute


# --- Location eligibility ---------------------------------------------------------

# Score for the "location" priority when a posting states eligibility as a
# category (common on international remote boards) instead of naming a city.
# The model was unreliable at reading these for a Serbia-based candidate.
# Reachability depends on EU work authorisation:
#  - worldwide: hire-anywhere -> 10.
#  - emea / cet: region/timezone-based (Serbia is in both) -> 8 either way.
#  - europe_eu: must be eligible to work in the EU -> 9 with a passport, else 2.
#  - europe_except_serbia: only workable by relocating -> 4 with a passport, else 0.
#  - us_only / other_restricted: 0 either way.
_RESTRICTION_TIER_SCORES_EU: dict[str, int] = {
    "worldwide": 10, "europe_eu": 9, "emea": 8, "cet": 8,
    "europe_except_serbia": 4, "us_only": 0, "other_restricted": 0,
}
_RESTRICTION_TIER_SCORES_NO_EU: dict[str, int] = {
    "worldwide": 10, "europe_eu": 2, "emea": 8, "cet": 8,
    "europe_except_serbia": 0, "us_only": 0, "other_restricted": 0,
}


def restriction_tier_score(location_restriction_tier: str | None, has_eu_passport: bool = True) -> tuple[int, str] | None:
    """None if no tier was stated (the common case) or it's unrecognized -
    left to other rules / the model rather than guessed."""
    if location_restriction_tier is None:
        return None
    table = _RESTRICTION_TIER_SCORES_EU if has_eu_passport else _RESTRICTION_TIER_SCORES_NO_EU
    score = table.get(location_restriction_tier)
    if score is None:
        return None  # unrecognized value - leave to the model's own judgment rather than guess
    basis = "an EU passport" if has_eu_passport else "no EU work eligibility"
    note = (
        f'Deterministic: stated eligibility "{location_restriction_tier}" scores {score}/10 '
        f"for a Serbia-based candidate with {basis}."
    )
    return score, note


# --- Combining into one score ---------------------------------------------------------

def _emphasize_extremes(score: int, extremity: float) -> float:
    """Reshapes a 0-10 score so gaps near the extremes count for more than
    gaps near the middle: a 9 vs 10 should move the result more than a 5 vs
    6. 5 (neutral), 0, and 10 always stay fixed. `extremity` is the curve's
    exponent: 1.0 is plain linear; 1.4 (the default) gives roughly a 2.5x gap
    ratio between the extremes and the center."""
    u = (score - 5) / 5  # -1..1
    shaped = (abs(u) ** extremity) * (1 if u >= 0 else -1)
    return 5 + 5 * shaped


def compute_overall_score(per_attribute: list[AttributeScore], priorities: list[dict], extremity: float = 1.4) -> float:
    """Deterministic weighted average of extremity-shaped scores, so priority
    weights are guaranteed to matter mathematically rather than relying on
    the model's own aggregation, and outlier scores (a 1 or a 10) pull the
    result harder than scores clustered around neutral."""
    weight_by_key = {p["key"]: p["weight"] for p in priorities}
    total_weight = 0
    weighted_sum = 0.0
    for attr in per_attribute:
        if not attr.applicable:
            continue  # excluded entirely, not averaged in as a neutral guess
        weight = weight_by_key.get(attr.key)
        if weight is None:
            continue
        total_weight += weight
        weighted_sum += weight * _emphasize_extremes(attr.score, extremity)
    if total_weight == 0:
        return 50.0
    return round(100 * weighted_sum / (10 * total_weight), 1)


# --- Pipeline (glue written for this sample) ------------------------------------------
# In the app this happens where a scoring result is saved; the overrides there
# also cover location text, stack and seniority, which aren't included here.


@dataclass
class PostingFacts:
    """What extraction and salary research found for one posting."""

    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    contract_type: str | None = None
    contract_duration_months: float | None = None
    seniority: str | None = None
    location_restriction_tier: str | None = None


def apply_location_tier_override(per_attribute: list[AttributeScore], tier: str | None, has_eu_passport: bool) -> list[AttributeScore]:
    """Replaces the "location" score when the posting stated an eligibility tier."""
    override = restriction_tier_score(tier, has_eu_passport)
    if override is None:
        return per_attribute
    score, note = override
    return [AttributeScore(key="location", score=score, note=note) if a.key == "location" else a for a in per_attribute]


def score_posting(
    llm_scores: list[AttributeScore],
    facts: PostingFacts,
    salary_expectation: SalaryExpectation | None,
    priorities: list[dict],
    has_eu_passport: bool = True,
    extremity: float = 1.4,
) -> tuple[list[AttributeScore], float]:
    """LLM per-attribute scores -> deterministic overrides -> weighted overall score."""
    scores = apply_salary_floor_override(
        llm_scores, facts.salary_min, facts.salary_max, facts.salary_currency, salary_expectation, facts.salary_period
    )
    scores = apply_location_tier_override(scores, facts.location_restriction_tier, has_eu_passport)
    scores = apply_contract_override(scores, facts.contract_type, facts.contract_duration_months, facts.seniority)
    return scores, compute_overall_score(scores, priorities, extremity)
