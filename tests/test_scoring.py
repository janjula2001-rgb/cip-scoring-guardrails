"""Contract and location overrides, the weighted overall score, and the full
LLM scores -> overrides -> overall score pipeline."""

import pytest

from models import AttributeScore, SalaryExpectation
from scoring import (
    PostingFacts,
    _emphasize_extremes,
    apply_contract_override,
    compute_overall_score,
    restriction_tier_score,
    score_posting,
    seniority_rank,
)


def _contract(score=6):
    return [AttributeScore(key="contract_type_stability", score=score, note="model guess")]


# --- contract type ------------------------------------------------------------

def test_permanent_internship_is_marked_not_applicable_not_verified():
    # A "three-month internship" extracted as permanent: don't trust it.
    result = apply_contract_override(_contract(), "permanent", None, "intern")
    assert result[0].applicable is False
    assert result[0].score == 6  # untouched, not force-corrected to a guessed value


def test_permanent_traineeship_is_also_untrusted():
    assert apply_contract_override(_contract(), "permanent", None, "trainee")[0].applicable is False


def test_permanent_non_internship_scores_a_flat_10():
    result = apply_contract_override(_contract(), "permanent", None, "senior")
    assert result[0].applicable is True and result[0].score == 10


def test_fixed_term_scales_with_duration():
    short = apply_contract_override(_contract(), "fixed_term", 3, "intern")[0].score
    long = apply_contract_override(_contract(), "fixed_term", 24, "mid")[0].score
    assert short < long < 10  # never reaches permanent


def test_unstated_contract_type_is_not_applicable():
    assert apply_contract_override(_contract(), None, None)[0].applicable is False


def test_seniority_rank_uses_word_boundaries():
    assert seniority_rank("Senior Staff Engineer") == 4  # highest match wins
    assert seniority_rank("leadership skills") is None  # "lead" inside a word doesn't count


# --- location eligibility ------------------------------------------------------------

@pytest.mark.parametrize(
    "tier, expected",
    [("worldwide", 10), ("europe_eu", 9), ("emea", 8), ("cet", 8), ("europe_except_serbia", 4), ("us_only", 0), ("other_restricted", 0)],
)
def test_restriction_tier_scores(tier, expected):
    score, note = restriction_tier_score(tier)
    assert score == expected and tier in note


def test_without_eu_passport_eu_only_tiers_drop():
    assert restriction_tier_score("europe_eu", has_eu_passport=False)[0] == 2
    assert restriction_tier_score("europe_except_serbia", has_eu_passport=False)[0] == 0
    assert restriction_tier_score("emea", has_eu_passport=False)[0] == 8  # region-based, unaffected


def test_unstated_or_unknown_tier_is_left_to_the_model():
    assert restriction_tier_score(None) is None
    assert restriction_tier_score("some_value_the_model_invented") is None


# --- overall score ----------------------------------------------------------------

PRIORITIES = [
    {"key": "salary", "weight": 5},
    {"key": "location", "weight": 3},
    {"key": "contract_type_stability", "weight": 2},
]


def test_not_applicable_priorities_are_excluded_not_averaged_as_5():
    scores = [AttributeScore("salary", 10, "n"), AttributeScore("location", 0, "n", applicable=False)]
    assert compute_overall_score(scores, PRIORITIES) == 100.0


def test_nothing_applicable_falls_back_to_neutral_50():
    assert compute_overall_score([AttributeScore("salary", 9, "n", applicable=False)], PRIORITIES) == 50.0


def test_emphasize_extremes_keeps_0_5_10_fixed_and_pulls_the_middle_in():
    assert [_emphasize_extremes(s, 1.4) for s in (0, 5, 10)] == [0, 5, 10]
    assert 5 < _emphasize_extremes(7, 1.4) < 7


# --- full pipeline ------------------------------------------------------------------

def test_overrides_replace_llm_mistakes_before_the_overall_score():
    # The model misread three things: a yearly salary as monthly, a US-only
    # remote role as reachable, and an internship as a permanent job.
    llm_scores = [
        AttributeScore("salary", 2, "Compared 120,000 against the monthly floor as if both were EUR"),
        AttributeScore("location", 9, "Remote, looks fine"),
        AttributeScore("contract_type_stability", 10, "Permanent position"),
    ]
    facts = PostingFacts(
        salary_min=120000, salary_max=120000, salary_currency="USD", salary_period="yearly",
        contract_type="permanent", seniority="intern", location_restriction_tier="us_only",
    )
    scores, overall = score_posting(llm_scores, facts, SalaryExpectation(min=3000, target=6000, currency="EUR"), PRIORITIES)

    by_key = {s.key: s for s in scores}
    assert by_key["salary"].score == 10  # 120,000 USD/year = ~9,300 EUR/month, above the 6,000 target
    assert by_key["location"].score == 0  # US-only isn't reachable
    assert by_key["contract_type_stability"].applicable is False  # untrusted extraction
    assert overall == compute_overall_score(scores, PRIORITIES)
