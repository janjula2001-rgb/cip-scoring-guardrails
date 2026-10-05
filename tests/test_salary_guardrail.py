"""Currency conversion and pay-period handling - the two things the LLM got
wrong when comparing a posting's salary against the candidate's floor."""

from models import AttributeScore, SalaryExpectation
from salary_guardrail import (
    apply_salary_floor_override,
    deterministic_salary_score,
    posting_salary_monthly_eur,
    to_monthly_eur,
)


# --- currency ---------------------------------------------------------------

def test_usd_is_actually_converted_not_treated_as_eur():
    assert posting_salary_monthly_eur(5000, 5000, "USD") == 5000 * 0.93


def test_rsd_uses_the_fixed_peg():
    assert posting_salary_monthly_eur(117500, 117500, "RSD") == 1000.0


def test_eur_passes_through_unchanged():
    assert posting_salary_monthly_eur(2000, 2000, "EUR") == 2000.0


def test_unrecognized_currency_returns_none_instead_of_guessing():
    for garbage in ("CAD/USD", "XYZ", "�"):
        assert posting_salary_monthly_eur(1000, 1000, garbage) is None


def test_currency_lookup_is_case_insensitive_and_trims_whitespace():
    assert posting_salary_monthly_eur(1000, 1000, "usd") == posting_salary_monthly_eur(1000, 1000, " USD ")


def test_low_value_currency_is_not_inflated_or_mistaken_for_annual():
    # ~150,000 INR/month is a modest salary. Read as EUR it would look huge -
    # and also trip the "must be annual" ceiling and be divided by 12.
    result = posting_salary_monthly_eur(150000, 150000, "INR")
    assert 1000 < result < 2500


def test_unknown_currency_defers_to_the_model():
    assert deterministic_salary_score(1000, 1000, "XYZ", floor_monthly_eur=800) is None


def test_known_currency_is_scored():
    score, note = deterministic_salary_score(1000, 1000, "USD", floor_monthly_eur=500)
    assert score == 10  # 1000 * 0.93 = 930, well above a 500 floor
    assert "930" in note


# --- pay period ---------------------------------------------------------------

def test_known_yearly_period_is_divided_even_below_the_magnitude_ceiling():
    # 11,000/year is below the 12,000 "implausible monthly" ceiling, so a
    # magnitude guess alone reads it as 11,000 per MONTH.
    assert posting_salary_monthly_eur(11000, 11000, "EUR") == 11000
    assert abs(posting_salary_monthly_eur(11000, 11000, "EUR", "yearly") - 11000 / 12) < 0.01


def test_known_monthly_period_is_not_divided_even_above_the_ceiling():
    assert posting_salary_monthly_eur(13000, 13000, "EUR") == 13000 / 12
    assert posting_salary_monthly_eur(13000, 13000, "EUR", "monthly") == 13000


def test_yearly_expectation_is_normalized_to_monthly():
    assert to_monthly_eur(93000, "USD", "yearly") == (93000 * 0.93) / 12


# --- scoring curve and override -----------------------------------------------------

def test_floor_to_target_interpolates_from_5_to_10():
    assert deterministic_salary_score(1000, 1000, "EUR", 1000, 2000)[0] == 5
    assert deterministic_salary_score(1500, 1500, "EUR", 1000, 2000)[0] == 8
    assert deterministic_salary_score(2500, 2500, "EUR", 1000, 2000)[0] == 10
    assert deterministic_salary_score(500, 500, "EUR", 1000, 2000)[0] == 2


def test_override_replaces_the_llm_salary_score():
    llm = [AttributeScore(key="salary", score=3, note="model thinks it's low"), AttributeScore(key="location", score=7, note="ok")]
    expectation = SalaryExpectation(min=1000, currency="EUR")
    result = apply_salary_floor_override(llm, 60000, 60000, "EUR", expectation, "yearly")  # 5,000/month
    assert result[0].score == 10
    assert result[0].note.startswith("Deterministic")
    assert result[1] == llm[1]  # other priorities untouched


def test_no_salary_information_is_marked_not_applicable():
    llm = [AttributeScore(key="salary", score=6, note="guess")]
    result = apply_salary_floor_override(llm, None, None, None, SalaryExpectation(min=1000))
    assert result[0].applicable is False


def test_no_floor_set_leaves_the_llm_score_alone():
    llm = [AttributeScore(key="salary", score=6, note="model")]
    assert apply_salary_floor_override(llm, 1000, 1000, "EUR", None) == llm
