"""Deterministic salary scoring: currency conversion and pay-period handling
are done in code, not left to the model.

LLMs were unreliable here in two concrete ways: comparing a USD/INR/... figure
against a EUR floor at face value, and treating a yearly figure as monthly
(a posting at ~100,000 EUR/year was scored as a poor match against a monthly
floor). This module normalizes every figure to monthly EUR and computes the
"salary" score itself.
"""

from models import AttributeScore, SalaryExpectation

# Long-standing quasi-managed NBS peg (~117.5 RSD/EUR for years) - a fixed
# approximation is fine here, this is a rough threshold check, not
# accounting, and doesn't need a live FX rate.
RSD_PER_EUR = 117.5

# Real bug this fixes: every currency except RSD was being compared against
# the EUR floor/target at face value, with zero conversion - silently wrong
# for USD, and catastrophically wrong for lower-value currencies like
# INR/PHP/BRL, which would show as a wildly inflated "amazing salary" instead
# of a modest one - worse, a large INR monthly figure would also trip
# IMPLAUSIBLE_MONTHLY_EUR_CEILING below and get divided by 12 as if it were
# annual, compounding the error further. Rough fixed-rate approximations for
# every currency actually observed in extracted/estimated salary data.
# 1 unit of the currency = this many EUR.
_CURRENCY_TO_EUR: dict[str, float] = {
    "EUR": 1.0,
    "RSD": 1 / RSD_PER_EUR,
    "USD": 0.93,
    "GBP": 1.17,
    "CHF": 1.05,
    "CAD": 0.68,
    "AUD": 0.61,
    "INR": 0.011,
    "PHP": 0.016,
    "BRL": 0.16,
    "KM": 0.5113,  # Bosnia-Herzegovina convertible mark, pegged by law to 1.95583 KM = 1 EUR
}

# Posting salaries (stated or estimated) don't always carry an explicit
# monthly-vs-yearly period - extraction just pulls raw numbers. When a figure
# is this implausibly high to be a real *monthly* net salary in this market,
# it's almost certainly annual.
IMPLAUSIBLE_MONTHLY_EUR_CEILING = 12000


def _to_eur(amount: float, currency: str | None) -> float | None:
    """None when the currency isn't one we know how to convert, rather than
    silently assuming it's already EUR (the exact bug fixed above) - the
    caller falls back to leaving the comparison to the model's own judgment
    in that case ("don't guess when unsure")."""
    rate = _CURRENCY_TO_EUR.get((currency or "EUR").strip().upper())
    return amount * rate if rate is not None else None


def to_monthly_eur(amount: float, currency: str | None, period: str) -> float | None:
    """Normalizes an explicit (amount, currency, period) figure - used for
    the candidate's own salary expectation, where the period is known
    directly rather than inferred."""
    eur = _to_eur(amount, currency)
    if eur is None:
        return None
    return eur / 12 if period == "yearly" else eur


def posting_salary_monthly_eur(
    min_: float | None, max_: float | None, currency: str | None, period: str | None = None
) -> float | None:
    """Normalizes a posting's stated/estimated salary to monthly EUR. When the
    period is known ("monthly"/"yearly") it's used directly; when it's None
    the period is inferred from magnitude. Returns None if there's no salary
    figure to normalize at all, or if the currency isn't one we can convert."""
    if min_ is None and max_ is None:
        return None
    amount = (min_ + max_) / 2 if min_ is not None and max_ is not None else (min_ if min_ is not None else max_)
    eur = _to_eur(amount, currency)
    if eur is None:
        return None
    if period == "yearly":
        return eur / 12
    if period == "monthly":
        return eur
    return eur / 12 if eur > IMPLAUSIBLE_MONTHLY_EUR_CEILING else eur


def salary_floor_monthly_eur(salary_expectation: SalaryExpectation | None) -> float | None:
    """The candidate's minimum acceptable salary, normalized to monthly EUR,
    or None if unset - the deterministic score is a no-op until it is."""
    if salary_expectation is None or salary_expectation.min is None:
        return None
    return to_monthly_eur(salary_expectation.min, salary_expectation.currency, salary_expectation.period)


def salary_target_monthly_eur(salary_expectation: SalaryExpectation | None) -> float | None:
    """The salary at or above which the deterministic score is a flat 10, or
    None if unset - falls back to the simpler hard-floor behavior below."""
    if salary_expectation is None or salary_expectation.target is None:
        return None
    return to_monthly_eur(salary_expectation.target, salary_expectation.currency, salary_expectation.period)


def deterministic_salary_score(
    resolved_min: float | None,
    resolved_max: float | None,
    resolved_currency: str | None,
    floor_monthly_eur: float | None,
    target_monthly_eur: float | None = None,
    resolved_period: str | None = None,
) -> tuple[int, str] | None:
    """Returns (score, note) to replace whatever the scoring model produced
    for the "salary" priority, or None if there's nothing to compare (no
    floor set, or no salary info at all for this posting) - in which case
    the model's own score is left alone.

    With a target set (and greater than the floor): at or above target
    scores a flat 10; between floor and target, score interpolates linearly
    from 5 up to 10; below floor, that same line continues down from 5 to 0.

    Without a target (or one at/below the floor, which isn't a valid range):
    falls back to a simple hard floor - flat 10 at or above the minimum,
    scaled down proportionally below it."""
    if floor_monthly_eur is None or floor_monthly_eur <= 0:
        return None
    posting_monthly = posting_salary_monthly_eur(resolved_min, resolved_max, resolved_currency, resolved_period)
    if posting_monthly is None:
        return None

    if target_monthly_eur is not None and target_monthly_eur > floor_monthly_eur:
        if posting_monthly >= target_monthly_eur:
            score = 10
        elif posting_monthly >= floor_monthly_eur:
            frac = (posting_monthly - floor_monthly_eur) / (target_monthly_eur - floor_monthly_eur)
            score = round(5 + 5 * frac)
        else:
            score = round(5 * max(0.0, posting_monthly / floor_monthly_eur))
        note = (
            f"Deterministic: posting salary normalizes to ~{posting_monthly:,.0f} EUR/month net "
            f"(your net minimum ~{floor_monthly_eur:,.0f}, target for a 10 ~{target_monthly_eur:,.0f} EUR/month)."
        )
    else:
        ratio = posting_monthly / floor_monthly_eur
        score = 10 if ratio >= 1.0 else max(0, round(9 * ratio))
        verdict = "meets or exceeds" if ratio >= 1.0 else "is below"
        note = (
            f"Deterministic: posting salary normalizes to ~{posting_monthly:,.0f} EUR/month net, which "
            f"{verdict} your net minimum of ~{floor_monthly_eur:,.0f} EUR/month."
        )
    return max(0, min(10, score)), note


def apply_salary_floor_override(
    per_attribute: list[AttributeScore],
    resolved_min: float | None,
    resolved_max: float | None,
    resolved_currency: str | None,
    salary_expectation: SalaryExpectation | None,
    resolved_period: str | None = None,
) -> list[AttributeScore]:
    """Replaces the "salary" entry in a scoring result with a deterministic
    one, if the candidate has set a minimum. LLMs - especially smaller local
    ones - are unreliable at correctly reasoning about currency conversion
    and, worse, salary *period* (monthly vs yearly) when comparing a
    posting's figure against the minimum; doing this comparison in code
    instead matches how the overall score is also computed deterministically
    rather than trusting the model's own aggregation."""
    override = deterministic_salary_score(
        resolved_min,
        resolved_max,
        resolved_currency,
        salary_floor_monthly_eur(salary_expectation),
        salary_target_monthly_eur(salary_expectation),
        resolved_period,
    )
    if override is not None:
        score, note = override
        return [AttributeScore(key="salary", score=score, note=note) if a.key == "salary" else a for a in per_attribute]
    if resolved_min is None and resolved_max is None:
        # No stated figure and no salary estimate at all - genuinely nothing
        # to judge, regardless of whether a minimum is set.
        return [
            AttributeScore(key="salary", score=a.score, note="Not applicable: no salary information available for this posting.", applicable=False)
            if a.key == "salary"
            else a
            for a in per_attribute
        ]
    return per_attribute
