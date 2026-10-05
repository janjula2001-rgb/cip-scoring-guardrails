"""Minimal stand-ins for the app's own data types (pydantic models there),
reduced to the fields the guardrail code actually reads."""

from dataclasses import dataclass


@dataclass
class AttributeScore:
    """One priority's 0-10 score from the LLM, possibly replaced by a guardrail."""

    key: str
    score: int
    note: str
    # False = this priority couldn't be judged for this posting; it's left out
    # of the weighted average instead of being counted as a neutral 5.
    applicable: bool = True


@dataclass
class SalaryExpectation:
    """The candidate's salary floor (min) and the level that earns a 10 (target)."""

    min: float | None = None
    target: float | None = None
    currency: str | None = None
    period: str = "monthly"  # what min/target are denominated in: "monthly" or "yearly"
