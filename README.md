# cip-scoring-guardrails

A small, self-contained sample from Career Intelligence Platform, a desktop job-search agent I built. It collects postings from 9 job boards and 38 company career pages and scores each one against a candidate profile and weighted priorities. An LLM reads each posting and scores every priority from 0 to 10; this sample shows the code that sits between those LLM scores and the final ranking.

## Why these checks exist

Some questions have a right answer that the LLM kept getting wrong, so code decides them instead of the model:

- **Currency.** The model was unreliable at converting a USD, GBP or INR salary before comparing it with a EUR minimum. Code converts with fixed rates, and returns "can't tell" for a currency it doesn't know rather than treating it as EUR (an INR figure read as EUR makes a modest salary look like an excellent one).
- **Salary period.** The model mixed up yearly and monthly figures. A posting paying about 100,000 EUR a year was scored as a poor match against a monthly minimum.
- **Location eligibility.** Remote postings often state who can apply ("US only", "EU work permit required", "EMEA"). The model treated restricted roles as reachable for a candidate based in Serbia.
- **Contract type.** A posting describing a three-month internship was extracted as a permanent job. A "permanent internship" is now treated as an extraction error, not a fact.

When the code can't decide (an unknown currency, no stated eligibility), it leaves the model's score alone instead of guessing.

## How it fits together

```
LLM per-attribute scores  ->  deterministic overrides  ->  weighted overall score
(0-10 + a note each)          (salary, location, contract)   (compute_overall_score)
```

- `salary_guardrail.py` converts any posting salary to monthly EUR and scores it against the candidate's minimum and target.
- `scoring.py` holds the location-eligibility and contract overrides and `compute_overall_score`, a weighted average where priorities that couldn't be judged are left out rather than counted as a neutral 5.
- `models.py` has minimal dataclasses standing in for the app's own types.

The functions are copied unchanged from the app. The `score_posting` pipeline at the end of `scoring.py` was written for this sample to show the order; in the app the same step also runs overrides for location text, tech stack and seniority, which aren't included here.

## Running the tests

Needs Python 3.10+ and pytest. No other dependencies.

```bash
pip install pytest
pytest
```

The rest of the project is private.
