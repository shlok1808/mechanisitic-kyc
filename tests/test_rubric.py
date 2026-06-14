import random
from collections import Counter

import pytest

from rubric import TIERS, load_rubric, risk_score, risk_tier, sample_profile
from s1_generate_profiles import generate_balanced_profiles

ALL_CONSERVATIVE = {
    "past_drawdown_reaction": "sold_everything",
    "horizon_years": 1,
    "investing_experience": "none",
    "income_stability": "precarious",
    "stated_goal": "capital_preservation",
    "emergency_fund_months": 0,
    "dependents": 4,
    "age": 75,
}

ALL_AGGRESSIVE = {
    "past_drawdown_reaction": "bought_more",
    "horizon_years": 35,
    "investing_experience": "extensive",
    "income_stability": "stable",
    "stated_goal": "maximum_growth",
    "emergency_fund_months": 24,
    "dependents": 0,
    "age": 23,
}

BORING_MIDDLE = {
    "past_drawdown_reaction": "held",
    "horizon_years": 18,
    "investing_experience": "some",
    "income_stability": "variable",
    "stated_goal": "balanced_growth",
    "emergency_fund_months": 12,
    "dependents": 2,
    "age": 49,
}


@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


def test_weights_sum_to_one(rubric):
    total = sum(spec["weight"] for spec in rubric["fields"].values())
    assert total == pytest.approx(1.0)


def test_all_conservative_profile_scores_zero(rubric):
    score = risk_score(ALL_CONSERVATIVE, rubric)
    assert score == pytest.approx(0.0)
    assert risk_tier(score, rubric) == "conservative"


def test_all_aggressive_profile_scores_hundred(rubric):
    score = risk_score(ALL_AGGRESSIVE, rubric)
    assert score == pytest.approx(100.0)
    assert risk_tier(score, rubric) == "aggressive"


def test_boring_middle_profile_is_moderate(rubric):
    score = risk_score(BORING_MIDDLE, rubric)
    assert score == pytest.approx(56.33, abs=0.01)
    assert risk_tier(score, rubric) == "moderate"


def test_balanced_generation_hits_target_counts(rubric):
    profiles, drawn = generate_balanced_profiles(rubric, n_per_tier=50, seed=42)
    counts = Counter(p["tier"] for p in profiles)
    for tier in TIERS:
        assert counts[tier] == 50
    assert drawn >= len(profiles)


def test_naive_distribution_matches_sanity_check(rubric):
    rng = random.Random(0)
    n = 50000
    counts = Counter()
    for i in range(n):
        profile = sample_profile(rubric, rng, profile_id=str(i))
        counts[profile["tier"]] += 1
    fractions = {tier: counts[tier] / n for tier in TIERS}
    assert fractions["conservative"] == pytest.approx(0.26, abs=0.03)
    assert fractions["moderate"] == pytest.approx(0.64, abs=0.03)
    assert fractions["aggressive"] == pytest.approx(0.10, abs=0.03)
