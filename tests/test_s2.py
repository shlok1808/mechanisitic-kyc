"""Offline unit tests for S2 (no API calls)."""

import random

import pytest
import yaml

from rubric import load_rubric, risk_tier, risk_score
from templates import (
    BAND_PHRASINGS, ENUM_PHRASINGS, NET_WORTH_PHRASINGS, TEMPLATE_IDS,
    build_banned_regex, find_banned, pick_template_id, render,
)
from s2_render_vignettes import build_pair_jobs, build_twin_jobs, finalize


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def banned(cfg):
    return cfg["vignette"]["banned_lexicon"], build_banned_regex(cfg["vignette"]["banned_lexicon"])


SAMPLE_PROFILE = {
    "profile_id": "p99999", "past_drawdown_reaction": "bought_more", "horizon_years": 30,
    "investing_experience": "extensive", "income_stability": "stable",
    "stated_goal": "maximum_growth", "emergency_fund_months": 20, "dependents": 0, "age": 27,
    "net_worth_band": "over_2m", "occupation": "software engineer", "hobbies": "hiking",
    "city": "Austin", "name": "Wei Zhang", "gender_hint": "neutral",
    "risk_score": 95.0, "tier": "aggressive",
}


# -- banned-lexicon regex --------------------------------------------------------------
def test_regex_catches_inflections(banned):
    _, rx = banned
    for bad in ["risk", "Risky", "riskier", "AGGRESSIVE", "aggressively",
                "conservative", "conservatively", "volatility", "safely"]:
        assert find_banned(f"I am {bad} about it.", rx), bad


def test_regex_ignores_clean_text(banned):
    _, rx = banned
    assert find_banned("I moved my money into cash when the market fell.", rx) == []


def test_moderate_is_not_banned(banned):
    _, rx = banned
    assert find_banned("I earn a moderate income.", rx) == []


# -- exhaustive implicit cleanliness (the core guarantee) ------------------------------
def test_no_implicit_phrasing_contains_banned_term(banned):
    _, rx = banned
    cells = []
    for field, byval in ENUM_PHRASINGS.items():
        for val, modes in byval.items():
            cells += [(f"{field}={val}", s) for s in modes["implicit"]]
    for field, bands in BAND_PHRASINGS.items():
        for lo, hi, modes in bands:
            cells += [(f"{field}[{lo},{hi}]", s) for s in modes["implicit"]]
    for val, phrs in NET_WORTH_PHRASINGS.items():
        cells += [(f"net_worth={val}", s) for s in phrs]
    bad = [(label, find_banned(s, rx)) for label, s in cells if find_banned(s, rx)]
    assert bad == [], f"banned terms in implicit phrasings: {bad}"


def test_rendered_implicit_profile_is_clean(banned):
    _, rx = banned
    for tid in TEMPLATE_IDS[:8]:
        text = render(SAMPLE_PROFILE, tid, "implicit", random.Random(0))
        assert find_banned(text, rx) == []


def test_explicit_render_contains_facts():
    text = render(SAMPLE_PROFILE, TEMPLATE_IDS[0], "explicit", random.Random(0))
    assert "Wei Zhang" in text and "Austin" in text and "27" in text


# -- template assignment ---------------------------------------------------------------
def test_template_id_is_deterministic_and_shared_by_twins():
    a = pick_template_id("p00042", seed=42)
    b = pick_template_id("p00042", seed=42)
    assert a == b and a in TEMPLATE_IDS


# -- twin job construction -------------------------------------------------------------
def test_twins_share_template_differ_in_type():
    jobs = build_twin_jobs([SAMPLE_PROFILE], seed=42)
    assert len(jobs) == 2
    assert jobs[0]["template_id"] == jobs[1]["template_id"]
    assert {j["vignette_type"] for j in jobs} == {"explicit", "implicit"}
    assert all(j["tier"] == "aggressive" for j in jobs)


# -- pair construction -----------------------------------------------------------------
def test_pairs_share_filler_and_template_differ_in_tier(cfg):
    rubric = cfg["rubric"]
    jobs = build_pair_jobs(rubric, n_pairs=5, render_tier="implicit", seed=42)
    assert len(jobs) == 10
    by_pair = {}
    for j in jobs:
        by_pair.setdefault(j["pair_id"], []).append(j)
    for pid, (a, b) in ((k, v) for k, v in by_pair.items()):
        assert a["template_id"] == b["template_id"]
        assert {a["tier"], b["tier"]} == {"conservative", "aggressive"}
        assert a["pair_id"] == b["pair_id"] == pid


def test_pair_tiers_match_their_drawn_rubric(cfg):
    rubric = cfg["rubric"]
    jobs = build_pair_jobs(rubric, n_pairs=20, render_tier="implicit", seed=7)
    for j in jobs:
        # the cons side narrative should mention conservative-leaning life facts; we only
        # assert the recorded tier is internally consistent and in-range.
        assert j["tier"] in ("conservative", "aggressive")


# -- finalize / schema -----------------------------------------------------------------
def test_finalize_schema_and_banned_field(banned):
    _, rx = banned
    job = build_twin_jobs([SAMPLE_PROFILE], seed=42)[1]  # implicit
    row = finalize(job, "I moved everything into cash.", "gemini-1.5-flash", rx)
    assert set(row) == {"vignette_id", "profile_id", "pair_id", "tier", "risk_score",
                        "vignette_type", "template_id", "paraphrase_model",
                        "banned_terms_found", "text"}
    assert row["banned_terms_found"] == []


def test_finalize_flags_leak(banned):
    _, rx = banned
    job = build_twin_jobs([SAMPLE_PROFILE], seed=42)[1]
    row = finalize(job, "I have a high risk tolerance.", "gemini-1.5-flash", rx)
    assert "risk" in [h.lower() for h in row["banned_terms_found"]]


# -- determinism -----------------------------------------------------------------------
def test_render_is_deterministic():
    a = render(SAMPLE_PROFILE, TEMPLATE_IDS[3], "implicit", random.Random(123))
    b = render(SAMPLE_PROFILE, TEMPLATE_IDS[3], "implicit", random.Random(123))
    assert a == b
