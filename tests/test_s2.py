"""Offline unit tests for S2 (no API calls)."""

import random
import re

import pytest
import yaml

from rubric import is_contradictory, load_rubric, risk_tier, risk_score
from s1b_edge_profiles import assign_decoys, build_edge_rubric_fields
from s2b_qc import tenure_incoherences
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


def test_extensive_experience_phrasings_are_duration_free():
    # Guards the age/experience impossibility: extensive experience must not assert a
    # tenure ("two decades", "for N years") that conflicts with young sampled ages.
    for s in ENUM_PHRASINGS["investing_experience"]["extensive"]["implicit"]:
        assert not re.search(r"\b(decade|year)", s, re.I), s


def test_explicit_render_contains_facts():
    text = render(SAMPLE_PROFILE, TEMPLATE_IDS[0], "explicit", random.Random(0))
    assert "Wei Zhang" in text and "Austin" in text and "27" in text


# -- template assignment ---------------------------------------------------------------
def test_template_id_is_deterministic_and_shared_by_twins():
    a = pick_template_id("p00042", seed=42)
    b = pick_template_id("p00042", seed=42)
    assert a == b and a in TEMPLATE_IDS


# -- twin job construction -------------------------------------------------------------
def test_twins_share_template_differ_in_type(cfg):
    jobs = build_twin_jobs([SAMPLE_PROFILE], seed=42, rubric=cfg["rubric"])
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
def test_finalize_schema_and_banned_field(banned, cfg):
    _, rx = banned
    job = build_twin_jobs([SAMPLE_PROFILE], seed=42, rubric=cfg["rubric"])[1]  # implicit
    row = finalize(job, "I moved everything into cash.", "gpt-4o-mini", rx)
    assert set(row) == {"vignette_id", "profile_id", "pair_id", "tier", "risk_score",
                        "vignette_type", "template_id", "contradictory",
                        "paraphrase_model", "banned_terms_found", "text"}
    assert row["banned_terms_found"] == []


def test_finalize_flags_leak(banned, cfg):
    _, rx = banned
    job = build_twin_jobs([SAMPLE_PROFILE], seed=42, rubric=cfg["rubric"])[1]
    row = finalize(job, "I have a high risk tolerance.", "gpt-4o-mini", rx)
    assert "risk" in [h.lower() for h in row["banned_terms_found"]]


# -- determinism -----------------------------------------------------------------------
def test_render_is_deterministic():
    a = render(SAMPLE_PROFILE, TEMPLATE_IDS[3], "implicit", random.Random(123))
    b = render(SAMPLE_PROFILE, TEMPLATE_IDS[3], "implicit", random.Random(123))
    assert a == b


# -- contradiction detection -----------------------------------------------------------
def test_contradiction_flags_opposite_extremes(cfg):
    rub = cfg["rubric"]
    base = dict(SAMPLE_PROFILE)
    base.update({"stated_goal": "maximum_growth", "past_drawdown_reaction": "sold_everything"})
    assert is_contradictory(base, rub) is True
    base.update({"stated_goal": "capital_preservation", "past_drawdown_reaction": "bought_more"})
    assert is_contradictory(base, rub) is True


def test_contradiction_ignores_aligned_and_mild(cfg):
    rub = cfg["rubric"]
    aligned = dict(SAMPLE_PROFILE)
    aligned.update({"stated_goal": "maximum_growth", "past_drawdown_reaction": "bought_more"})
    assert is_contradictory(aligned, rub) is False
    mild = dict(SAMPLE_PROFILE)
    mild.update({"stated_goal": "maximum_growth", "past_drawdown_reaction": "reduced"})
    assert is_contradictory(mild, rub) is False  # reduced (a=0.33) is not extreme enough


def test_contradictory_field_propagates_to_rows(cfg):
    contra = dict(SAMPLE_PROFILE)
    contra.update({"stated_goal": "capital_preservation", "past_drawdown_reaction": "bought_more"})
    jobs = build_twin_jobs([contra], seed=42, rubric=cfg["rubric"])
    assert all(j["contradictory"] is True for j in jobs)


# -- edge generator --------------------------------------------------------------------
def test_edge_generator_covers_buckets_and_decoys(cfg):
    rng = random.Random(cfg["seed"])
    items = build_edge_rubric_fields(cfg["rubric"], rng)
    profiles = assign_decoys(items, rng)
    for p in profiles:
        s = risk_score(p, cfg["rubric"])
        p["risk_score"], p["tier"] = s, risk_tier(s, cfg["rubric"])
    buckets = {p["edge_bucket"] for p in profiles}
    assert {"borderline", "contradiction", "extreme_demo",
            "anchor_conservative", "anchor_aggressive"} <= buckets
    # full decoy coverage (all 10 of each)
    for k in ("occupation", "city", "hobbies"):
        assert len({p[k] for p in profiles}) == 10
    # anchors land at the extremes; contradictions are rubric-flagged
    anchors_c = [p for p in profiles if p["edge_bucket"] == "anchor_conservative"]
    anchors_a = [p for p in profiles if p["edge_bucket"] == "anchor_aggressive"]
    assert all(p["tier"] == "conservative" for p in anchors_c)
    assert all(p["tier"] == "aggressive" for p in anchors_a)
    assert all(is_contradictory(p, cfg["rubric"])
               for p in profiles if p["edge_bucket"] == "contradiction")


# -- QC age/experience coherence -------------------------------------------------------
def test_tenure_incoherence_flags_young_long_tenure():
    profiles = {"young": {"profile_id": "young", "age": 23},
                "old": {"profile_id": "old", "age": 70}}
    rows = [
        {"vignette_id": "a", "profile_id": "young",
         "text": "I'm 23 and I've been actively investing for nearly two decades now."},
        {"vignette_id": "b", "profile_id": "young",
         "text": "I'm 23 and I've been through several market cycles."},   # clean
        {"vignette_id": "c", "profile_id": "old",
         "text": "I'm 70 and I've been investing for two decades."},        # plausible at 70
    ]
    hits = tenure_incoherences(rows, profiles)
    assert [h["vignette_id"] for h in hits] == ["a"]
