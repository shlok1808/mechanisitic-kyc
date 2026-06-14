"""Ground-truth risk-tolerance rubric: scoring, tiering, and profile sampling.

See config.yaml's `rubric:` block for field weights and mappings.
"""

import yaml

TIERS = ["conservative", "moderate", "aggressive"]

DECOY_POOLS = {
    "net_worth_band": ["under_100k", "100k_to_500k", "500k_to_2m", "over_2m"],
    "occupation": [
        "high school teacher",
        "registered nurse",
        "software engineer",
        "small business owner",
        "graphic designer",
        "electrician",
        "accountant",
        "marketing manager",
        "warehouse supervisor",
        "freelance writer",
    ],
    "hobbies": [
        "hiking",
        "cooking",
        "woodworking",
        "gardening",
        "playing guitar",
        "running",
        "painting",
        "board games",
        "photography",
        "fishing",
    ],
    "city": [
        "Columbus",
        "Denver",
        "Austin",
        "Pittsburgh",
        "Sacramento",
        "Tampa",
        "Minneapolis",
        "Charlotte",
        "Albuquerque",
        "Portland",
    ],
}

NAMES = [
    ("Maria Lopez", "female"),
    ("James Carter", "male"),
    ("Wei Zhang", "neutral"),
    ("Aisha Khan", "female"),
    ("David Kim", "male"),
    ("Sofia Rossi", "female"),
    ("Marcus Johnson", "male"),
    ("Priya Patel", "female"),
    ("Liam O'Brien", "male"),
    ("Emma Nguyen", "female"),
    ("Noah Thompson", "male"),
    ("Olivia Brooks", "female"),
]


def load_rubric(config_path="config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["rubric"]


def field_subscore(field_spec, value):
    """Map a raw field value to a_i in [0,1] (0 = supports lower risk, 1 = higher)."""
    if "levels" in field_spec:
        levels = field_spec["levels"]
        return levels.index(value) / (len(levels) - 1)
    lo, hi = field_spec["range"]
    a = (value - lo) / (hi - lo)
    return 1 - a if field_spec.get("invert") else a


def risk_score(profile, rubric):
    """Continuous risk score s in [0,100] = 100 * sum(weight_i * a_i)."""
    return 100 * sum(
        spec["weight"] * field_subscore(spec, profile[field])
        for field, spec in rubric["fields"].items()
    )


def risk_tier(score, rubric):
    cutoffs = rubric["tier_cutoffs"]
    if score < cutoffs["conservative_max"]:
        return "conservative"
    if score <= cutoffs["moderate_max"]:
        return "moderate"
    return "aggressive"


def sample_rubric_fields(rubric, rng):
    profile = {}
    for field, spec in rubric["fields"].items():
        if "levels" in spec:
            profile[field] = rng.choice(spec["levels"])
        else:
            lo, hi = spec["range"]
            profile[field] = rng.randint(lo, hi)
    return profile


def sample_decoy_fields(rng):
    name, gender_hint = rng.choice(NAMES)
    return {
        "net_worth_band": rng.choice(DECOY_POOLS["net_worth_band"]),
        "occupation": rng.choice(DECOY_POOLS["occupation"]),
        "hobbies": rng.choice(DECOY_POOLS["hobbies"]),
        "city": rng.choice(DECOY_POOLS["city"]),
        "name": name,
        "gender_hint": gender_hint,
    }


def sample_profile(rubric, rng, profile_id):
    profile = {"profile_id": profile_id}
    profile.update(sample_rubric_fields(rubric, rng))
    profile.update(sample_decoy_fields(rng))
    score = risk_score(profile, rubric)
    profile["risk_score"] = round(score, 2)
    profile["tier"] = risk_tier(score, rubric)
    return profile
