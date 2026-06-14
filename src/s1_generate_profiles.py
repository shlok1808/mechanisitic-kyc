"""S1: sample synthetic client profiles and write a tier-balanced profiles.jsonl.

Usage: python src/s1_generate_profiles.py [--config config.yaml] [--out data/profiles.jsonl]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from rubric import TIERS, sample_profile


def generate_balanced_profiles(rubric, n_per_tier, seed):
    """Reject-sample profiles until each tier has n_per_tier members."""
    rng = random.Random(seed)
    counts = Counter()
    profiles = []
    drawn = 0
    while any(counts[tier] < n_per_tier for tier in TIERS):
        profile = sample_profile(rubric, rng, profile_id=f"p{drawn:05d}")
        drawn += 1
        tier = profile["tier"]
        if counts[tier] < n_per_tier:
            counts[tier] += 1
            profiles.append(profile)
    return profiles, drawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/profiles.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rubric = config["rubric"]
    n_per_tier = config["data"]["n_profiles"] // len(TIERS)
    profiles, drawn = generate_balanced_profiles(rubric, n_per_tier, config["seed"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for profile in profiles:
            f.write(json.dumps(profile) + "\n")

    counts = Counter(p["tier"] for p in profiles)
    print(f"Wrote {len(profiles)} profiles to {out_path} ({drawn} sampled)")
    for tier in TIERS:
        print(f"  {tier}: {counts[tier]}")


if __name__ == "__main__":
    main()
