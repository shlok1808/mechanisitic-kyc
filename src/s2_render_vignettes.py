"""S2: render synthetic client profiles into explicit/implicit vignettes + pairs.

Reads data/profiles.jsonl (from S1), renders each profile as an explicit + implicit twin,
paraphrases both (same backend, §5.3), and writes:
    data/vignettes/explicit.jsonl
    data/vignettes/implicit.jsonl
    data/vignettes/pairs.jsonl     (matched conservative/aggressive counterfactuals)

Vignettes store the CLIENT NARRATIVE ONLY; advice-task options are composed at inference
(S3). Run S2B QC afterwards as the go/no-go gate.

Usage:
    python src/s2_render_vignettes.py [--config config.yaml] [--limit N]
                                      [--no-paraphrase] [--backend gemini|openai]
"""

import argparse
import json
import random
from pathlib import Path

import yaml

from rubric import (
    is_contradictory, risk_score, risk_tier, sample_decoy_fields, sample_rubric_fields,
)
from templates import build_banned_regex, find_banned, pick_template_id, render
from paraphrase import ParaphraseClient


def load_profiles(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def draw_tier_rubric(rubric, rng, target_tier, max_tries=2000):
    """Reject-sample a rubric-field draw whose tier == target_tier."""
    for _ in range(max_tries):
        fields = sample_rubric_fields(rubric, rng)
        score = risk_score(fields, rubric)
        if risk_tier(score, rubric) == target_tier:
            return fields, score
    raise RuntimeError(f"could not draw a {target_tier} profile")


def build_twin_jobs(profiles, seed, rubric):
    """One render job per (profile, tier). Returns list of job dicts with template_text."""
    jobs = []
    for p in profiles:
        tid = pick_template_id(p["profile_id"], seed)
        contradictory = is_contradictory(p, rubric)
        for vtype in ("explicit", "implicit"):
            rng = random.Random(f"{seed}|{p['profile_id']}|{vtype}")
            jobs.append({
                "vignette_id": f"v_{p['profile_id']}_{vtype}",
                "profile_id": p["profile_id"],
                "pair_id": None,
                "tier": p["tier"],
                "risk_score": p["risk_score"],
                "vignette_type": vtype,
                "template_id": tid,
                "contradictory": contradictory,
                "template_text": render(p, tid, vtype, rng),
            })
    return jobs


def build_pair_jobs(rubric, n_pairs, render_tier, seed):
    """n_pairs matched counterfactuals: shared filler/name/template, cons vs aggr rubric."""
    rng = random.Random(f"{seed}|pairs")
    jobs = []
    for i in range(n_pairs):
        pair_id = f"pair_{i:04d}"
        decoy = sample_decoy_fields(rng)                       # shared name/occupation/city/...
        tid = pick_template_id(pair_id, seed)
        for side, tier in (("cons", "conservative"), ("aggr", "aggressive")):
            fields, score = draw_tier_rubric(rubric, rng, tier)
            profile = {"profile_id": f"{pair_id}_{side}", **fields, **decoy,
                       "risk_score": round(score, 2), "tier": tier}
            prng = random.Random(f"{seed}|{pair_id}|{side}")
            jobs.append({
                "vignette_id": f"v_{pair_id}_{side}",
                "profile_id": profile["profile_id"],
                "pair_id": pair_id,
                "tier": tier,
                "risk_score": profile["risk_score"],
                "vignette_type": render_tier,
                "template_id": tid,
                "contradictory": is_contradictory(profile, rubric),
                "template_text": render(profile, tid, render_tier, prng),
            })
    return jobs


def finalize(job, text, model, banned_regex):
    """Attach paraphrase output + banned scan, drop the intermediate template_text."""
    banned = find_banned(text, banned_regex) if job["vignette_type"] == "implicit" else []
    return {
        "vignette_id": job["vignette_id"],
        "profile_id": job["profile_id"],
        "pair_id": job["pair_id"],
        "tier": job["tier"],
        "risk_score": job["risk_score"],
        "vignette_type": job["vignette_type"],
        "template_id": job["template_id"],
        "contradictory": job["contradictory"],
        "paraphrase_model": model,
        "banned_terms_found": banned,
        "text": text,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def submit_batch_and_exit(client, all_jobs, results_dir):
    """Write uncached jobs to a Batch API input file, submit, persist batch_id + sidecar."""
    results_dir.mkdir(parents=True, exist_ok=True)
    lines, sidecar = client.build_batch([(j["template_text"], j["vignette_type"]) for j in all_jobs])
    cached = len(all_jobs) - len(lines)
    if not lines:
        print(f"All {len(all_jobs)} jobs already cached -- nothing to batch. "
              f"Run normally to render + QC.")
        return

    input_path = results_dir / "s2_batch_input.jsonl"
    with open(input_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    (results_dir / "s2_batch_sidecar.json").write_text(json.dumps(sidecar))

    batch_id = client.submit_batch(input_path)
    (results_dir / "s2_batch_id.txt").write_text(batch_id)
    print(f"Submitted {len(lines)} uncached jobs ({cached} already cached) to Batch API.")
    print(f"  batch_id: {batch_id}  ->  {results_dir / 's2_batch_id.txt'}")
    print(f"  collect with: python3 src/s2_collect_batch.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--profiles", default=None,
                    help="input profiles.jsonl (default: <data_dir>/profiles.jsonl)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: config paths.vignettes_dir)")
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="override pair count (0 to skip pairs, e.g. for edge sets)")
    ap.add_argument("--limit", type=int, default=None, help="cap profiles (dev)")
    ap.add_argument("--no-paraphrase", action="store_true", help="template-only, no API")
    ap.add_argument("--backend", default=None, help="override config paraphrase backend")
    ap.add_argument("--batch", action="store_true",
                    help="submit uncached jobs to the OpenAI Batch API and exit "
                         "(collect later with src/s2_collect_batch.py)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    rubric = cfg["rubric"]
    vcfg = cfg["vignette"]
    pcfg = vcfg["paraphrase"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["vignettes_dir"])

    profiles_path = (Path(args.profiles) if args.profiles
                     else Path(cfg["paths"]["data_dir"]) / "profiles.jsonl")
    profiles = load_profiles(profiles_path)
    # S1 writes profiles in generation order (aggressive tier clusters at the end), so
    # shuffle deterministically before any --limit slice keeps dev subsets tier-representative.
    random.Random(seed).shuffle(profiles)
    if args.limit:
        profiles = profiles[: args.limit]

    banned_words = vcfg["banned_lexicon"]
    banned_regex = build_banned_regex(banned_words)

    if args.n_pairs is not None:
        n_pairs = args.n_pairs
    elif args.limit:
        n_pairs = max(2, args.limit // 4)
    else:
        n_pairs = vcfg["n_pairs"]
    twin_jobs = build_twin_jobs(profiles, seed, rubric)
    pair_jobs = build_pair_jobs(rubric, n_pairs, vcfg["pair_render_tier"], seed)
    all_jobs = twin_jobs + pair_jobs
    print(f"Rendered {len(twin_jobs)} twin + {len(pair_jobs)} pair templates; paraphrasing...")

    client = ParaphraseClient(
        backend=args.backend or pcfg["backend"], model=pcfg["model"],
        banned_words=banned_words, banned_regex=banned_regex,
        cache_dir=Path(cfg["paths"]["cache_dir"]) / "paraphrase",
        temperature=pcfg["temperature"], max_retries=pcfg["max_retries"],
        max_workers=pcfg.get("max_workers", 8), enabled=not args.no_paraphrase,
    )

    if args.batch:
        submit_batch_and_exit(client, all_jobs, Path(cfg["paths"]["results_dir"]))
        return

    outputs = client.paraphrase_batch([(j["template_text"], j["vignette_type"]) for j in all_jobs])
    rows = [finalize(j, t, m, banned_regex) for j, (t, m) in zip(all_jobs, outputs)]

    by_file = {"explicit": [], "implicit": [], "pairs": []}
    for r in rows:
        if r["pair_id"] is not None:
            by_file["pairs"].append(r)
        else:
            by_file[r["vignette_type"]].append(r)

    for name, rs in by_file.items():
        write_jsonl(out_dir / f"{name}.jsonl", rs)

    leaks = sum(1 for r in rows if r["banned_terms_found"])
    fallbacks = sum(1 for r in rows if r["paraphrase_model"] == "template-only" and not args.no_paraphrase)
    contra = sum(1 for r in rows if r["contradictory"])
    print(f"Wrote {len(by_file['explicit'])} explicit, {len(by_file['implicit'])} implicit, "
          f"{len(by_file['pairs'])} pair rows to {out_dir}")
    print(f"  contradictory: {contra} rows  |  banned-term leaks: {leaks}  |  "
          f"template-only fallbacks: {fallbacks}")
    if leaks:
        print("  WARNING: leaks present -- S2B QC will hard-fail until resolved.")


if __name__ == "__main__":
    main()
