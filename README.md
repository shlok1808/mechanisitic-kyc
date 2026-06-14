# Mechanistic KYC

Looking inside an LLM financial advisor to audit *Know Your Customer* (suitability).

**Thesis:** Behavioral testing can show an AI advisor gave bad advice. Only mechanistic
(internal) testing shows *why* — and the *why* determines the fix.

> Status: direction pending team-sync confirmation (June 2026). Target: ICAIF 2026 (Aug 2).

## Research questions
- **RQ1 (Existence):** Does the model hold a linearly-decodable internal estimate of the
  *client's* risk tolerance, even when the client never uses risk words?
- **RQ2 (Causation):** Does that estimate *causally drive* advice? (patching, steering,
  and the novelty: is the *client's* risk direction distinct from the model's *own*?)
- **RQ3 (Pressure):** Under client pressure, is it *belief capture* (internal estimate
  moves) or *hypocritical compliance* (estimate holds, advice caves)?

## Pipeline
| Phase | What it does |
|-------|--------------|
| 0 | Synthetic profiles -> rubric -> tier; explicit/implicit/twin vignettes |
| 1 | Behavioral gate: does advice track true tier? (go/no-go) |
| 2 (RQ1) | Linear probe trained on explicit, tested on implicit |
| 3 (RQ2) | Causal mediation: patching + steering + self-vs-client decomposition |
| 4 (RQ3) | Pressure ladder + belief-capture vs hypocritical-compliance |
| 5 | SAE labeling (Gemma Scope) -- name the steering direction |
| 6 | Probe-based suitability monitor vs LLM-judge baseline |

## Models & data
- Primary: `google/gemma-2-9b-it` (bf16). Replication: `meta-llama/Llama-3.1-8B-Instruct`.
- SAEs: Gemma Scope (`google/gemma-scope-9b-it-res`), Llama Scope (`fnlp/Llama-Scope`).
- Data: fully synthetic, self-generated (~6k profiles -> ~8k vignettes). No scraping.

## Pre-registered thresholds
All success thresholds live in `config.yaml` and are meant to be **agreed by the team
before any results are seen** (no moving goalposts). Missing a threshold is not failure —
it changes which claim headlines.

## Layout
    notebooks/   Colab dev (prototype on 2B/T4, run on A100)
    data/        generated data (gitignored)
    results/     outputs, figures, metrics (gitignored)
    config.yaml  all experiment params + pre-registered thresholds
