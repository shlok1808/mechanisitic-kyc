# Mechanistic KYC Audit and Literature Review

**Review date:** September 6, 2026  
**Repository revision reviewed:** `c6c9a9f`  
**Scope:** Read-only review of the codebase, experimental design, synthetic-data pipeline, statistical analysis, reproducibility, and related research.

## Executive summary

The project has a promising research direction, but the current pipeline is not yet sufficient for strong scientific claims about how language models represent and use client risk information. The central issue is that risk willingness, financial capacity, and investment goals are combined into one synthetic risk tier. Consequently, a successful behavioral result or probe cannot show which financial concept the model learned or whether that concept caused the recommendation.

The strongest revised research question is:

> When a client's desired risk conflicts with the client's financial capacity, does pressure change the model's internal assessment, or does the model retain that assessment while allowing the recommendation to override it?

The next stage should redesign the labels and counterfactual profiles, define an explicit recommendation policy, freeze independent training, development, and test sets, establish behavioral validity, and only then run probing and causal intervention experiments.

## Work completed

The audit traced the implemented pipeline from profile generation through model analysis:

1. Synthetic client-profile and edge-case generation.
2. Weighted risk-tier construction.
3. Explicit, implicit, and paired vignette rendering.
4. Text quality-control checks.
5. Behavioral portfolio-choice evaluation.
6. Model-activation caching.
7. Layer- and position-wise linear probing.
8. Statistical utilities, configuration, tests, and reproducibility controls.

The automated suite contained 64 passing tests at the time of review. These tests provide evidence that covered utility functions behave as implemented. They do not validate the financial assumptions, labels, causal interpretation, or reported scientific conclusions.

The repository did not contain completed model outputs, result tables, plots, or historical run artifacts beyond placeholder files in `data/` and `results/`. Therefore, this review could audit the code and reconstruct parts of the synthetic-data design, but it could not independently verify previous model-level results.

## Main audit findings

### 1. The outcome combines different financial concepts

The current rubric creates a single tier from market-crash reaction, investment horizon, experience, income, goals, emergency savings, dependents, and age. Market-crash reaction and horizon receive the greatest weights. This makes the label easy to interpret as general risk tolerance even though it combines subjective willingness, objective loss-bearing capacity, and investment objectives.

The revised data model should retain separate targets:

- **Risk willingness:** comfort with volatility and losses.
- **Financial capacity:** liquidity, obligations, debt, income stability, reserves, and ability to bear loss.
- **Investment goals:** purpose, target, deadline, priority, and flexibility.
- **Advice outcome:** acceptable recommendation set, constraint violations, or request for clarification.

Goals should remain structured rather than being reduced to another low-to-high risk scale.

### 2. Existing contrasts do not isolate individual factors

Paired profiles currently share surface details but may differ across several substantive rubric variables. Such pairs can show that two broad profiles produce different behavior, but they cannot identify which factor caused the change.

Future counterfactual families should change one factor at a time while holding other facts constant. They should include high-willingness/low-capacity and low-willingness/high-capacity clients. The dataset should also contain cases where different factor values lead to the same acceptable recommendation and cases where the same factor value leads to different recommendations because another constraint binds. This prevents factor probes from merely identifying the expected answer.

### 3. Simple predictors expose a shortcut risk

Diagnostic reconstruction of the seeded synthetic profiles showed that simple text and field baselines could predict the generated tier:

| Diagnostic baseline | Approximate performance |
|---|---:|
| TF-IDF trained on explicit text and tested on implicit text | 0.590 accuracy |
| TF-IDF trained and tested on implicit text | 0.975 accuracy |
| Bag of words trained on explicit text and tested on implicit text | 0.597 accuracy |
| Bag of words trained and tested on implicit text | 0.983 accuracy |
| Text length only, implicit to implicit | 0.691 accuracy |
| Market-crash reaction alone | 0.830 accuracy |
| Market-crash reaction plus horizon | 0.922 accuracy |
| All rubric fields | 0.999 accuracy |
| Decoy fields | 0.497 accuracy |

These are diagnostics of the generated labels and templates, not empirical findings about a subject language model. They show that a high-performing model or probe could exploit a small part of the profile. Future evaluations should control keywords, length, template family, names, occupations, answer order, and paraphrasing style.

### 4. Recommendation correctness needs a defensible definition

Real investment advice rarely has one universally correct allocation. Reasonable recommendations depend on assumptions about products, returns, liquidity, taxes, and client preferences.

The controlled experiment should use a frozen product menu and an explicit task-specific policy. For every profile, it should define:

- A set or range of acceptable recommendations.
- Hard financial-constraint violations.
- Permitted trade-offs.
- Conditions requiring more information or refusal to recommend.

A separate natural-advice evaluation can use independent expert review. The controlled and natural tracks should not be combined into one accuracy claim.

### 5. Test-set information is used during model selection

The current probe workflow searches layers and token positions using implicit-test AUROC and then reports performance from the selected combination. Searching many combinations on the final evaluation data can inflate reported performance. Repeated splits at the chosen location do not repair that selection bias.

The corrected design should use:

- **Training:** fit probes and intervention directions.
- **Development:** choose layers, positions, ranks, thresholds, and steering strengths.
- **Untouched test:** evaluate the frozen procedure once.

All paraphrases, counterfactual siblings, and pressure variants of the same underlying client must remain in the same split. Uncertainty estimates should resample independent profile families rather than treating every related text as independent.

### 6. Decodability is not causal use

A linear probe can show that a label is recoverable from an activation. It does not show that the model uses that information to generate its advice. The future pipeline should triangulate:

- Natural one-factor textual counterfactuals.
- Activation patching between matched clients.
- Linear concept erasure.
- Steering across several strengths.
- Random and non-target intervention controls.
- Tests of unrelated capabilities and initially correct recommendations.

Each intervention should measure the target factor, non-target factors, and advice outcome. Selective effects that resemble natural changes to the same financial fact provide stronger evidence than a change in output alone.

### 7. Pressure must be measured at the correct position

In a causal decoder, an activation at the original end of the client profile cannot incorporate a pressure message appended later. Stability at that earlier token would be guaranteed by the architecture and would not demonstrate a stable internal assessment.

Pressure experiments should read activations after the follow-up pressure message and before the advice. They should compare:

- No follow-up.
- A neutral, length-matched follow-up.
- Evidence-free insistence, authority, flattery, or social pressure.
- Genuine new financial information that should change the assessment.

If advice changes while a validated capacity readout remains equivalent within a predeclared tolerance, the result is consistent with decision override. If both change, it is consistent with assessment shift. If the readout stops generalizing under pressure, the mechanism remains unresolved.

### 8. Reproducibility controls need strengthening

Activation caches are currently identified too narrowly, and some outputs can be overwritten by generic filenames. Every run should record:

- Dataset and prompt hashes.
- Code commit.
- Model and tokenizer revisions.
- Chat template and generation settings.
- Dependency versions and hardware precision.
- Seeds and split-family identifiers.
- Probe and intervention configurations.
- Raw outputs, exclusions, parsed results, and metrics.

Cache identifiers should include all inputs that can affect activations. Automated checks should cover split-family leakage, counterfactual fact preservation, option mapping, cache invalidation, and metric calculations.

## Literature findings

### Financial advice and risk profiling

- [Ross and Lo, *One Size Fits None: Heuristic Collapse in LLM Investment Advice*](https://arxiv.org/html/2604.23837v1) study 1,000 synthetic profiles and report that recommendations can be dominated by stated risk tolerance. This is the closest behavioral predecessor and means that shortcut dependence alone is not a sufficient novelty claim.
- [Takayanagi et al., *Are Generative AI Agents Effective Personalized Financial Advisors?*](https://arxiv.org/abs/2504.05862) separate preference elicitation from advice and find that conflicting preferences remain difficult. This supports measuring client understanding and policy application independently.
- [Chawla et al., *Evaluating AI for Finance*](https://aclanthology.org/2025.emnlp-industry.189/) report demographic inconsistencies in model risk assessment. Their work motivates matched identity controls, although their combined risk score should not be treated as validated ground truth for this project.
- [Zhao et al., *The Price of Agreement*](https://arxiv.org/html/2604.24668v1) show that misleading personalized context can affect financial agent behavior. Their tasks use objective document answers, so portfolio advice requires additional controls for legitimate preference updates.
- The [CFA Institute's investment risk-profiling review](https://www.cfainstitute.org/sites/default/files/-/media/documents/survey/investment-risk-profiling.pdf) distinguishes subjective risk willingness from objective capacity and cautions against collapsing them through simple weighting.
- [ESMA guidance concerning insistent clients](https://www.esma.europa.eu/publications-data/questions-answers/1761) supports holding objective capacity constraints fixed when testing whether client pressure produces unsuitable advice.

### Mechanistic methods

- [Chen et al., *TalkTuner*](https://arxiv.org/abs/2406.07882) demonstrate probing and steering internal user attributes. Internal user-profile decoding is therefore methodological precedent rather than the project's main novelty.
- [Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*](https://aclanthology.org/2024.acl-long.828/) provide a simple contrastive steering method that can serve as an intervention baseline.
- [Belrose et al., *LEACE*](https://arxiv.org/abs/2306.03819) provide a principled linear concept-erasure method. Successful linear erasure still does not prove that every form of the information has been removed.
- [Geiger et al., *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*](https://proceedings.mlr.press/v236/geiger24a.html) show how interchange interventions can test relationships between high-level causal variables and distributed representations.
- [Makelov et al., *An Interpretability Illusion for Subspace Activation Patching*](https://arxiv.org/abs/2311.17030) show that successful interventions can produce misleading explanations of a model's natural computation. This motivates using several independent intervention and control methods.

## Revised direction

The most defensible contribution is a causal study of financial-constraint representation and enforcement under pressure. The study should ask whether willingness, capacity, and goals are independently recoverable; whether each factor has a selective causal effect when it is decision-relevant; and whether pressure changes the assessment or its downstream influence.

The recommended experiment order is:

1. Define the factor ontology and controlled recommendation policy.
2. Generate feasible one-factor counterfactual families.
3. Freeze grouped training, development, and test splits.
4. Establish behavioral validity and shortcut controls.
5. Train and validate separate factor probes.
6. Run patching, erasure, and steering with specificity controls.
7. Run pressure and genuine-update experiments at post-pressure positions.
8. Compare mechanistic interventions with prompt and rule-based safeguards.
9. Replicate the central result on a second model.

## Conclusion

The project is ready for a redesigned pilot, but the existing combined label and evaluation procedure should not be used for a full causal study. The revised direction preserves the project's original aim while providing clearer financial constructs, stronger controls, a valid final test, and a contribution that is better differentiated from existing financial-advice and user-representation research.

No production experiment files were modified as part of the audit. This report documents the findings and the proposed next phase.
