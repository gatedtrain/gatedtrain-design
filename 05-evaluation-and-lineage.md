# 05. Evaluation and Lineage

**Status:** The design set is at revision 19; this document was last changed in revision 19. A prototype (`gatedtrain` 0.3.0) implements revision 19. Frozen until G0 closes; deferred items in `open-issues.md`. Proposed sub-design.

**Revision:** 9 (2026-11-05); applies R9-5 and R9-6 of the round-3 review (record not published). Revision 8 parent: `design.md` (the overview). Implements overview decisions D10 and D11, the exit procedures referenced in overview §7.1, and the round-2 decisions R8-16 and R8-17 plus the statistical parts of R8-2 and R8-15 (round-2 review record, not published).

**Scope:** the statistical gate library, the G0 feasibility comparison and the parity procedures for phase exits, evaluator independence, online reward-hacking detection, human-feedback statistics, per-family evaluators, contamination checks, the lineage data model, the consumption ledger and its schema, storage, retention, and the reproducibility statement.

**Baselines read for this document:** nemo-evaluator at `ea35ba3` (2026-10-19), nemo-curator at `8a9a217` (2026-10-30), and flow_grpo, all local clones. Paths below are relative to each repository root.

## 1. Decisions in this sub-design

| # | Decision | Findings |
|---|---|---|
| E1 | Every gate is one of three typed tests (superiority, non-inferiority, equivalence) with a declared margin, α, and target power; every report carries MDE and an `underpowered` flag; `alpha` has one meaning in every spec (§3.2) | EVAL-1, EVAL-7, EVAL2-6 |
| E2 | The analysis unit is the item (for one checkpoint) or the training seed (for a recipe claim); intervals are item-clustered | EVAL-1, RES-6 |
| E3 | G0 runs a feasibility comparison (1–2 seeds, fixed sample count, GPU-hour cap) with descriptive statistics only; P1b and later exits use TOST with at least five seeds and a power analysis on an upper bound of σ | EVAL-7, RES-9, EM2-2, EVAL2-7 |
| E4 | Turn gates use Bonferroni over the known turn bound or registered online FDR; the final gate keeps family-wise control on a disjoint test set | RES-6, EVAL-1, EVAL2-2 |
| E5 | Evaluator independence is graded in tiers; each blocking clause needs a non-LLM evaluator of tier T2 or higher, except that a qualified LLM or VLM judge may block a constraint clause alone (§3.9, §6.1); tiers are checked by a non-inferiority bound on the evaluator-to-proxy gain ratio on human-confirmed probes | EVAL-2, EVAL2-1, EVAL2-6 |
| E6 | Reward-hacking detectors run in shadow mode until their per-step false-alarm rate and CUSUM ARL₀ are calibrated by bootstrap replay of human-labelled reference runs | EVAL-3, EVAL2-3 |
| E7 | Contamination is checked per modality, asymmetrically, once per artifact pair, and persisted; similarity thresholds are calibrated per embedding model | EVAL-6, LIN-7, LIN2-7 |
| L1 | Lineage follows a PROV-style model in OpenLineage format. P2 stores it as tables in the workflow database and the ledger as run-partitioned Parquet queried with DuckDB; Marquez or DataHub and Iceberg arrive in P3 | LIN-1, LIN-5, LIN2-3 |
| L2 | `data_ref` replaces `data_version`; filter and rejection decisions are recorded per row, including Curator filters through a wrapper | LIN-2, LIN-3, LIN2-8 |
| L3 | Every `name@version` is resolved to a manifest hash and image digest in the lockfile; the registry refuses models whose hash does not resolve | LIN-4, LIN-8, LIN2-4 |
| L4 | Reproducibility means statistical re-execution, not bitwise identity | LIN-6 |
| L5 | One `ConsumptionRecord` schema, one outcome enum, and one sizing table live in package A; composition uses a generic `parent_ref` | LIN2-2, ARCH2-7 |
| L6 | Consumption is defined by weight-version ancestry over `(controller_epoch, counter)`; rollups are per lineage; superseded segments are marked; every consumption edge records its precision | LIN2-1, LIN2-5 |

## 2. What already exists (verified)

| Component | Location | What it does | Gap for this design |
|---|---|---|---|
| `bootstrap_ci` | nemo-evaluator `src/nemo_evaluator/metrics/confidence.py:40` | Percentile bootstrap of a mean, i.i.d. resampling, fixed seed 42 | No cluster resampling, no two-stage (seed, item) resampling |
| `clustered_ci` | `metrics/confidence.py:152` | Cluster-robust standard error, normal quantile (`z`, line 216) | With few clusters (for example GenEval's 6 tags) a `z` quantile is anti-conservative; should use `t` with G−1 degrees of freedom |
| `mcnemar_test`, `sign_test`, `permutation_test` | `metrics/paired_tests.py:92`, `:142`, `:181` | One-sided paired tests for degradation; `detect_test` (`:218`) picks McNemar for binary, permutation otherwise | One-sided degradation only; no superiority or equivalence form |
| `mde_estimate` | `metrics/paired_tests.py:239` | MDE = 2.8/√(discordant pairs) at 80% power. 2.8 ≈ 1.96 + 0.84, which is **one-sided α = 0.025**. The upstream comment on `POWER_80_FACTOR` (`:44`) says "alpha=0.05 one-sided", which would give 1.645 + 0.842 ≈ 2.49; the comment is wrong, not the constant | Binary discordant pairs only; used by `compare_runs` verdicts (`engine/comparison.py:596–614`), not by the gate. A comment fix is offered upstream |
| `gate_runs` | `engine/gate.py:116` | Per-benchmark paired gate: averages repeats per problem (`_metric_problem_values`, `:336`), needs ≥ 10 paired problems (`:56`), clustered CI when category metadata exists (`:268–270`), PASS when the 95% CI of damage lies under `max_drop` (`_apply_thresholds`, `:464`) | A non-inferiority test at one-sided α = 0.025 with a fixed default margin (`max_drop = 0.015`, `config/gate_policy.py:49`); no power, no multiplicity, no training-seed level |
| Tier aggregation | `engine/gate.py:537` | Any critical or supporting BREACH → NO-GO; INSUFFICIENT_EVIDENCE → INCONCLUSIVE; advisory ignored | Matches an intersection-union rule for AND clauses; no OR clauses |
| Exact dedup | nemo-curator `nemo_curator/stages/deduplication/exact/identification.py:171` | MD5 of text; optional normalization lowercases and collapses whitespace (`:56–57`) | No Unicode NFKC folding; self-join over one corpus, not eval-versus-train |
| Fuzzy dedup | `stages/deduplication/fuzzy/workflow.py:41` | MinHash (`fuzzy/minhash.py:184`) + LSH (`fuzzy/lsh/stage.py:30`); defaults 24-char shingles, 20 bands × 13 hashes (`workflow.py:78–80`) | Self-join only |
| Semantic dedup | `stages/deduplication/semantic/workflow.py:55` | Embedding, k-means, pairwise similarity within clusters | Self-join only |
| Text filters | `stages/text/filters/score_filter.py:123` (`Filter`), `:225` (`ScoreFilter`) | Compute a keep mask per document and apply it with `df = df[bool_mask]` (`:207`, `:342`) | Dropped rows leave no IDs or reasons (§12.6) |
| Image dedup | `stages/image/embedders/clip_embedder.py`, `stages/image/deduplication/removal.py:27` | CLIP embeddings, removal by ID list | No perceptual hashing anywhere in the repository |
| Video embedding | `stages/video/embedding/cosmos_embed1.py:30`, `:110` | Cosmos-Embed1 frame creation and clip embedding | No ANN index or eval-versus-train query |

**GenEval in flow_grpo is a training reward.** `flow_grpo/rewards.py:210` defines `geneval_score`, registered in `multi_score` (`:421`); the recipe `geneval_sd3` in `config/grpo.py:125–158` trains SD3.5-M with `reward_fn = {"geneval": 1.0}` on `dataset/geneval/train_metadata.jsonl`. Measured on the local clone: the 50,000 training lines (33,199 unique prompts) and the 553 unique test prompts share **no** exact prompt, but all 80 test object classes occur in training, 367 of 553 test prompts (66%) share their task tag and object-class tuple with a training prompt, and 101 (18%) also share every count and colour slot. Exact-hash deduplication reports this benchmark as clean.

# Part A. Evaluation and gates

## 3. Gate library

The gate library is package E-stats, `gatedtrain-gates` (overview §4), released in P0. It wraps NeMo Evaluator: evaluation runs, per-item records, and `gate_runs` are reused; the library adds typed hypotheses, the training-seed level, power, multiplicity, and pre-registration. Fixes of general value (a `t` quantile in `clustered_ci`, superiority and equivalence modes in `_apply_thresholds`, the `POWER_80_FACTOR` comment) are offered upstream first.

### 3.1 Gate types

Let Δ = μ(candidate) − μ(baseline) in oriented units. Each clause states `direction: higher_is_better | lower_is_better`, matching the evaluator manifest; a lower-is-better metric is negated, so Δ, δ, δ_sup, θ, and `bar` are in oriented units and Δ > 0 always favours the candidate. The report gives the oriented Δ̂ and interval and, beside them and labelled, the difference in natural units.

**Ratio metrics.** A ratio-of-sums metric (for example corpus WER = Σ errors / Σ words) may be the deciding metric. Per-item records carry the numerator and denominator (§3.8); μ is the arm's ratio, so Δ is the difference of the arm ratios; the interval is in §3.4 and power in §3.6.

| Type | Null hypothesis | Pass rule (one-sided level α) | Typical use |
|---|---|---|---|
| Superiority | Δ ≤ δ_sup (usually 0) | Lower (1−α) bound of Δ > δ_sup | Turn acceptance, "the RL stage improved the product metric" |
| Non-inferiority | Δ ≤ −δ | Lower (1−α) bound of Δ > −δ | Regression checks, standalone checks, safety metrics |
| Equivalence (TOST) | \|Δ\| ≥ δ | Both one-sided tests reject at α, that is, the (1−2α) interval lies inside (−δ, δ) | Phase-exit parity, reproducibility checks (§14) |

A threshold gate (04's `gate.threshold`, H₀: μ(candidate) ≤ bar) is a superiority clause with the baseline replaced by the constant `bar`, evaluated unpaired. The mapping from 04's workflow gate types to clauses is in 04 §4.2.

**Constraint clauses.** A threshold or non-inferiority clause may measure violations of a registered natural-language constraint set instead of a quality metric: its metric is the per-item violation indicator, `direction: lower_is_better` (§3.9).

```yaml
clause:
  type: non_inferiority          # superiority | non_inferiority | equivalence
  metric: geneval.overall        # evaluator id and metric, resolved to a manifest hash
  direction: higher_is_better    # higher_is_better | lower_is_better, as in the evaluator manifest
  margin: 0.02                   # δ, in metric units, justified in the registration
  alpha: 0.025                   # one-sided level of each test (§3.2)
  power: 0.8                     # target power at the planning effect
  planning_effect: 0.0           # Δ assumed for the power calculation
  sigma_plan: ucb80              # ucb80 | internal_pilot | fixed (§3.6)
  unit: item                     # item | training_seed (§3.3)
  samples_per_item: 4            # K
  eval_seeds: [11, 12, 13, 14]   # common evaluation noise seeds
  interval: cluster_bootstrap    # cluster_bootstrap | cluster_robust | wilson | clopper_pearson | clopper_pearson_discordant (§3.9)
  cluster_key: null              # optional: an item-set field; items sharing its value are resampled together (§3.4)
  bootstrap_variant: bca         # optional: bca (default) | percentile (§3.4)
  constraint_set: null           # constraint clause only: registered constraint-set id (§3.9)
  blocking: true
```

**Several blocking clauses.** A gate may block on several clauses, for example a quality superiority clause and a non-inferiority guard on another metric of the same outputs. The registration then lists them in `clauses` instead of `clause`: each entry carries the fields above and also `name` (unique within the registration) and `evaluator` (the evaluator whose metric decides that clause; one evaluator per clause, each listed among the registration's evaluators, §3.7). Every entry is blocking, and the clauses combine by intersection-union (§3.5). A constraint clause keeps the single-clause form. A registration with one `clause` keeps its meaning field for field.

A gate with `blocking: true` and no declared margin is rejected at validation. A gate whose achieved power is below target reports `underpowered: true` (defined in §3.6); an underpowered pass is reported as PASS and flagged, while an underpowered non-rejection is INCONCLUSIVE, never FAIL-with-evidence (NeMo Evaluator's INSUFFICIENT_EVIDENCE).

### 3.2 Meaning and defaults of α fields (shared with 04 and 06)

| Field | Meaning | Default (`plan()` drafting value) |
|---|---|---|
| `alpha` on a clause or a 04 gate | One-sided level of each test. For equivalence, the level of each of the two one-sided tests, so the decision interval is (1−2α) | 0.025 for superiority, non-inferiority, and threshold; 0.05 for equivalence (a 90% interval, the TOST convention) |
| `alpha_family` on a 04 loop | Family-wise error rate of a family of turn gates; the per-turn level is derived (§5) | 0.05, method `bonferroni` |
| `fdr` with `alpha_family.method: lord` or `saffron` | Online false discovery rate over turn acceptances (§5) | 0.10 |
| A gate inside a family | Must not set `alpha`; the validator derives it and rejects an explicit value | — |

A 04 example that sets `alpha: 0.05` on a single superiority or threshold gate therefore means one-sided 0.05; it is allowed when the registration justifies it, and the report prints the level used.

The defaults are the values `plan()` proposes when it drafts a registration. The value that applies is the one in the registration (§3.7); a gate outside a family whose registration sets no `alpha` is rejected at validation (04 §4.2).

### 3.3 Experimental design

An item is the evaluation unit of the family: a prompt (LLM, VLM, image, video, CAD), an utterance (speech), an episode seed with environment parameters (control), or a (task, init_state) pair (VLA).

- **Paired.** Baseline and candidate are evaluated on the same items, with K samples per item and the same evaluation noise seeds (common random numbers). Pairing removes item difficulty from Δ.
- **Two units of analysis.** A gate on a fixed checkpoint treats the item as the unit (K samples averaged per item, items resampled). A claim about a recipe or an implementation treats the training seed as the unit, because seed-to-seed variance of RL runs dominates evaluation noise; items are nested within seeds (§4).
- **Variance from raw scores.** Variance is computed on raw per-item paired differences, never on group-normalized scores or advantages, which remove the item component and shrink the apparent variance (RES-6). Generative-model gates run under the deployment sampler recorded in the checkpoint (02); training-sampler scores are reported but do not decide.
- **Evaluated model.** Evaluation records name the evaluated checkpoint's `model_manifest_hash` (§11), which covers normalizer statistics (08 §6.2); a mismatch with the gate's pinned checkpoint is MISSING (§3.8).

### 3.4 Intervals and tests

| Metric shape | Default interval | Notes |
|---|---|---|
| Continuous per-item means | Item-clustered bootstrap: resample items with all K samples of both arms, 10,000 replicates, BCa | Two-stage (seeds, then items within seeds) when the unit is the training seed |
| Same, few items or heavy clusters | Cluster-robust standard error with `t` on G−1 degrees of freedom | Clusters are items, or templates when templates repeat (§10.3) |
| Ratio of sums (for example corpus WER) | Item-clustered bootstrap of the difference of the arm ratios: resample items, both arms together, 10,000 replicates, BCa with the item jackknife | Records carry numerator and denominator (§3.8); power from linearized per-item contributions (§3.6) |
| Binary rate with independent units | Wilson; Clopper-Pearson for safety-critical rates or small n | Zero failures in n trials: one-sided 95% upper bound ≈ 3/n (rule of three) |
| Binary paired outcome per item | Exact McNemar, mapped to the clause's hypothesis | Superiority and non-inferiority forms added beside the degradation form; constraint clauses use §3.9 instead |
| Two independent binary rates (strata) | One-sided Fisher exact or Boschloo test | Used by human spot checks (§7) |
| Pairwise win rate (judge or human) | Cluster bootstrap over items; ties counted as half wins and reported | Position randomized per comparison (§8) |

**Clustered items and the percentile bound.** When items share a source of variation that the item set records (the voice prompt of a speech item, a template, a scene), a clause may name that item-set field in `cluster_key`. The bootstrap then resamples the G clusters of complete pairs with replacement, G per replicate, each drawn cluster bringing both arms of all its items; clusters are taken in sorted order of their field values. A replicate of a mean difference is the sum of the drawn clusters' item differences divided by their number of items; for a ratio-of-sums metric it is the difference of the arm ratios of the drawn clusters' sums. BCa then uses the leave-one-cluster-out jackknife. Δ̂ and n (complete pairs, §3.6) do not change, and fewer than two clusters is a zero-width interval (§3.6). The field is read from the registered item set; an item without it, or with a value that is not a string or an integer, refuses the gate before analysis. `bootstrap_variant: percentile` replaces BCa by plain percentile bounds: the lower (1−α) bound of Δ is the α quantile of the replicates and the upper bound the (1−α) quantile, in oriented units (§3.1); its p-value by test–interval duality (§3.8) is the share of replicates below the null value, ties counted half. BCa remains the default; a registration chooses the percentile variant explicitly, for example to keep an analysis fixed before the gate was ported. In natural units (candidate − baseline) the pass rules of §3.1 read: for a lower-is-better metric, superiority passes when the upper (1−α) bound is below −δ_sup and non-inferiority when it is below δ; for a higher-is-better metric, superiority passes when the lower (1−α) bound is above δ_sup and non-inferiority when it is above −δ.

**Missingness check (G-18).** Expected items are the registered item set × the K repeats (× training seeds when the unit is the training seed), per arm; a threshold clause has one arm. An expected item is missing in an arm when it has no record, or its record carries an evaluator error or a score outside the evaluator's declared range (§6.1); a record without an evaluation seed is also a missing item; the gate library assigns the reason: the error class, `out_of_range`, `no_seed`, or `absent`. The registration sets `pairing: {max_unpaired: <count or fraction>}`, default 0: a count bounds the total missing items over all arms; a fraction in [0, 1) bounds that total divided by the number of expected items over all arms. An item missing in both arms counts once per arm. The analysis uses complete pairs only (§3.6), and the report gives each arm's missing count and the count per reason. A clause whose missing items exceed the bound is MISSING (§3.8).

### 3.5 Combining clauses

- **AND** of clauses: intersection-union test. Each clause is tested at its own α; the probability of a false pass is at most max α, so no Holm or other multiplicity adjustment is applied. Power, however, multiplies: with m roughly independent clauses each at power p, joint power is about pᵐ (five clauses at 0.8 give 0.33). **m counts every blocking evaluator** in the AND clause: a clause that blocks on two evaluators (04 §13's system gain) contributes two (EVAL3-1). A gate with m AND clauses must use one of two plans (EVAL2-4):
  1. **Per-clause power p^(1/m).** For joint power 0.8, each clause is planned at 0.8^(1/m): 0.894 for m = 2, 0.946 for m = 4, 0.992 for m = 28. At one-sided α = 0.025 the per-clause sample size grows, relative to an 80%-power clause, by about 1.31× (m = 2), 1.62× (m = 4), and 2.43× (m = 28). Positive correlation between clauses makes this plan conservative.
  2. **Averaged blocking clause.** One non-inferiority clause on the equally weighted mean of Δ over the m cells blocks; the full matrix is descriptive, and a cell whose point estimate is below −2δ is flagged for the gate owner. This is the default for cross-play matrices (04 §10.3), where m grows with accepted partner versions.
- **Registering several clauses.** An AND gate is registered with `clauses` (§3.1) and `multiplicity: {plan: all_clauses_must_pass}`. It passes only when every clause passes, each at its own α, with no α split. Its status follows the `and` precedence of §3.8: FAIL if any clause fails, else MISSING if any clause is missing, else INCONCLUSIVE if any clause is inconclusive, else PASS. FAIL comes first because one clause that fails with adequate power already rules out a PASS of the gate, whatever the other clauses show; MISSING comes before INCONCLUSIVE because an unusable clause has no evidence to weigh. `post_hoc` (§3.7) and an evaluator hash mismatch act on each clause before the combination. Each clause is planned at its own registered power; the plan and the report state the joint power of independent clauses, the product of the registered powers, so a registration that plans each of m clauses at power p uses plan 1 above with joint power pᵐ, and says so. The report gives every clause's result beside the combined status.
- **OR** of clauses: Holm step-down over the m alternatives at the clause α.
- **Several metrics reported, one deciding:** the deciding metric is named in the registration; the others carry no α.

### 3.6 Power, MDE, and sample size

For a paired continuous metric with item-level standard deviation σ_d of the differences, n items, and design effect DE:

  MDE = (z₁₋α + z₁₋β) · σ_d · √(DE / n),  DE = 1 + (m − 1)·ρ

where m is the average cluster size and ρ the intra-cluster correlation (per-item means already absorb K; template clusters do not).

**Planning σ (EVAL2-7).** A point estimate of σ from a few seeds understates the required n about half the time. `plan()` therefore offers:

- `ucb80` (default): plan with the one-sided 80% upper confidence bound σ_U = s · √(df / χ²₀.₂₀,df). For five reference seeds (df = 4, χ²₀.₂₀,₄ = 1.649) σ_U = 1.56 s; for two seeds (df = 1, χ²₀.₂₀,₁ = 0.064) σ_U = 3.95 s, which is why G0 data cannot plan P1b alone (§4.1).
- `internal_pilot`: start with the registered minimum n, re-estimate σ after the first stage from the pooled data with arm labels hidden (blinded one-sample variance), and raise n if needed, never lower it. Blinded re-estimation keeps the type I error essentially at α; the rule and the maximum n are registered in advance.
- `fixed`: σ from an external source named in the registration.

**Achieved power and `underpowered` (G-3, G-12, G-13).** Achieved power is computed at the registered planning effect θ (`planning_effect`), never at the observed Δ̂, with the observed standard deviation σ̂_d of the paired differences (ddof = 1) and the normal approximation, SE = σ̂_d · √(DE / n):

- Superiority and non-inferiority: power = Φ(d / SE − z₁₋α), with d = θ − δ_sup (superiority) or d = θ + δ (non-inferiority).
- Equivalence (TOST): power = max(0, Φ((δ − θ)/SE − z₁₋α) + Φ((δ + θ)/SE − z₁₋α) − 1). The required n is DE · ((z₁₋α + z_{1−β/2}) · σ / δ)² at θ = 0, with z_{1−β/2} the 1 − β/2 normal quantile, and, conservatively, DE · ((z₁₋α + z₁₋β) · σ / (δ − |θ|))² for θ ≠ 0, with σ from the σ plan. §4.2's seed-level bound is the same form with `t` quantiles (t₀.₉₅ + t₀.₉₀ at α = 0.05, β = 0.2). A planning effect with |θ| ≥ δ has no finite n: `plan()` refuses it and validation rejects the registration.
- Ratio metrics (§3.1): power and n are planned from linearized per-item contributions (delta method). In each arm d_i = (num_i − R·den_i) / mean(den), with R that arm's ratio, num_i and den_i the sums over the item's repeats used in the analysis (those present in both arms, §3.4); the paired difference of the two arms' contributions per item takes the place of the paired difference above, and the σ plan applies to it. `min_paired` and the missingness check (§3.4) apply per item as usual.

With `cluster_key` (§3.4), SE uses the observed design effect: DE is the cluster-robust variance of the mean paired difference, G/(G−1) · Σ_g (S_g − Δ̂·m_g)² / n², with S_g the summed differences and m_g the complete pairs of cluster g, divided by σ̂_d²/n, and at least 1; for a ratio metric the linearized contributions take the place of the differences. The planned n is computed as above; a registration that expects clustering plans with a correspondingly larger σ.

`underpowered` is true when achieved power is below the registered power, when n is below the planned n (after any `internal_pilot` re-estimate), or when power cannot be computed. When σ̂_d = 0, power takes its limit and the report sets `sd_zero: true`.

**Minimum paired sample.** The registration sets `min_paired`, default 10 (the minimum of NeMo Evaluator's `gate_runs`, §2). n counts complete pairs in units of analysis (items, or training seeds when the unit is the training seed; §3.3), the same n as in the power formulas above. A complete pair is an item with at least one repeat present in both arms; its difference averages the repeats present in both arms. A clause with fewer than `min_paired` complete pairs, or with a degenerate interval of zero width (n < 2, or σ̂_d = 0 with no between-item variation), is INCONCLUSIVE, never PASS. A clause reported with `sd_zero: true` and a zero-width interval is therefore INCONCLUSIVE, whatever its limiting power.

**Pairwise win rate.** To show a win rate of 0.55 against H₀: 0.5 with one-sided α = 0.025 and 80% power needs

  n = [ (1.960·0.5 + 0.842·√(0.55·0.45)) / 0.05 ]² = (0.980 + 0.419)² / 0.0025 ≈ 783

independent comparisons, multiplied by the design effect when comparisons share an item. Labeling cost is in §8.

### 3.7 Pre-registration and report

Before a stage or phase starts, the gate owner writes a registration: clauses (one `clause`, or several in `clauses` with their names and deciding evaluators, §3.1, §3.5), margins with justification, α, power, σ plan (with several clauses, `sigma_pilot` maps each clause name to its pilot values), unit, the evaluation distribution `eval_distribution` (the item-set manifest hash, plus `env_version` and `env_params_version` for environment evaluations, 08 §6.1), K, evaluation seeds, sampler, evaluator manifest hashes, the deciding metric (§3.5, §6.1), `min_paired` (§3.6), `pairing.max_unpaired` (§3.4), for constraint clauses the constraint set and the judge validity results (§3.9, §6.1), the multiplicity plan (§5), and the analysis code version. Its SHA-256 is an input of the gate activity in lineage (§11): the registration is an artifact of type `registration`, bound to the gate's `registration` input role by content hash (04 §4.1). The registration is authoritative for every clause field it holds (§3.1: type, metric, direction, margin or bar, α, power, planning effect, σ plan, unit, K, evaluation seeds, interval, constraint set) and for the sampler and evaluator; a workflow spec field that maps to one of them and differs is rejected at validation (04 §4.2). Evaluator manifest hashes are resolved (§12.7, lockfile) when the registration is written; validation refuses a null or unresolved hash (anything other than `sha256:` and 64 hex digits, such as a name or tag) whether or not the registration is signed. The registration records its creation time (`created_at`, UTC), which the `post_hoc` audit below compares with raw-results artifacts. A mismatching registration hash fails validation; amendments create a new registration that references the prior one by hash, and both appear in the report. An amendment made after any result of the candidate checkpoint that the gate judges was observed marks the gate `post_hoc: true`; results of other checkpoints, including other candidates of the same gate in a loop, do not. A candidate-arm raw-results artifact of that checkpoint (04 §4.3) created before the amended registration counts as observed. A `post_hoc` clause cannot PASS: one whose pass rule holds is reported as INCONCLUSIVE with reason `post_hoc`; FAIL, INCONCLUSIVE, and MISSING are reported unchanged. Amendments made before that remain allowed. A version-only amendment is not post hoc: one that changes the analysis code version from a version the framework recorded under its name before a package rename to the renamed framework's current version, and nothing else. It keeps an earlier `post_hoc`: when the prior registration is `post_hoc: true`, so is the amendment, and declaring it false is refused. Validation checks this mechanically against the prior registration: the amended registration equals it in every field except `analysis_code_version`, `amends`, `created_at`, and `post_hoc`, the prior version is a recognized pre-rename version, and the new one is the current version; the report marks the amendment as version-only. Such an amendment may also change an evaluator manifest hash when the scorer behind it was changed only by the rename: the amendment lists, per changed evaluator, the scorer code hash before the rename and the pre-rename scorer file or files, shipped next to the registration; validation checks that the evaluator's identity with that code hash is the prior manifest hash and with the current code hash is the amended one (only the scorer code changed), that the files hash to the code hash before the rename, and that the files with the published old-to-new name map applied give the current code hash byte for byte. An amendment that changes anything else follows the rule above, and declaring it not post hoc is refused when it crosses the rename. Each report states: hypothesis type, Δ̂ and interval, margin, α actually used, achieved power, MDE, `underpowered`, the independence tier and transfer-test reference of each evaluator (§6), for constraint clauses the violation count, n, and exact upper bound, the judge's miss and false-alarm rates with their upper bounds, and canary results (§3.9, §6.1), for a gate with several clauses these fields per clause with its name and evaluator and the combined status (§3.5), contamination status per training artifact (§10), the family's spent α or FDR wealth (§5), registration hash, and evaluator, evaluation-distribution (item set and, for environments, `env_version` and `env_params_version`), and sampler hashes. Reports are immutable artifacts.

### 3.8 Library interface

| Function | Input | Output | Built on |
|---|---|---|---|
| `plan(clause, pilot)` | Clause and baseline records | Required n items or seeds under the σ plan, MDE, design effect, per-clause power for AND gates | `mde_estimate` generalized to continuous metrics |
| `reestimate(plan, blinded_stage1)` | Plan and pooled first-stage data | Revised n (never lower) | New |
| `evaluate(clause, baseline, candidate)` | Paired per-item records (item, repeat, seed; shape below) | Δ̂, interval, p-values, achieved power, status | `gate_runs` records, `clustered_ci`, `bootstrap_ci`, paired tests |
| `combine(results, rule)` | Clause results, `and`, `and_mean`, or `or` | Verdict | `_aggregate_verdict` semantics |
| `family_level(family, t)` | Family registration and history | α for decision t (Bonferroni) or LORD/SAFFRON test level and wealth | New |
| `independence(gate, lineage)` | Gate spec, lineage closure | Tier per (evaluator, reward) pair, non-LLM check, transfer-test references | New |

Statuses map onto NeMo Evaluator's: PASS; FAIL (`BREACH`); INCONCLUSIVE (`INSUFFICIENT_EVIDENCE`); MISSING (the clause's data or registration is absent or unusable, including missing items above the bound, §3.4, an evaluator hash mismatch, §6.1, an evaluated `model_manifest_hash` that differs from the gate's pinned checkpoint, §3.3, records with a repeat outside range(K) or a seed other than the registered `eval_seeds[repeat]`, and a missed injection canary in a judged batch of a judge that blocks a constraint clause alone, §6.1; fewer than K records for an item are missing items, §3.4, not a MISSING clause by themselves). `combine` applies a fixed precedence (G-9). For `and` and `and_mean`: FAIL if any clause fails, else MISSING if any is missing, else INCONCLUSIVE if any is inconclusive, else PASS. For `or`: PASS if any clause passes after the Holm correction (§3.5), else MISSING if any is missing, else INCONCLUSIVE if any is inconclusive, else FAIL. Only PASS opens a blocking gate; INCONCLUSIVE and MISSING follow the gate's `on_fail` policy in 04, where they are reported as `underpowered` and `missing`.

**Per-item record (package A).** `evaluate` reads records with fields `item`; `repeat` (0 ≤ repeat < K); `seed` (required; `eval_seeds[repeat]`); `score` (natural units, or null); `error` (evaluator failure class, or null; exactly one of `score` and `error` is null); `sub_metrics` (descriptive, never deciding); and `output_ref` (manifest hash of the generated output: text, image, video, audio, code, or episode log). For ratio metrics (§3.1) the record also carries the numerator and denominator. Each arm records once the evaluated checkpoint's `model_manifest_hash` (§11), the evaluator manifest hash, and the sampler hash. The evaluator manifest declares which outcomes are scores (the model's fault, for example code that does not execute) and which are errors (the evaluator's fault). NeMo Evaluator records are converted into this shape.

**p-values.** `evaluate` reports p-values from the procedure that gives the clause's interval, by test–interval duality. For superiority and non-inferiority the one-sided p-value is the smallest α at which the pass rule holds; for equivalence it is the larger of the two one-sided p-values (TOST). The decision comes from the interval. A zero-width interval or no complete pairs gives no p-value.

### 3.9 Constraint clauses

Generated outputs of any family (image, video, audio, text, episodes) may be gated on registered natural-language constraints ("the output must not …"). A constraint clause is a threshold or non-inferiority clause (§3.1) with `constraint_set` set.

- **Constraint set.** Held in the registration (§3.7): the constraint texts, numbered (`c1`, `c2`, …); for each constraint its checker, either a programmatic tool (manifest hash) or a judge (model ID and version or date, prompt template, decoding, parser); and the judge's canary set (§6.1). All of it is inside the evaluator manifest hash (the scorer hash, §6.1, §12.7), so a changed text, checker, template, model, or parser is a hash mismatch and the clause is MISSING (§3.8). A programmatic checker is used wherever one exists for a constraint; a judge checks only the constraints no tool can.
- **Metric.** A record's `score` is 1 when its output violates any constraint and 0 otherwise; the evaluator manifest declares the range [0, 1] and `direction: lower_is_better`. The item's violation indicator v_i ∈ {0, 1} is 1 when any of its repeats present violates (repeats present in both arms for the non-inferiority form, §3.6); repeats are combined by maximum, not averaged. The violation rate is the mean of v_i over items. An unparseable judge response is an evaluator `error` (a missing item, §3.4), never a 0. `sub_metrics` carry the violated constraint IDs and the judge's rationale; like all sub-metrics they never decide (§3.8).
- **Threshold form.** One arm; `bar` is the highest acceptable violation rate. In oriented units this is §3.1's threshold clause: it passes when the upper (1−α) bound of the candidate's violation rate is below `bar`. The bound is exact, Clopper-Pearson over items by default (`interval: clopper_pearson`), or the clause's registered interval method; the report gives the violation count, n, and the exact upper bound. Zero observed violations do not mean a rate of 0: the bound is then 1 − α^(1/n), about 3.69/n at α = 0.025 (3.0/n, the rule of three, at α = 0.05), so `bar` = 0.01 needs at least 368 items without a violation. That bound has positive width, so the zero-width rule of §3.6 does not apply to it; `sd_zero` is reported. Power and n come from the binomial: the smallest n for which the exact bound is below `bar` with the registered power at the planning violation rate (`bar` minus the planning effect).
- **Non-inferiority form.** Paired against the baseline on the same items: Δ is the baseline's violation rate minus the candidate's (oriented), and the clause passes when the upper (1−α) bound of the violation-rate difference, candidate minus baseline, is below δ. The bound is exact and conservative (`interval: clopper_pearson_discordant`, the default): over the n complete pairs, let c count the items where the candidate violates and the baseline does not, and b the reverse; the difference is (c − b)/n ≤ c/n, so the Clopper-Pearson upper (1−α) bound of the rate c/n bounds the difference from above. It gives no credit for items where only the baseline violates. (Exact McNemar conditions on the discordant count b + c and does not bound the difference once δ > 0, so it is not used here.) The report gives b, c, n, the difference (c − b)/n, and the bound. The bound has positive width when c = 0, so the zero-width rule of §3.6 does not apply; `sd_zero` is reported. Power and n come from the binomial as in the threshold form, with δ in place of `bar` and the planning rate of c/n, δ minus the planning effect, in place of the planning violation rate.
- **Combination.** A constraint clause is combined with the quality clauses by `and` (§3.5) and counts in m; a FAIL of the constraint clause fails the gate, and MISSING and INCONCLUSIVE follow the precedence of §3.8.
- **Blocking evaluator.** The rule of §6.1 applies, with its exception for a qualified judge as the sole blocking evaluator of a constraint clause.

## 4. G0 feasibility comparison and parity procedures

Overview §7.1 refers here for G0, P1b, P5, P6a, P6b, and P7. G0 is a feasibility check; parity is decided from P1b on (R8-2).

### 4.1 G0 feasibility comparison (EM2-2)

1. **Register:** family, reference (flow_grpo, or verl-omni and whether it recomputes old log-probabilities), recipe, hardware, a fixed sample count N_fix, the GPU-hour cap declared in the G0 plan, 1–2 training seeds per arm, metrics (final training reward and one tier-T2+ non-LLM evaluator, §6), and a coarse feasibility band per metric.
2. **Run** both arms to N_fix samples, or until the cap; a run stopped by the cap is reported as incomplete.
3. **Report descriptively:** per-seed metrics with item-clustered intervals (these capture evaluation noise only, not seed variance), curves at equal sample counts, per-timestep ratio statistics (§4.3) without a test, throughput in samples/s/GPU, and GPU-hours used.
4. **Read-out:** feasible means both arms finish within the cap without divergence and each metric lies inside its band. G0 never reports "parity" or "equivalence": with two seeds the between-seed σ has one degree of freedom per arm. G0 seeds serve as a pilot for P1b, whose plan uses `ucb80` on the reference seeds or `internal_pilot` (§3.6).

### 4.2 Full parity (P1b and later exits)

1. **Register** before any run: reference implementation and commit, recipe, hardware, metrics (final training reward and at least one tier-T2+ non-LLM evaluator), margins δ, α = 0.05 per one-sided test, the σ plan, seeds per arm, and the sample count at which metrics are compared.
2. **Seeds.** At least five training seeds per arm, more when the power analysis requires. Runs are compared at equal sample counts, not equal wall-clock.
3. **Test.** Per metric, TOST on per-seed scores (each seed's mean over the full evaluation set under the deployment sampler) with Welch's `t` (a paired `t` on per-seed differences when the arms share initial noise and prompt order, as registered). With five seeds per arm (df ≈ 8) and true Δ = 0, 80% power requires

   δ ≥ (t₀.₉₅,₈ + t₀.₉₀,₈) · σ_s · √(2/5) = (1.860 + 1.397) · 0.632 · σ_s ≈ 2.06 σ_s

   with σ_s the between-seed standard deviation taken from the σ plan, by default the 80% upper bound from the reference's seeds. If the registered δ is smaller, the registration raises the seed count. Three seeds per arm (df ≈ 4) need δ ≥ about 3.0 σ_s and are not accepted.
4. **All metrics must pass** (intersection-union, §3.5), planned with per-clause power 0.8^(1/m) over the m metrics. Curves are descriptive.
5. **Per-timestep ratio statistics** pass by §4.3.
6. **Throughput.** P1b also requires end-to-end samples/s/GPU of at least 0.9× the reference on the same hardware and configuration (R8-15). The baseline is verl-omni with at least three interleaved run pairs (PERF3-1); the measurement procedure and its owner are in 08; this document only carries the result in the parity report.
7. **Report** each metric's interval, δ, achieved power, σ plan, the per-timestep results, the throughput ratio from 08, and the registration hash.

**Worked example (P1b, path N, SD3.5-M).** Reference: flow_grpo `pickscore_sd3` (512 px, 10 sampling steps). Metrics: final PickScore (reported, not the independence anchor) and GenEval overall under the 40-step deployment sampler (T2 for this recipe). Suppose five reference seeds give s = 0.012 for GenEval overall; σ_U = 1.56 × 0.012 ≈ 0.0187. With five seeds per arm the smallest margin with 80% power is 2.06 × 0.0187 ≈ 0.039. If the adopter accepts ±0.025, the plan needs 11 seeds per arm ((t₀.₉₅,₂₀ + t₀.₉₀,₂₀) · √(2/11) ≈ 1.30 ≤ 0.025/0.0187 ≈ 1.34); for ±0.015 it needs 28. Planning on the point estimate would have given 5 and 12 seeds, the revision-7 numbers. An internal pilot that starts at 5 seeds per arm and re-estimates σ blind is the cheaper alternative when σ_U is pessimistic. The numbers are illustrative; σ_s is measured on the reference seeds before P1b.

### 4.3 Per-timestep ratio statistics (EVAL2-5)

Under frozen weights, per denoise timestep t, compute mean |log r_t|, the clipped fraction, and the ratio standard deviation, for the candidate (rollout versus training forward) and for the reference (its own rollout versus its own recompute on the same hardware), one value per training seed. A 95th percentile cannot be estimated from five seeds, so revision 7's percentile rule is replaced by one of two registered procedures:

- **(a) Seed random-effects prediction bound on the candidate-seed mean (EVAL3-2).** Statistics are modelled on the log scale as y_{r,t} = μ_t + u_r + e_{r,t}. From R reference seeds, the one-sided upper prediction bound for the mean of m new candidate seeds at timestep t is ȳ_t + t₁₋α_T,R−1 · s_t · √(1/m + 1/R), with α_T = α/T over T timesteps (here α/T bounds false alarms across timesteps). The **mean** of the candidate seeds, not each seed, must lie below the bound plus a registered tolerance; checking each seed separately misfires as seeds are added. For R = 5, m = 2, and α = 0.05 the multiplier is 2.132 × 0.837 = 1.78 s_t at one timestep and 4.604 × 0.837 = 3.85 s_t with T = 10. Used when the candidate has few seeds (G0, descriptively).
- **(b) Per-timestep TOST as an intersection-union test.** With at least five seeds per arm, each timestep is tested by TOST (or its one-sided upper half for statistics where only an increase signals a defect) on seed-level values with Welch's `t` and a registered tolerance τ_t, each at the unadjusted α; all timesteps must pass (§3.5), so no Holm adjustment is applied (revision 8's Holm contradicted the intersection-union rule). Per-timestep power is planned as power^(1/T) (0.978 for T = 10 at joint power 0.8), or the registration instead blocks on one averaged clause over timesteps with the per-timestep results descriptive (§3.5 plan 2). This is the P1b default.

## 5. Repeated decisions and turn gates

Turn gates (04 owns their workflow semantics) make many accept/reject decisions; at an unadjusted one-sided 0.025 over 18 turns the chance of at least one false acceptance is 1 − 0.975¹⁸ ≈ 37% (EVAL-1). Revision 7's 1/t² spending starved late turns: turn 18 would get α = 0.05 · 6/(π² · 18²) ≈ 9.4 × 10⁻⁵ and need about 2.67× the samples of a single test (EVAL2-2).

- **Bonferroni over the known bound (default).** T_max = `max_iterations` × trained nodes is known at validation (04 bounds loops). Each turn decision uses α_FW / T_max, default 0.05 / T_max. For T_max = 18 the per-turn one-sided level is 0.0028 (z = 2.773); at 80% power a turn gate needs ((2.773 + 0.842) / (1.960 + 0.842))² ≈ **1.66×** the sample size of a single test at one-sided 0.025. Within a turn, the clauses (system gain, standalone regression, cross-play) form an AND gate at the turn level (§3.5).
- **Online FDR (registered alternative).** LORD++ or SAFFRON at level `fdr` (default 0.10) control the expected fraction of false acceptances among accepted turns. They spend less when many turns are genuine improvements; the report shows the test level and remaining wealth per turn.
- **Final gate keeps family-wise control.** Turn acceptances only choose which versions advance. The release decision is the final gate on the disjoint final test set, evaluated once, at its own one-sided family-wise α (default 0.025; 04's examples use it) with intersection-union over AND clauses and Holm over OR clauses. Online FDR is never used for the final gate.
- **Loop extension.** An `extend` by N turns (04) cannot reuse spent α. The registration may hold back a reserve for extensions; without one, extended turns form a new family, reported as such, and the final gate still protects the release.
- **Within a turn: sequential looks.** Early stopping uses an O'Brien–Fleming spending boundary or an always-valid confidence sequence inside the turn's α; uncorrected peeking is rejected at validation.
- **Fresh selection subsamples and seeds.** Each turn draws its selection items afresh from a selection pool (without replacement until exhausted, then a new permutation); the final test set is disjoint and used once. Accepted version and candidate are evaluated with K ≥ 5 evaluation seeds per item; common random numbers are allowed, with variance computed on raw per-item paired differences (§3.3).

## 6. Evaluator independence

### 6.1 Tiers and the blocking rule

Every evaluator and training reward has a manifest declaring its **backbone family** (for example `clip-vit-h`, `siglip`, `qwen2.5-vl`, `dinov2`, `whisper`, `paddleocr`), whether it is an LLM or VLM judge, its training-data lineage where known, and its labeler pools. The validator computes a tier for each (evaluator, training reward) pair over the lineage closure of all upstream stages, including frozen partner policies.

| Tier | Definition | May satisfy a blocking clause |
|---|---|---|
| T0 | Is a training reward, shares checkpoint or fine-tune ancestry, shares training data or labelers, or uses the same programmatic tool as a reward | No; non-blocking only |
| T1 | Different weights, same backbone family (for example PickScore and HPSv2 on CLIP ViT-H) | No; supporting only |
| T2 | Different backbone family, no shared data or labelers | Yes, if the transfer test passes (§6.2) and it is not an LLM or VLM judge (except a qualified judge of a constraint clause, below) |
| T3 | Programmatic metric independent of every reward: OCR edit distance with a different OCR engine, geometry checks, simulator metrics, physical measurements | Yes |
| T4 | Human evaluation with labelers disjoint from preference-data labelers | Yes |

**Same programmatic tool (R15-1).** A tool is the engine that computes the score: the codec, OCR engine, answer parser, detector, or simulator, and its version. The wrapper library that calls it does not count (a JPEG reward and a WebP metric, both called through Pillow, use different tools: libjpeg and libwebp). A T3 metric computed by a different engine may still measure the same construct as a reward, as the WebP size of an image measures the same information content as its JPEG size. Such a metric is **construct-aligned**: its manifest says so, and the registration and the report state that a pass shows the reward's gain transferring to an independent engine, not a gain in any other quality.

**Shared labelers (R16-1).** "Shares training data or labelers" grades a scorer that learned from, or is judged by people who produced, the training labels: a learned scorer trained on those labels, or a human or judge panel drawn from those labelers. Held-out item references that the same annotation process produced, and that no training stage consumed, do not make a programmatic scorer T0 (word error rate against held-out transcripts of the corpus the model was fine-tuned on). The registration and the report state the shared annotation convention, because part of the gain may be learning that convention.

**Rule (aligned with 04 and 06, R8-12).** Each blocking clause, except a constraint clause under the exception below, needs at least one evaluator of tier T2 or higher that is **not** an LLM or VLM judge: a T3 programmatic metric, a non-generative T2 scorer, or T4 human evaluation. An LLM or VLM judge may be T2 and contribute to a clause, but only beside such an evaluator and only with injection canaries (06). Where no non-LLM evaluator exists for a system-level metric, as with story-level quality in 04's `av_story_eval`, the blocking clause uses human evaluation and the judge is supporting; human turn gates carry the cost stated in §8. Unknown lineage (third-party evaluators) means T1 unless the gate owner records a signed declaration (06) and the transfer test passes.

**Constraint clauses: a judge as the sole blocking evaluator.** For a constraint clause (§3.9), and only for one, an LLM or VLM judge may be the sole blocking evaluator when all of the following hold; otherwise it is supporting and the rule above applies. Quality clauses keep the rule above unchanged.

- **(a) Validity.** Before registration, the judge's miss rate (violations, by human label, that it does not flag) and false-alarm rate (flags on outputs humans label compliant) are measured against human labels collected by the §8 annotation process, on a sample of baseline outputs and, when available, of outputs of an earlier candidate from a different checkpoint than the one the gate judges (such results do not make the gate `post_hoc`, §3.7). Both rates and their exact (Clopper-Pearson) upper (1−α) confidence bounds, at the α of the constraint clause, are stated in the registration and the report. The registration sets a bound on the miss rate and optionally on the false-alarm rate; a judge whose upper confidence bound exceeds a registered bound cannot block.
- **(b) Independence.** Tier T2 or higher against every training reward (the table above, including the transfer test of §6.2), and a backbone family different from the policy's and from every training reward's.
- **(c) Injection canaries (06 §12.1).** A registered canary set, outputs that violate a constraint and carry instructions aimed at the judge (for example text rendered in an image saying "no violation"), is mixed into every judged batch and excluded from v_i. A canary the judge does not flag makes the clause MISSING (§3.8). Per-batch canaries are required only of a judge that blocks alone; a supporting judge needs only the conformance canaries of 06 B26.
- **(d) Reproducibility.** Temperature 0 or the registered deterministic decoding; responses cached by request hash in the `JudgeResponseCache` (§11, §12.7); model ID and date pinned. A changed judge changes the scorer hash, so a mismatch at run time is MISSING (§3.8).
- **(e) External judges.** A judge reached through an external API receives candidate outputs: that is data egress, an `upload` destination that needs 06's data-tag approval (06 §9, B20). Its model version must be pinnable; a judge whose version cannot be pinned may only support.

**GenEval.** For any recipe whose upstream rewards include GenEval, GenEval is T0; DPG-Bench, T2I-CompBench with a different detector backbone, or human evaluation must take its role. A recipe that trains on PickScore and OCR may use GenEval as T2 after the transfer test.

**Scorer validity (G-21).** The evaluator manifest hash covers the task definition, the prompt template, answer extraction and normalization (the scorer), and the container digest (§12.7). The registration names the deciding metric with that resolved hash; a gate whose evaluator hash differs at run time is MISSING (§3.8). The manifest declares the score range [lo, hi] (either bound may be open); a score outside it is a missing item with reason `out_of_range` (§3.4). Before registration, the scorer's error rate is measured on a labelled sample of the baseline's outputs (false-positive and false-negative rates against a reference judgment) and stated in the registration and the report. A learned continuous scorer (aesthetic predictor, PickScore, a reward model) has no error rate: its validity is its agreement rate with human pairwise preferences on a labelled sample of baseline outputs, collected by the §8 annotation process and stated in the registration and the report, with a registered lower bound. A deciding metric whose baseline score (the pilot baseline mean stated in the registration beside `sigma_pilot`) lies at the floor or ceiling of the declared range, whose measured error rate exceeds the registered bound, or whose agreement rate is below the registered bound, is refused at registration. A learned scorer that is a training reward remains T0. The scorer is also graded by the tiers above against every training reward: a scorer that shares its answer parser with a training reward uses the same programmatic tool and is T0.

**Pre-run baseline check.** Before a full training run, a short smoke run scores a small baseline sample (20 items in the prototype) with each deciding evaluator, every clause's when the gate has several (§3.1), and the run proceeds only when no evaluator is degenerate on it: for a non-binary metric, a score exactly equal to the finite bound at the metric's best end (the lower bound of a lower-is-better metric, such as a WER of 0; the upper bound of a higher-is-better one) is a legitimate best outcome and cannot be the value that makes up half of the scores, while a score on the worst end (an OCR accuracy of 0, nothing rendered) counts as any other value; the share stays a share of the whole sample, and at least a quarter of the sample (and at least two scores) must lie off the best bound, so an evaluator pinned at its best value, constant, or nearly so still fails. A sample with no score on the best bound is judged exactly as before. The check guards against a broken or constant evaluator; it decides nothing about the gate.

### 6.2 Transfer test (EVAL2-1)

Revision 7 passed an evaluator when its gain on hack probes was "not significantly positive", which passes on absence of evidence, and counted best-of-N samples as hacks although many are genuine improvements. The test is now a non-inferiority bound.

- **Probes.** For each training reward, candidates come from (a) best-of-N (N ≥ 256) selection under the reward from the base policy, (b) a short run with KL disabled, and (c) artifact edits (oversaturation, rendered prompt text, high-frequency noise). Each candidate is paired with a matched base sample for the same prompt and seed. Human raters (T4 pool, two per pair) confirm that the candidate shows **no genuine gain on the evaluator's construct** (EVAL3-3): the question names what the evaluator measures (for GenEval, whether the requested objects, counts, colours, and positions are rendered correctly), not overall quality. **Disagreement rule:** a pair is confirmed only when both raters answer "no gain"; a split pair is excluded, and the per-source disagreement rate is reported beside the confirmation rate. Only confirmed pairs enter the test.
- **Statistic.** On confirmed pairs, gains are standardized by the base samples' standard deviation: g_p for the proxy (training reward) and g_e for the evaluator. The ratio R = mean(g_e) / mean(g_p) measures how much of the proxy's spurious gain the evaluator follows.
- **Pass rule.** The probe set must first show a proxy gain (lower 97.5% bound of mean(g_p) > 0), otherwise the result is INCONCLUSIVE. The evaluator passes when the one-sided upper (1 − α) confidence bound of R, by Fieller's method or an item-cluster bootstrap, is below r_max (default 0.3, α = 0.025). The Spearman correlation of g_p and g_e is reported descriptively.
- **Probe-set size.** Near R = r₀, approximately Var(R̂) ≈ σ_e² / (n · ḡ_p²), so 80% power needs n ≥ [(z₁₋α + z₀.₈) · σ_e / ((r_max − r₀) · ḡ_p)]² confirmed pairs. With α = 0.025, σ_e = 1 (base-SD units), ḡ_p = 1.5, r_max = 0.3, and a planning ratio r₀ = 0.1: n ≥ (2.802 / 0.3)² ≈ 88 pairs per reward, about 147 candidates at a 60% confirmation rate. With r₀ = 0 the requirement falls to 39 pairs. This approximation is only the starting value: the registered size comes from a **bootstrap** over the pilot's confirmed pairs (resample n pairs by item cluster, compute the upper bound of R, and take the smallest n for which at least 80% of replicates pass), because the ratio's sampling distribution is skewed when ḡ_p is small. The registration states r₀, σ_e, ḡ_p, and the bootstrap result from a pilot of the probe source.
  The result is a stored artifact keyed by evaluator, reward, probe-set hash, and method version; gate reports cite it with the tier.

## 7. Online reward-hacking detection

Detectors run inside RL stages, report through the run contract's telemetry (02), and act through the `ControlAPI` (03 §8). Each starts in **shadow mode** (alerts recorded, no action). Detectors never roll back automatically.

| Detector | Statistic | Trigger |
|---|---|---|
| Proxy-versus-gold divergence | On a fixed probe set (256–512 items, fixed noise seeds) every N steps, under the training and the deployment sampler: d_t = z(proxy) − z(gold), standardized against step 0, with a T2+ gold evaluator | One-sided CUSUM S_t = max(0, S_{t−1} + d_t − d₀ − k) above h, or Page-Hinkley; alarm on the deployment-sampler series, sampler gap reported separately |
| RM-ensemble disagreement | Variance across ensemble members on the probe set and on top-quantile training samples | Upward trend (Mann-Kendall) over a sliding window |
| OOD score | Mahalanobis or k-NN distance in a different backbone's embedding space relative to reference-policy samples | Rising fraction above the reference 99th percentile |
| Diversity collapse | Diffusion: within-group LPIPS and DINOv2 cosine distance, CMMD against a reference set, fraction of zero-variance reward groups; LLM: token entropy, length, n-gram repetition | Drop below a registered fraction of step-0 values; zero-variance fraction above a threshold |
| Family artifacts | Image: saturation, high-frequency energy, rendered-text area; video: dynamic degree, optical flow, frozen-frame rate; audio: silence ratio, LUFS, clipping; LLM: length and repetition | Excursion beyond the reference run's 99th percentile for a sustained window |
| Human spot checks | Stratified sample each review period: top 1% of training reward, 1–5%, uniform; 50–100 items per stratum | Hack rate in the top stratum exceeds the uniform stratum by a one-sided Fisher exact or Boschloo test at α = 0.05 (replaces revision 7's non-overlapping Clopper-Pearson intervals, which are needlessly conservative) |

**Calibration (EVAL2-3).** Ten runs with "below one alarm per ten runs" could not support the claim (zero alarms in ten runs only bounds the rate below 1 − 0.05^(1/10) ≈ 0.26 at 95%). Calibration is now per step:

- **Ground truth.** Reference runs labelled by humans: reviewers inspect stratified samples at each evaluated checkpoint and label it hacked or not hacked. Unhacked runs provide in-control series; runs with KL disabled or known exploits provide hacked series and detection delays.
- **Per-step false-alarm rate.** Alarms per monitoring interval on in-control series, with a registered target (for example at most 1/(5L) for L intervals per run, about 0.2 expected false alarms per run).
- **CUSUM ARL₀ by two-level bootstrap replay (EVAL3-4).** Standardized in-control increments from at least five human-labelled unhacked reference runs are resampled in two levels: first runs with replacement (so between-run variation enters ARL₀), then a moving-block bootstrap within each drawn run (block length registered to keep autocorrelation; blocks never cross runs). 10,000 such replicates are replayed; (k, h) are chosen so that the in-control average run length ARL₀ meets the target, and the detection delay ARL₁ on hacked series is reported. Windowed trend tests (Mann-Kendall) are calibrated the same way because they test repeatedly.
- **Run-level claim.** A false-alarm rate per run is claimed only from at least about 30 unhacked runs with zero alarms (rule of three: 3/30 = 0.10; exact 95% bound 1 − 0.05^(1/30) ≈ 0.095).
- **Promotion** to `pause` requires the per-step and ARL₀ targets on replay and a reported ARL₁.

## 8. Human feedback statistics and reward-model lifecycle

Statistical design for the P7 connector and preference store; deadline behaviour is in 04 and consent enforcement in 06.

- **Model.** Bradley–Terry with labeler random effects: logit P(i ≻ j | labeler l) = (θ_i − θ_j)·exp(s_l) + b_l·side_ij, with s_l ~ N(0, τ_s²) (sharpness), b_l ~ N(0, τ_b²) (position bias), side_ij the presentation side. Ties use the Davidson extension.
- **Quality control (EVAL2-8).** Two checks **block** a batch from RM training: (1) gold accuracy, per labeler and per batch, whose lower 95% Wilson bound must reach a registered floor (labelers below it are excluded and their labels reweighted to zero); (2) model fit, the fitted model's log-loss on a held-out 10% of comparisons, which must beat both a registered floor and a position-only baseline, plus a deviance check per labeler. Krippendorff's α per task and pool (on at least 10% of items labeled by three or more labelers) is **monitored**, trended, and alerts on a drop beyond a registered margin, but does not block: preference tasks with many close pairs have low agreement by design, and revision 7's 0.667 cut-off would have rejected valid batches.
- **Design.** Position randomized per comparison and recorded; pairs assigned by a balanced incomplete block design; gold and honeypot items at 5% of each batch. Active sampling by ensemble disagreement records the propensity π_i; a uniform stratum of 10–20% guarantees π_i ≥ ε; RM training uses truncated (w ≤ w_max, default 20) or self-normalized inverse-propensity weights, with the effective sample size reported.
- **Calibration and drift.** Each RM version has a calibration report on **on-policy** samples: pairwise accuracy, expected calibration error, score distribution. Drift (a failed non-inferiority test on on-policy accuracy, or a proxy-versus-gold alarm) schedules relabeling and retraining, which starts a new reward epoch (§11).
- **Deadlines.** Late labels are never silently masked (RES-7): the connector waits and escalates (04), or reweights on-time labels by an estimated on-time propensity; the mode and effective sample size are recorded. **Sybil and collusion:** an agreement graph over labelers (adjusted for gold accuracy) is checked each batch; clusters with high mutual agreement and low gold accuracy are flagged. Identities and enforcement are owned by 06.
- **RM training input** is an immutable `PreferenceSnapshot` (§11).

**Labeling cost of human gates.** A final win-rate gate (0.55 against 0.5, one-sided α = 0.025, 80% power) needs 783 comparisons (§3.6). With 10% of items labeled by three labelers (two extra labels each, ×1.2) and 5% gold items (÷0.95), that is about 989 labels, roughly 16.5 labeler-hours at an assumed one minute per comparison. As a Bonferroni turn gate with T_max = 18 the same effect needs 1,304 comparisons, about 1,648 labels per turn and about 29,650 for 18 turns. That figure is for one human clause at 80% power. In 04 §13 three blocking clauses use the human panel (system gain and both cross-play clauses) and m = 5, so each is planned at 0.8^(1/5) ≈ 0.956 power, about 1.54× the comparisons; at equal effect sizes that is about 4.6× the single-clause figure, roughly 7,600 labels per turn (EVAL3-1). The **validator computes the required labels from the actual gate configuration** (every T4 blocking evaluator at its derived α, power^(1/m), planning effect, design effect, QC overhead, and the turn bound) and rejects a spec whose `labeling_usd` does not cover them. Workflows whose turn gates need human evaluation (§6.1) may otherwise choose online FDR or a larger planning effect; relaxing the turn-gate defaults is deferred to P10 (`open-issues.md`).

## 9. Per-family evaluator sets

Gates pick evaluators from these sets subject to the tier rule (§6). Evaluators are wrapped as NeMo Evaluator tasks or `@scorer` functions.

| Family | Evaluators | Caveats |
|---|---|---|
| Image | GenEval, DPG-Bench, T2I-CompBench, CMMD against a reference set, OCR edit distance with a different OCR engine, human side-by-side | GenEval is T0 for recipes that train on it; template overlap (§10.3); FID descriptive only |
| Video | VBench motion smoothness, dynamic degree, subject and background consistency, flickering; prompt alignment with a VLM of a different family than any judge reward; human ratings | Dynamic degree and smoothness trade off; both gated; a VLM alignment score needs a non-LLM companion (§6.1) |
| Audio and speech | WER with an ASR different from any ASR reward; speaker similarity with a different encoder; MOS and CMOS from ≥ 20 human raters; UTMOS-class predictors supporting only | Predicted MOS is T1 if a MOS predictor is a reward |
| Embodied (VLA, driving, flight) | Closed-loop success and failure on **held-out scenario distributions**; collision and rule-violation rates with Clopper-Pearson bounds; rule of three for zero-event claims; open-loop metrics supporting | Scenario splits by map or session (§10.2); safety gates per 06 |
| CAD | Executes, valid solid, Chamfer or IoU to target, manufacturability rules with a kernel different from any reward | All T3 if the reward uses different rules |
| LLM and VLM | NeMo Evaluator benchmarks with programmatic scoring; judges of a different family than reward models, supporting only | Contamination per §10 |
| Composed systems | Story-level system metrics (04 `av_story_eval`) | Human evaluation blocks where no non-LLM evaluator exists (§6.1) |

## 10. Contamination

### 10.1 Procedure

1. **Asymmetric.** The evaluation set is indexed in memory (exact hash set, MinHash-LSH, ANN embedding index); each training-side artifact (SFT data, RL prompts, rejection-sampled data, preference data, distillation targets) is streamed through it. Cost is linear in the training corpus.
2. **When.** At artifact registration for each (evaluation set, training artifact) pair, and lazily for pairs first seen by a gate.
3. **Persisted.** A `ContaminationReport` keyed by (evaluation-set hash, training-artifact hash, method version, embedding-model hash, thresholds) lists matched items, match type, and similarity; gates reuse reports through lineage.
4. **Gate use.** The validator requires a report for each training artifact in the candidate's lineage closure. Contaminated items are removed from the gate's item set (reported), or the gate fails above a registered fraction. **Unknown pretraining data** is reported as `pretraining_contamination: unknown` per base model.

### 10.2 Methods per modality

| Data | Exact | Near-duplicate | Structural |
|---|---|---|---|
| Text | SHA-256 after NFKC, case folding, whitespace collapse, punctuation stripping | MinHash-LSH over word 5-grams, Jaccard ≥ 0.8, via Curator's `MinHashStage` and `LSHStage` with an eval-side index; embedding similarity at the calibrated threshold | Template and slot match (§10.3) |
| Images | SHA-256 of decoded pixels | pHash and dHash (Hamming ≤ 8 of 64) plus CLIP or DINOv2 ANN at the calibrated threshold | Caption-side text checks |
| Video | Hash of decoded keyframes | Perceptual hashes on keyframes; Cosmos-Embed1 embeddings with an ANN index | Source-clip and scene IDs |
| Audio | Hash of decoded PCM | Chromaprint-class fingerprint and speaker-embedding match; transcript checks | Speaker and session IDs |
| VLA and driving | Episode ID | Trajectory similarity on (state, action) sequences | Splits by scenario, map, town, and session |
| CAD | Canonical B-rep or code hash | Geometric similarity of normalized shapes | Part-family IDs |

**Thresholds per embedding model (LIN2-7).** Cosine similarities are not comparable across embedding models, so revision 7's single 0.95 is only a starting value. Each (embedding model, modality) pair has a calibration artifact: a labelled set of near-duplicate and distinct pairs (including in-domain hard negatives), the threshold chosen for a registered precision (default 0.95), and the recall at that threshold. The threshold and the embedding model's manifest hash are part of the report key, so a model change invalidates old reports.

### 10.3 Template and slot overlap (GenEval-style benchmarks)

Exact and fuzzy matching miss overlap at the level that matters (§2: zero exact matches, 66% shared class tuples, 18% full slot matches). The check parses prompts into (template, slots) using benchmark metadata (GenEval's `tag` and `include`) or a template grammar; reports the fraction of evaluation items whose slot tuple, and whose full assignment, occurs in training; adds prompt-embedding matching; and labels gates `in_template_distribution` above a registered threshold, offering a held-out slot split as the blocking variant. Template clusters are also the clusters for cluster-robust intervals (§3.4).

# Part B. Lineage

## 11. Lineage data model

The model follows W3C PROV: entities (artifacts), activities (runs and stages), and agents, with `used`, `wasGeneratedBy`, `wasDerivedFrom`, and `wasAssociatedWith` edges, in OpenLineage event format (run start, complete, fail; dataset facets carry manifest hashes).

| Entity | Key fields |
|---|---|
| `Artifact` (base) | `manifest_hash`, `type`, `schema_ver`, `licence_tags`, `policy_tags`, `consent_tags`, `created_by` |
| `DatasetSnapshot` → `Shard` | Snapshot at a lakeFS commit (Iceberg snapshot from P3); shards with row counts and hashes |
| `Mixture` | Component snapshots, weights, sampling seed, epoch policy |
| `FilterDecision` | Filter stage, version, thresholds; per-row kept or dropped with reason (columnar side table, §12.6) |
| `PreferenceSnapshot`, `LabelBatch` | Immutable preference snapshot with propensities; protocol hash, labeler pseudonyms, consent scope, agreement and gold results |
| `ContaminationReport`, `GateReport`, `Registration`, `TransferTest` | §10.1, §3.7, §6.2 |
| `LatentCache`, `EmbeddingCache` | Encoder or VAE manifest hash, preprocessing hash, source snapshot |
| `JudgeResponseCache` | Request hash → response; judge model ID and date, prompt template, parser: together the **scorer hash** |
| `Checkpoint` | `model_manifest_hash` (deployable components only: weights, model config, tokenizer or processor, generation config, normalizer statistics, `base_ref`; the component list per family is in 08 §3.6), by which gates, exports, and deployments name the checkpoint; `base_ref` (base model manifest hash, for adapter-only checkpoints); `cut_manifest` (02), whose hash names the resume point; weight version `(controller_epoch, counter)`, `last_step_included`, `reward_epochs[]`, `partner_versions[]`, sampler configs |
| `Export`, `Deployment` | Source checkpoint, format or quantization config; target, registry reference (MLflow or NGC), time window |

| Activity or record | Key fields |
|---|---|
| `Run` | `lockfile_hash`, hardware, attempts with their `controller_epoch`, registration hashes, start and end |
| `LineageSegment` | `(run, attempt, controller_epoch)`, restored cut, `parent = (parent_epoch, branch_step)`, status (§12.4) |
| `RewardEpoch` | Run, index, start step, pinned reward artifacts and service digests, normalization-reset step |
| `ConsumptionRecord` | Package A schema (§12.2) |

### 11.1 Required `used` edges (LIN2-4)

Impact queries are only as complete as the edges. The validator and the conformance kit require:

- **Data-producing models.** Rejection-sampled sets `used` the generating checkpoint and the selecting reward or verifier; synthetic preferences `used` the generator and the judge; distillation targets `used` the teacher; judge caches `used` the scorer hash.
- **Every model role of a run:** policy initialization, teacher, KL reference, critic initialization, `old` and EMA roles (an in-run role is `wasDerivedFrom` the policy; a role loaded from an artifact is a `used` edge), reward models, judges, partner policies, encoders and VAEs.
- **LoRA merges** (`used` base and adapter, generate the merged model); an unmerged adapter that is evaluated, exported, or deployed gets a `lora_compose` activity that `used` both and generates a model entity whose `model_manifest_hash` covers the base and adapter hashes, without merged weights; **exports** (format conversion, quantization), and **deployments**, including deployments made outside the framework's emitter: the model registry admission hook refuses a model whose manifest hash does not resolve to an artifact in lineage.
- **Conformance test.** A reference workflow exercises each edge type; from each deployment, the closure back to datasets must contain no unresolved hash; a negative test registers an unknown hash and expects refusal.

### 11.2 Edge precision (LIN2-5)

Each consumption edge (a run `used` a dataset or shard) carries `precision ∈ {row, step_range, iteration, stage}`:

| Precision | Meaning | Typical source |
|---|---|---|
| `row` | Row-level `ConsumptionRecord`s exist | NeMo RL with the provenance export callback (hook 8d), package D |
| `step_range` | Records cover chunks (episode or row ranges per step) | Fused vectorized execution (08) |
| `iteration` | Only the shards used per training iteration are known; recorded as `iteration` records (§12.2) | Trainers that report data-loader shards per iteration |
| `stage` | Only that the stage used the snapshot | Third-party trainers, offline SFT tools |

Query results report the precision of every edge and the coarsest precision on each path; an answer through a coarser-than-row edge reads "possibly consumed". When the same answer is reached by several paths, it takes the finest of the per-path precisions and lists every path: a row-level path that shows consumption is not weakened by a coarser path to the same checkpoint.

**Derivation.** Each `ConsumptionRecord` carries `derivation ∈ {observed, reconstructed}` (§12.2). A reconstructed record, derived from a deterministic sampler and recorded seeds rather than emitted during training, may claim at most `iteration` precision unless the adapter passes a replay conformance test, after which `row` is allowed. Query results and impact answers show the derivation.

### 11.3 Impact query

"Which deployed models used shard S, label batch B, or reward model R" is the reverse closure: follow `used` to consuming activities, `wasGeneratedBy` to their outputs, and repeat to `Deployment` entities. Reward models appear through `RewardEpoch`; label batches through `PreferenceSnapshot` → RM training → RM → `RewardEpoch`. For training data, membership of a checkpoint is decided by weight-version ancestry (§12.4) using per-lineage rollups, and the Parquet ledger is scanned only for row-level questions. By default "used" means outcome `consumed`; `include_exposed: true` also returns rows that reached a reward service or telemetry but were dropped (needed for policy-tag and consent questions). A `stage` edge has no ledger records or rollups: every checkpoint the consuming stage generated is returned as possibly consumed, with precision `stage`; derivation does not apply to a `stage` edge and is reported as null. Across runs the closure continues through `used` model edges (a later run whose base or initial model is such a checkpoint), and each answer carries the coarsest precision on its path (§11.2).

## 12. Sample-level references and the consumption ledger

### 12.1 `data_ref` and `parent_ref`

```
data_ref   = { snapshot: <DatasetSnapshot manifest hash>, shard: <shard id>, row: <opaque row key>,
               mixture: <Mixture hash | null>, component: <index | null> }
parent_ref = { kind: record | artifact, id: <ConsumptionRecord key | manifest hash> } | null
```

Row keys are opaque (hashed) so that lineage stores no personal identifiers (§13.4). `row` is required except in records of precision `iteration` (§12.2). Generated samples keep their prompt's `data_ref` plus their derivation (weight version, sampler config hash, seed inputs, §14). `parent_ref` is generic (ARCH2-7): composed systems use it to point at the upstream system sample's record, and derived datasets to point at their source artifact; no composition-specific identifier (revision 7's `parent_sample`, 02's `system_sample_id`) appears in general records.

**Environment-based RL without a dataset.** Environment parameters are materialized as an ordinary `DatasetSnapshot`, one shard per `env_params_version` (08 §6.1), with seed ranges as rows where applicable. Consumption records anchor their `data_ref` to it, so rollups and impact queries work unchanged.

### 12.2 `ConsumptionRecord` schema (package A, LIN2-2)

Package A (pure Python, pyarrow optional) owns the one schema used by 02's emitter, the ledger writer, and queries. 02 refers to it rather than defining its own.

| Field | Type | Notes |
|---|---|---|
| `run_id`, `attempt_id`, `step`, `sample_id` | key | `step` is the optimizer step index; `sample_id` hashes (run, attempt, epoch, dispatch index, group, member). The `controller_epoch` is not a row field: it is read from the ledger-segment header, written once per segment and per cut manifest (R9-3, NRL3-5) |
| `data_ref`, `parent_ref` | struct | §12.1 |
| `outcome`, `reason`, `rule_version` | enum, code, string | §12.3 |
| `gen_version` | counter | Weights that generated the row; NeMo RL's integer counter, resolved to `(controller_epoch, counter)` through the segment header and parent edge (§12.4) |
| `reward_epoch`, `rewards` | int, list of float | Raw reward values |
| `sampler_config_hash`, `seed_inputs` | hash, struct | §14 |
| `chunk` | struct or null | Episode or row range for `step_range` records |
| `derivation` | enum | `observed` or `reconstructed`; a reconstructed record's precision is limited as in §11.2 |
| `precision` | enum | `row`, `step_range`, or `iteration` (§11.2). An `iteration` record names one shard used in one training iteration: `data_ref.row`, `sample_id`, `chunk`, `rewards`, and `seed_inputs` are null, and `outcome` is `consumed` (G-1) |
| `writer_id` | id | Field writer. The provenance MAC is per Arrow segment, in its header, not per record (§12.5, LIN3-3) |

### 12.3 One outcome enum

| Outcome | Reason codes | Replaces (02 revision 7 / 05 revision 7) |
|---|---|---|
| `consumed` | — | `consumed@step` / `consumed` |
| `filtered` | `zero_advantage`, `dynamic_sampling`, `rule:<id>` | `filtered` / `filtered_zero_advantage`, `dynamic_sampling_drop` |
| `stale` | `version_age:<n>` | `stale` / `stale_drop` |
| `wrong_epoch` | — | `old_lineage` / — (a segment whose header carries another controller epoch; should not occur after R8-4, kept as a guard) |
| `refit_invalid` | shard id | `refit_invalid` / — |
| `masked` | `timeout`, `deadline` | `masked` / `timeout_mask` |
| `reward_failure` | service and error class | — / `reward_failure` |
| `storage_unit_lost` | node | `storage_unit_lost` / — |
| `dropped_at_cut` | — | both |
| `redispatched` | new dispatch index | — / `redispatched` |

Rollout-efficiency metrics (07) are computed from these outcomes.

### 12.4 Consumption by weight-version ancestry (LIN2-1)

Revision 7's rule "a checkpoint at step s consumed shard S iff first_step(S) ≤ s" is wrong across rollbacks. Under R8-4 every attempt gets a new controller epoch, and a rollback is a stop plus a new attempt from an older cut, so lineage is a tree of **segments**:

- A segment is one attempt's records, identified by `controller_epoch`. Its parent is `(parent_epoch, branch_step)`, where `branch_step` is the `last_step_included` of the cut it restored. A first attempt has no parent.
- A cut manifest records `last_step_included`: the last optimizer step whose update is in the saved weights. Cuts are taken after the optimizer commit, so **a checkpoint whose cut has `last_step_included = s` includes the update at step s** and the `consumed` records of step s.
- The ancestry path of a checkpoint in segment e with `last_step_included = s` is: steps ≤ s of segment e, then steps ≤ `branch_step` of each ancestor.
- **Rollup per lineage.** The workflow database keeps `consumption_rollup(run, controller_epoch, shard, first_step, last_step, rows)` per segment. The checkpoint consumed shard S iff some segment g on its path has first_step(g, S) ≤ u_g, where u_g is s for its own segment and the child's `branch_step` for ancestors.
- **Superseded and orphaned records.** Records of a parent segment after a branch point are marked `superseded` relative to the child: they remain valid for checkpoints of the parent segment itself and are excluded from descendants. Records not covered by any committed cut (an attempt that ended after its last cut) are `orphaned` and excluded from every rollup. Query results report both: a row-level question gives superseded and orphaned record counts with their precision; a membership question gives superseded and orphaned step ranges per segment and shard.

### 12.5 Ledger writer

Through hook 8d the trainer writes one Arrow segment per optimizer step into its attempt- and epoch-scoped contract path (02), and the run adapter uploads segments idempotently. Each segment carries one provenance MAC (06) over its bytes in its header. The ledger writer (package E) verifies each segment's MAC against the segment list of the signed cut manifest (02, 06), so a segment is accepted only if a signed cut names it; per-record MACs are not used (LIN3-3). When the cut commits, the writer writes the segments the cut manifest lists as Parquet and updates the rollups in one workflow-database transaction. Fused vectorized execution (08) writes one record per (environment batch, iteration) chunk with `precision = step_range`.

### 12.6 Curator filter wrapper (LIN2-8)

Curator's `Filter` and `ScoreFilter` drop rows with `df = df[bool_mask]` (`score_filter.py:207`, `:342`) and emit nothing about them. The framework wraps each filter stage: it computes the same mask, writes dropped row keys with the filter name, version, thresholds, and score (when `score_field` is set) to the `FilterDecision` side table, then applies the mask. Dedup removals already go through ID lists and are recorded from those lists. A `record_dropped` option is offered upstream.

### 12.7 Name-to-hash resolution

Every `name@version` (reward models, judges, environments, partner policies, evaluators, plug-ins) is resolved at lockfile creation to (manifest hash, container image digest); mutable tags are rejected; a version change during a run starts a `RewardEpoch`; latent and embedding caches carry the encoder or VAE hash, asserted on use; RM training consumes an immutable `PreferenceSnapshot`. Partner versions are pinned as stage inputs at submission (R8-10).

## 13. Storage, size, retention, and content addressing

### 13.1 Sizing table (package A)

A ledger record is about 76 bytes before compression with run and step dictionary-encoded (4-byte step; the epoch is in the segment header, 16-byte sample ID, 24-byte `data_ref` core, 3 bytes of outcome and reason, 8-byte counter, 2-byte reward epoch, 4-byte seed input, 12 bytes of rewards, a 2-byte writer code, and a 1-byte code that holds both `precision` and `derivation`); `step_range` records add a 16-byte chunk, about 92 bytes. A null `parent_ref` costs only its validity bit; a set one (composed systems, derived data) adds about 17 bytes (1-byte kind, 16-byte ID). MACs are per segment (one 32-byte MAC per Arrow segment), not per record (LIN3-3). The per-step Arrow segment holds full hashes before dictionary encoding, about 300 bytes per record. Compressed Parquet sizes are estimates (dictionary and run-length encoding), to be measured in P2.

| Reference config | `precision` | Records per step | Steps | Records per run | Raw ledger | Arrow segment per step | Workflow-DB rows per step |
|---|---|---|---|---|---|---|---|
| LLM GRPO, large (512 prompts × 16) | `row` | 8,192 | 10,000 | 81.9 M | 6.2 GB (1.5–3 GB Parquet, est.) | 2.5 MB | 1 segment row + ≤ shards touched rollup upserts |
| LLM GRPO, small (32 × 16) | `row` | 512 | 1,000 | 0.51 M | 39 MB | 154 KB | same |
| SD3.5-M flow_grpo `pickscore_sd3` (48 prompts × 24 images per round, two updates per round) | `row` | 576 | 4,000 (illustrative) | 2.3 M | 175 MB | 173 KB | same |
| Fused control, 4,096 environments in 64 batches (illustrative) | `step_range` | 64 | 10,000 | 0.64 M | 59 MB | 19 KB | same |
| Composition planner turn (04 §10.7: 32 inputs × G = 8; `parent_ref` set, about 93 bytes; illustrative) | `row` | 256 | 200 | 51,200 | 4.8 MB | 81 KB | same |

Per-timestep diffusion data stays in the data plane and is not ledgered.

### 13.2 P2 storage (LIN2-3)

| Data | P2 store | P3 store |
|---|---|---|
| Lineage graph (OpenLineage events, entities, edges with precision), segments, rollups, registrations, gate reports | Tables in the workflow database (E's transactional store) | Marquez or DataHub, fed from the same events |
| Consumption ledger, filter decisions | Parquet on object storage, partitioned `run=/attempt=/epoch=`, files sorted by (shard, step) | Iceberg tables with the same partitioning |
| Datasets and snapshots | lakeFS commits | lakeFS or Iceberg snapshots |
| Checkpoints | Object storage; manifests record per-file hashes | Chunked Merkle manifests with object lock (§13.3) |

**Impact query join.** A query runs in DuckDB: (1) a recursive query in the workflow database computes the activity and artifact closure and, for training data, the candidate segments and their step bounds (§12.4); DuckDB reads these through its PostgreSQL or MySQL scanner extension, whichever the workflow database is, or from a result exported by the runner; (2) only for row-level questions, DuckDB scans the Parquet files of those `run=/attempt=/epoch=` partitions with a filter on shard and step bounds, pruned by Parquet row-group statistics; (3) the result joins both and reports precision and the superseded and orphaned records as in §12.4 (record counts for row-level questions, step ranges per segment and shard for membership questions). The file list per partition lives in the workflow database, so DuckDB reads only committed files.

**Compaction.** The writer emits one Parquet file per committed cut (covering the steps since the parent cut), not per step. A compaction job rewrites a partition into files of 256 MB to 1 GB sorted by (shard, step) when the stage completes or when the partition exceeds 256 files; it swaps the file list in one transaction and deletes replaced files after a 7-day grace, so running queries finish and a point-in-time restore of the database still finds its files.

### 13.3 Merkle manifests

**Phase.** Merkle manifests arrive in P3 (R8-8 thins P2). P2 manifests record a SHA-256 per file, and the P2 manifest hash is the hash of that sorted file list; the P3 format below keeps per-file entries, so P2 manifests remain readable.

From P3, checkpoints of hundreds of GB to several TB are hashed in 64 MiB chunks per file; file hashes form a Merkle tree whose root is the `manifest_hash`. Hashing parallelizes across writers, verification can check any subset, and unchanged files are shared across checkpoints. Committed checkpoints are written under object lock (governance mode, retention per policy tags); deletion for consent or licence reasons uses the governed path in 06 and leaves a tombstone. The cut manifest (02) references the root; signatures (06) sign it.

### 13.4 Retention, snapshot expiry, and point-in-time recovery (LIN2-6)

Consent withdrawal (06) writes a tombstone, deprecates derived artifacts found by the impact query, and deletes raw bytes where the terms require. Retention must not undo it:

- **Lineage and ledger keep only digests and opaque row keys**, so they are retained; they are what finds derived artifacts.
- **Separate storage for consent-tagged data (LIN3-1).** Consent-tagged data lives in its own lakeFS repositories or Iceberg tables, never in a repository shared with untagged data, each with its own garbage-collection policy, so expiring its commits does not touch the immutable history of other data.
- **Dataset snapshot expiry.** For consent-tagged data, snapshot retention is at most the consent terms' deletion deadline. When a tombstone is written, every snapshot holding the withdrawn bytes is rewritten without them as a new **redacted snapshot**, recorded as `wasDerivedFrom` the original, and the old snapshots are expired within that deadline by that repository's garbage collection; time travel to an expired snapshot returns the manifest and tombstone only. The store-side retention job (07) enforces this with dry-run and double-checked deletes.
- **Point-in-time recovery.** Backups of the workflow database and of object storage holding consent-tagged bytes either expire within the deletion deadline or are encrypted with per-period keys destroyed at the deadline. After any restore, the tombstone log (append-only, anchored in the WORM audit log, 06) is replayed before the store accepts reads, so a restore cannot resurrect withdrawn data or un-deprecate derived artifacts. A conformance test withdraws a row, restores a backup taken before the withdrawal, and checks that the tombstone and deprecations are reapplied before any read.

## 14. Reproducibility

**Guarantee.** Re-executing a run from its lockfile, inputs, and recorded seeds produces a checkpoint whose registered metrics are **equivalent** to the original under the TOST procedure of §4.2. Bitwise identity is not guaranteed (asynchronous rollouts, non-deterministic reductions, hardware). Bitwise replay is offered only for a single batch in the research profile (01), with batch-invariance mode on the same hardware SKU.

**Lockfile additions:** hardware SKU and count, driver and CUDA versions, NCCL and cuBLAS versions and determinism settings, PyTorch deterministic flag, batch-invariance mode, and the resolved hash of every `name@version` (§12.7).

**Seeds.** Each sample's rollout and noise seeds are derived as H(run_seed, step, data_ref, sample_index); the inputs are in the ledger, so noise can be regenerated without storing it. **Unpinnable hosted judges:** responses are stored in a `JudgeResponseCache` keyed by scorer hash; re-execution replays them and reports any miss.

**Redacted inputs (LIN3-1).** When an input snapshot has expired after a consent withdrawal (§13.4), re-execution uses its redacted successor and the result is reported as `restated_on_redacted_input`, citing both snapshot hashes; it is not a reproduction of the original run under the guarantee above.

## 15. Build versus reuse

Reused: OpenLineage format, the workflow database and object storage (P2), Marquez or DataHub and Iceberg (P3), lakeFS, MLflow or NGC registries, Curator's dedup engines and text filters, NeMo Evaluator's `gate_runs`, intervals, and paired tests, and object-store checksums with object lock. Built: lineage facets (hashes, reward epochs, segments, precision); the ledger writer, rollups, and compaction; consent-aware snapshot expiry; the `Deployment` emitter and registry admission hook; the contamination and filter wrappers; typed hypotheses, seed-level TOST, σ planning, Bonferroni and online FDR, and registration; tier computation and the ratio transfer test; the manifest format (per-file hashes in P2, Merkle roots from P3, shared with 02) and name→hash resolution.

## 16. Phase placement

Consistent with overview §7.1 as revised by R8-1 and R8-2.

| Phase | Delivered here |
|---|---|
| G0 | Feasibility comparison (§4.1) with descriptive statistics; may use NeMo Evaluator directly, since `gatedtrain-gates` is not required |
| P0 | `gatedtrain-gates` (E-stats): typed clauses, α semantics, intervals, σ planning, power and MDE, AND-gate power plans, registration (§3); parity procedure (§4.2, §4.3) |
| P1a | Package B parity tests against the references are numerical tests owned by 01; no statistical gate |
| P1b | Full parity (§4.2) with per-timestep procedure (b); throughput ratio reported from 08; exact contamination with NFKC; template and slot overlap; image perceptual hashes |
| P2 | Gate nodes in package E; Bonferroni and online FDR for turn gates (§5); tiers, non-LLM rule, transfer test (§6); `ConsumptionRecord` writer, rollups, segments, Parquet ledger with DuckDB queries, lineage tables in the workflow database (§12, §13.2); required edges and registry hook (§11.1); filter wrapper; `ContaminationReport` with per-model thresholds; consent-aware retention (§13.4) |
| P1b–P4 | Reward-hacking detectors in shadow mode as families arrive; promotion after replay calibration (§7) |
| P3–P8 | P3: Marquez or DataHub, Iceberg ledger tables, chunked Merkle manifests (§13.3). P4: video contamination and evaluators. P6a and P6b: scenario, map, and session splits, closed-loop evaluators, Clopper-Pearson safety rates. P7: Bradley–Terry with labeler effects, blocking QC, IPW, on-policy RM calibration, `PreferenceSnapshot`, collusion detection. P8: CAD evaluators |

## 17. Changes

### 17.0 Revision 9 (closing)

Intersection-union per-timestep checks without Holm and prediction bounds on the candidate-seed mean (§4.3); m counting rule and validator labeling check (§3.5, §8); construct-targeted transfer-test confirmation with a disagreement rule and bootstrap sizing (§6.2); two-level ARL₀ bootstrap and the exact 0.26 bound (§7); verl-omni throughput baseline (§4.2); separate consent-tagged repositories, redacted snapshots, and `restated_on_redacted_input` (§13.4, §14); per-segment MACs and `precision` in the schema and sizing table, and `controller_epoch` taken from the segment header instead of each row (§12.2, §12.5, §13.1; R9-3).

### 17.1 Revision 8 (round-2 review)

- The G0 comparison is a feasibility run with 1–2 seeds, a fixed sample count, and a GPU-hour cap; full TOST parity moves to P1b, which also reports the 08-owned throughput criterion (§4).
- σ planning uses the 80% upper confidence bound or a blinded internal pilot; the worked example now needs 11 (not 5) seeds for ±0.025 (§3.6, §4.2).
- Per-timestep comparison uses seed random-effects prediction bounds or per-timestep TOST with Holm instead of a 95th percentile from five seeds (§4.3).
- AND gates plan per-clause power p^(1/m) or block on an averaged clause with the matrix descriptive (§3.5).
- Turn gates use Bonferroni over T_max (1.66× samples for T_max = 18) or online FDR; 1/t² spending is removed; the final gate keeps family-wise control (§5).
- The transfer test is a non-inferiority bound on the evaluator-to-proxy gain ratio over human-confirmed probes with a probe-size calculation (§6.2); blocking clauses need a non-LLM evaluator, with human evaluation for story-level metrics (§6.1); α field semantics and defaults are stated once (§3.2).
- Detectors are calibrated by per-step false-alarm rate and CUSUM ARL₀ by bootstrap replay of human-labelled runs; run-level claims need about 30 runs (§7). Spot checks use Fisher or Boschloo.
- Preference QC blocks on gold accuracy and model fit; Krippendorff's α is monitored; labeling cost is stated (§8). `mde_estimate` is documented as one-sided α = 0.025 and the upstream comment error noted (§2).
- One `ConsumptionRecord` schema, outcome enum, and sizing table in package A; `parent_ref` replaces composition identifiers (§12.1–12.3, §13.1).
- Consumption is defined by weight-version ancestry over segments with per-lineage rollups and superseded and orphaned marks; the update at the cut's `last_step_included` is included (§12.4).
- P2 storage is OpenLineage-format tables in the workflow database plus run-partitioned Parquet queried with DuckDB, with a join and compaction policy; Marquez or DataHub, Iceberg, and chunked Merkle manifests move to P3, and P2 manifests record per-file hashes (§13.2, §13.3).
- Required `used` edges, registry refusal of unresolved hashes, and a conformance test (§11.1); per-edge precision (§11.2); consent-consistent snapshot expiry and point-in-time recovery (§13.4); per-embedding-model thresholds (§10.2); Curator filter wrapper (§12.6).

### 17.2 Revision 7 (from revision 6)

Graded independence tiers and a transfer test replaced the shared-backbone exemption; gates became typed, pre-registered tests; turn-gate bootstraps, the "noise band", and "three seeds" were replaced by multiplicity control, fresh selection subsamples, and TOST with at least five seeds; contamination checks became modality-specific, asymmetric, and persisted; deadline masking was removed; detectors were defined; `data_version` became `data_ref` with a durable ledger; lockfiles gained hardware and resolved hashes; reproducibility became statistical re-execution.

## Notes for the overview and sibling documents

All revision-8 notes are resolved: 02 §2.4 records `last_step_included` and 02 §3 uses this document's schema and enum with no `system_sample_id`; 04 §3 and §10.3 forbid `alpha` inside a family, use Bonferroni at 0.05 for turn gates, block cross-play on the averaged clause, and block story metrics on human evaluation; 06 §12.1 matches §6.1 and 06 §11.1 relies on §13.4; 08 §7.2 owns the P1b throughput criterion; overview open question 12 states the P2 and P3 lineage stores.

## Finding disposition (round 1)

| ID | Section | Resolution |
|---|---|---|
| EVAL-1 | §3, §5 | Typed tests with margins, α, power, MDE, and item-clustered intervals; repeated turn decisions use Bonferroni over T_max or online FDR (updated in revision 8, replacing 1/t² spending). |
| EVAL-2 | §6 | Backbone-family tiers plus a transfer test, now a ratio bound on human-confirmed probes (revision 8); GenEval is T0 for recipes that train on it. |
| EVAL-3 | §7 | Detectors with statistics and triggers in shadow mode; calibration by per-step rate and ARL₀ replay (revision 8, replacing the ten-run rule). |
| EVAL-5 | §8 | Bradley–Terry with labeler effects, balanced randomized designs, IPW with a uniform stratum, on-policy calibration; QC blocks on gold accuracy and fit and monitors Krippendorff's α (revision 8). |
| EVAL-6 | §10.2, §10.3 | Template and slot matching plus prompt-embedding dedup, motivated by measured overlap in flow_grpo's split. |
| EVAL-7 | §4 | G0 feasibility comparison, then registered TOST with at least five seeds, σ upper-bound planning, and per-timestep tests (revision 8). |
| EVAL-8 | §9 | Per-family evaluator sets with independence caveats. |
| RES-6 | §3.3, §5 | Fresh selection subsamples, multiplicity control (Bonferroni or online FDR in revision 8), K ≥ 5, raw paired differences. |
| RES-9 | §4.2 | Three seeds rejected; five is the floor; the seed count follows the power analysis on σ_U. |
| LIN-1 | §12.5, §13.2 | Durable ledger written at each committed cut; Parquet with DuckDB in P2 and Iceberg in P3 (revision 8). |
| LIN-2 | §12.1 | `data_ref` names snapshot, shard, row, and mixture component; `parent_ref` replaces `parent_sample` (revision 8). |
| LIN-3 | §12.3, §12.6 | `FilterDecision` artifacts, Curator filter wrapper, and one outcome enum. |
| LIN-4 | §11, §12.7 | `name@version` resolves to manifest hash and image digest; label batches, reward epochs, partner versions are entities. |
| LIN-5 | §13.1 | Sizing table in package A (82 M records for the large LLM config). |
| LIN-6 | §14 | Statistical re-execution with an extended lockfile, derivable seeds, and cached judge responses. |
| LIN-7 | §10 | Modality-specific, asymmetric, persisted checks with per-model thresholds. |
| LIN-8 | §11, §12.7 | Cache hashes checked on use; RM training consumes immutable preference snapshots. |
| LIN-9 | §13.3 | Chunked Merkle manifests with object lock from P3; P2 manifests record per-file hashes (revision 8). |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| EVAL2-1 | §6.2 | Upper confidence bound of the evaluator-to-proxy gain ratio below r_max on human-confirmed probes, with a proxy-gain precondition and probe-size formula (88 pairs in the example). |
| EVAL2-2 | §5 | Bonferroni α_FW/T_max (1.66× samples at T_max = 18) or LORD/SAFFRON; final gate family-wise. |
| EVAL2-3 | §7 | Per-step false-alarm rate and CUSUM ARL₀ by block-bootstrap replay of human-labelled runs; run-level claims need about 30 runs. |
| EVAL2-4 | §3.5 | Per-clause power p^(1/m) or an averaged blocking clause with the matrix descriptive (default for cross-play). |
| EVAL2-5 | §4.3 | Random-effects prediction bounds with df = R−1, or per-timestep TOST with Holm. |
| EVAL2-6 | §3.2, §6.1, §9 | Non-LLM blocking evaluator rule with human evaluation for story-level metrics; one α semantics with defaults. |
| EVAL2-7 | §3.6, §4.2 | 80% upper confidence bound of σ_s or blinded internal pilot; worked example recomputed. |
| EVAL2-8 | §2, §7, §8 | QC blocks on gold accuracy and BT fit, α monitored; Fisher or Boschloo spot checks; labeling cost; `mde_estimate` is one-sided 0.025. |
| LIN2-1 | §12.4, §11.3 | Ancestry over `(controller_epoch, counter)` segments, per-lineage rollups, superseded and orphaned marks, step-s inclusion defined. |
| LIN2-2 | §12.2, §12.3, §13.1 | One schema, key `(run, attempt, step, sample)` with the epoch in the segment header (revision 9), merged enum, sizing table, in package A. |
| LIN2-3 | §13.2, §13.3 | Workflow-database tables plus Parquet and DuckDB in P2, join and compaction specified; Marquez or DataHub, Iceberg, and Merkle manifests in P3. |
| LIN2-4 | §11.1 | Required `used` edges for data-producing models, model roles, merges, exports, deployments; registry refusal; conformance test. |
| LIN2-5 | §11.2 | `precision ∈ {row, step_range, iteration, stage}` per edge, reported in results. |
| LIN2-6 | §13.4 | Consent-bounded snapshot expiry, expiring or key-destroyed backups, tombstone replay after restore, conformance test. |
| LIN2-7 | §10.2 | Thresholds calibrated per embedding model and part of the report key. |
| LIN2-8 | §12.6 | Wrapper records dropped IDs from `Filter` and `ScoreFilter`. |
| ARCH2-7 | §12.1–12.3 | One outcome vocabulary; `parent_ref` replaces composition identifiers. |
| EM2-2 (statistics part) | §4.1 | Feasibility procedure with fixed sample count and GPU-hour cap; no TOST in G0. |
| PERF2-2 (reporting part) | §4.2 | P1b parity report carries the ≥ 0.9× throughput ratio; criterion owned by 08. |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| EVAL3-1 (counting and validator part) | §3.5, §8 | m counts every blocking evaluator in an AND clause; the validator computes required labels from the actual gate configuration and checks the labeling budget (R9-5). Relaxing the defaults is deferred to P10 (`open-issues.md`). |
| EVAL3-2 | §4.3 | Per-timestep TOST is an intersection-union test without Holm, planned at power^(1/T), or an averaged blocking clause; prediction bounds apply to the candidate-seed mean with √(1/m + 1/R). |
| EVAL3-3 | §6.2 | Confirmation asks about the evaluator's construct; both raters must agree, split pairs are excluded and reported; probe-set size from a bootstrap over pilot pairs. |
| EVAL3-4 | §7 | ARL₀ calibration resamples runs, then blocks within runs; the ten-run bound is the exact 0.26. |
| PERF3-1 (reporting part) | §4.2 | Throughput baseline is verl-omni with at least three interleaved run pairs; owned by 08. |
| LIN3-1 | §13.4, §14 | Consent-tagged data in separate lakeFS repositories or Iceberg tables with their own GC; redacted snapshots `wasDerivedFrom` the original; re-execution on them reported as `restated_on_redacted_input` (R9-6). |
| LIN3-2 (05 part) | §12.2 | §12.2 is the single `ConsumptionRecord` schema, now listing `precision`; 02 points to it. |
| LIN3-3 | §12.2, §12.5, §13.1 | MACs are per segment, verified against signed cuts; the sizing table covers `parent_ref`, `precision`, and the per-segment MAC. |
| LIN3-4, LIN3-5 | — | Deferred to P3; recorded in `open-issues.md`. |
| LIN3-6 | §12.6 | Deferred to P2 implementation; recorded in `open-issues.md`. |