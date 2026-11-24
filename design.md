# gatedtrain, a Composable Post-Training Framework: Overview

**Status:** Proposed design; a prototype (`gatedtrain` 0.4.0, a separate repository) implements revision 24 except the training-seed unit of revision 20 (05 §3.3). **Frozen until G0 closes — changes only through G0 outcomes, entries in `open-issues.md`, or design defects found during implementation (DA3-4 as amended in revision 10).** Sub-designs 03 and 07, and the P3+ parts of 04 and 06, carry the status **"Direction — detailed review after G0"** (R8-20).

**Revision:** 25 (2026-11-22). Adds AI-native step 3a: rootless Docker for the user and the keeper with a printed host setup, the detection of rootful Docker access by the user's account and the downgraded keeper guarantee, the launch rules on a rootless host, a local backend that runs the AI-native path's jobs with GPU accounting, budget and retries, and batch generators run in the keeper's inference container on a copy of the model files it hashed; gatedtrain starts processes on the AI-native path only; rule ids AN-20 to AN-25 and EXT-263 (see "Revision 25 changes"). Revision 24 (2026-11-22). States that a registration's analysis code version names the analysis code (gate statistics and registration semantics), not the package release, so a package release that leaves that code unchanged leaves existing registrations valid (see "Revision 24 changes"). Revision 23 (2026-11-21). Adds AI-native step 2: the keeper daemon that holds the seal under its own OS user, Ed25519 signatures with a trust anchor and `gatedtrain verify`, Foster-Stine alpha-investing keyed by the sealed set (each sealed evaluation spends alpha first, replacing one sealed scoring per project), the scorer library, and the rule ids AN-12 to AN-19 (see "Revision 23 changes"). Revision 22 (2026-11-20). Lets an external NeMo RL run resume after preemption: attempt ids `<run_id>.<A>`, consumption counted per segment for linear chains, rollback refused, and the rule ids EXT-RES-1 to EXT-RES-7 (see "Revision 22 changes"). Revision 21 (2026-11-20). Adds the AI-native project and stage machine: the project, the stage protocol, the pipeline spec and its trust-stage order, the trust-backend and alpha-ledger interfaces with their step 1 stand-ins, and the rule ids AN-1 to AN-11 (see "Revision 21 changes"). Revision 20 (2026-11-19). Specifies the training-seed unit of analysis that 05 §3.3 named but did not define: registered training seeds, seed-level scores, the seed-level `t` (default) or a two-stage bootstrap, seed-level power and minimum seeds (see "Revision 20 changes"). Revision 19 (2026-11-18). A version-only registration amendment keeps a prior `post_hoc: true` (see "Revision 19 changes"). Revision 18 (2026-11-17). Lets one gate block on several clauses, each decided by its own evaluator and combined by intersection-union, adds cluster resampling by an item field and percentile bounds to the bootstrap interval, and exempts scores on the best-end bound from the smoke check's one-value rule (see "Revision 18 changes"). Revision 17 (2026-11-17) exempted a version-only registration amendment across a package rename, including a scorer changed only by the rename, from the `post_hoc` rule (see "Revision 17 changes"). Revision 16 (2026-11-13) defined "shares labelers" for held-out references in the evaluator tiers (see "Revision 16 changes"). Revision 15 (2026-11-12) defined "same programmatic tool" in the evaluator tiers and names construct-aligned T3 metrics (see "Revision 15 changes"). Revision 14 (2026-11-11) defined the precision of an impact answer reached by several paths (see "Revision 14 changes"). Revision 13 defined the interval of the non-inferiority constraint clause, two details of judge qualification, and four definitions the revision-11 text left open, found while writing the prototype's tests, listed in "Revision 13 changes" at the end of this document. Revision 12 added constraint clauses and the conditions under which an LLM or VLM judge may be their sole blocking evaluator.

**Revision 11** (2026-11-11) applied fixes to the gate library, evaluation records, and lineage found in the triage of revision 10, and renamed the analysis unit "prompt" to "item", listed in "Revision 11 changes".

**Revision 10** (2026-11-11) applied design fixes found while building the walking-skeleton prototype (Tier A and Tier B; gaps recorded in the maintainers' decision records), listed in "Revision 10 changes". The implementation plan, the panel review records, and the decision records cited in this document and its change tables are the maintainers' working material and are not published.

**Revision 9** (2026-11-05) was the **closing revision**. Applies the MVP findings of the round-3 review of revision 8 (round-3 review record, not published; decisions R9-1 to R9-10): staged scope, P2-min by default, clarified hooks 7, 8a, and 8d, and package R split by trust zone. Every Deferred round-3 finding is recorded with its phase gate and owner role in `open-issues.md`. Revision 8 implemented R8-1 to R8-20 (round-2 review record, not published); revision 7 restructured the design after the first panel (round-1 review record, not published). Every finding is resolved in the place named in §12; earlier rounds and the archived revision 6 are kept with the maintainers' working material (not published).

**Scope:** Post-training (SFT, preference optimization, online RL, human feedback, distillation) for LLMs, VLMs, image/video/3D diffusion models, speech models, VLA/autonomous-driving policies, flight-control policies, and CAD generation

**Baseline:** NeMo RL `main` at commit `4f39fd8` (2026-11-05), read from a local full clone; verl-omni and the other surveyed frameworks from local clones (sources in §14)

## 1. Summary

Post-training stacks are fragmented by model paradigm. Token models (LLM, VLM, speech tokens, CAD-as-code, tokenized actions) use one family of frameworks; diffusion and flow models use another; simulator-trained control policies use a third. The engineering cost that repeats across them is infrastructure: launching on Slurm and Kubernetes, checkpoint and restart, weight sync, provenance, evaluation gates, and multi-stage pipelines. Algorithms do not repeat: they stay family-specific and change monthly.

The framework therefore ships **contracts and libraries first, and services only when a pilot user needs them**:

1. **Framework-neutral contracts** for decision steps, artifacts, lineage, and a **run contract** through which any trainer reports progress, checkpoints, and completion (§4; sub-designs 01 and 02).
2. **Libraries** usable by NeMo RL, verl-omni, or FastVideo: the denoising-RL math with parity tests (01), and statistical evaluation gates (05).
3. **Narrow, staged upstream hooks in NeMo RL**, landed as small behavior-preserving PRs. NeMo RL never depends on the framework's control plane and never reads leases (§5, D2).
4. **Optional packages**, each gated on a pilot user: a diffusion family package, a workflow layer, a control plane, embodied execution, and composition (§4).

The design is valid whichever trainer runs non-token families. A decision gate (§3) compares the NeMo RL path with adopting verl-omni before any diffusion code is written; the contracts, libraries, and the thin workflow layer are needed in both outcomes.

## 2. Problem, goals, and non-goals

### 2.1 Problem

- Teams re-implement scheduling, weight sync, checkpointing, fault tolerance, evaluation gates, and pipeline glue per modality.
- Engineered generation-model RL exists in verl-omni (a plug-in package on pinned verl with 25+ pipelines; §10) and in paper code; NeMo RL `4f39fd8` has no diffusion or denoise code.
- Domains such as driving, robotics, drones, and CAD need simulators, safety gates, and rewards that general frameworks do not ship.
- Multi-stage pipelines (SFT, RL, distillation, human review, sim-to-real) are scripted by hand, with no lineage, statistical gates, or resumability.

### 2.2 Goals

1. One set of contracts for decision steps, artifacts, lineage, and runs, implementable by any trainer.
2. A new model family is added by writing family plug-ins; a new algorithm that fits existing stages is one file, which may inline its math in a user package (01).
3. The existing NeMo RL LLM path carries zero new bytes per token and stays within 2% per measured stage (§7).
4. Every artifact and every consumed training sample is traceable to its inputs, and impact queries ("which deployed models used this shard, labeler batch, or reward model") are answerable, with the precision of each edge recorded (05).
5. Gates are statistical tests with declared error rates (05).
6. Runs on Slurm first and Kubernetes later, with the site's cluster health service (hardware detection, node diagnostics, and remediation) as the single owner of hardware remediation (03).
7. A minimum security baseline from the first phase that runs on shared clusters (06).
8. Adoption targets with leading metrics, baselines, and owners (§7.3), not only technical parity.

### 2.3 Non-goals

- Replacing NeMo RL's LLM path, its data plane, or its training backends.
- A general-purpose workflow engine or cluster scheduler.
- Splitting one tightly coupled training job across clusters, live migration, or cross-cluster rollout.
- Joint (simultaneous) training of several models; a direction is recorded in 04 for a separate design.
- A labeling product.
- Claims about training quality. Each family's results are measured against a reference, not promised.

## 3. Decision gate G0 (before any family code)

G0 takes three to four weeks and produces four written outcomes. Only hooks 1, 4, and 5 (§5), package A drafts, E-stats, and package B (P1a) may proceed before G0 closes; no other P0 work and no family code starts until it closes. The one exception is a controlled prototype under rule 13 of the project's implementation plan (§2.2; not published): its outputs are tagged `prototype` and `unpromotable` and count toward no task, milestone, or phase exit (R10-11).

| Item | Required outcome | If not met |
|---|---|---|
| **Pilot users** | One LLM pilot user (a team that will run the workflow layer), and one diffusion pilot user (or one embodied pilot user, §3.3) with a product metric, each with a written success criterion. The LLM pilot user states in writing whether it needs automatic recovery in P2 (P2-full, D19) | Stop after P0; ship contracts and libraries only. No written need: P2-min |
| **verl-omni feasibility comparison** | Flow-GRPO on the pilot user's family (SD3.5 Medium or Qwen-Image; both have references) in verl-omni on the target hardware: 1–2 seeds, a fixed sample count taken from the reference configuration, and a declared GPU-hour cap (placeholder: **2,000 GPU-hours**, confirmed at G0 kickoff). Record reward trajectory against the reference, end-to-end samples/s/GPU, and whether verl-omni used recomputed or rollout old log-probs (it recomputes by default). Full TOST equivalence moves to P1b | — |
| **Path cost tables** | Two written cost tables over four quarters (§3.1), reviewed by the project lead | Path decision deferred; G0 does not close |
| **Dated MVP prerequisites** | Upstream: the NeMo RL maintainers' public response to the staged hook list (§5) (no review capacity is assumed; plans use a placeholder pace of 3–4 hook PRs a month), and who tracks data-plane trainer parity (`grpo_train_sync` versus `grpo_train`). Site prerequisites: the P1 and P2 items of §5.1 (exclusive nodes, allocation firewall, per-team Slurm accounts and QOS, a site-local model and data mirror with direct Hugging Face and object-store egress denied, a service node for the per-team P2 runners, and either a second OS user inside allocations or the service node to hold the adapter signing key, 06 §5.2). A site KMS/PKI for signing keys (§5.1). Cluster health service, P3 store-operator, and other P3 prerequisites are not G0 items; they form the **P3 entry gate** (§7.1) | Without NeMo RL acceptance: hooks land as a plug-in against a pinned NeMo RL and P1 is re-estimated. Without the site prerequisites: P1 and P2 run on dedicated partitions by default |

### 3.1 Path rule (symmetric and cost-aware)

Each path is costed in the same table form, with engineer-weeks and recurring effort per quarter:

| Path N: build and maintain package D | Path V: adapter and contributions |
|---|---|
| Build D for the G0 family (P1b) | Run-contract adapter for verl-omni, lineage emitter wiring, gate wiring |
| Per-release rebase of D on NeMo RL | Per-release rebase of the adapter on verl and verl-omni |
| Onboarding each further family into D | Upstream contributions to verl-omni needed by the users (and their review latency) |
| Upstream hooks beyond P0 that only D needs (hooks 7, 9, 10, 11) | Operating and securing a second training stack (06 hardening of Ray, data plane, allow-lists) |
| Support load on the gatedtrain maintainers | Support load split with the verl-omni maintainers |
| P1b GPU-hour budget, extrapolated from the G0 feasibility run to the P1b seed count | Same, on verl-omni |

The rule is applied in both directions. A path is eligible if the feasibility run shows it can plausibly meet the P1b exit criteria and it meets the pilot user's written requirements (Megatron-scale models, NeMo support, the data-plane trainer, or acceptance of the verl stack). If both paths are eligible, the cheaper total wins. If the pilot user's requirements force the more expensive path, the cost difference is recorded with the pilot user's written agreement. The choice and both tables are recorded in this document.

### 3.2 What package A serves under path V

Under path V package D is not built, and nothing consumes `StepForward` (which now lives in package B, §4). Package A is still consumed by: **gates** (E-stats reads artifact manifests and evaluator records), **lineage** (the verl-omni adapter emits `ConsumptionRecord`s through the A emitter library), **the run contract** (verl-omni runs report heartbeat, cut manifests, and completion through the adapter), and **the workflow layer** (P2 stage inputs and outputs are A manifests). The persisted step-field layout stays experimental until an `ACTION` family uses it (§8).

### 3.3 Embodied users

An embodied pilot user may take **P6a** (fused continuous control, §7) in place of P1b in the MVP. Then the diffusion comparison is skipped unless a diffusion user also exists, and the P6a exit criteria replace those of P1b.

**Kill criteria** for the whole project, reviewed at the end of each phase: no pilot user by the end of G0; the NeMo RL maintainers decline the hooks and the plug-in fallback costs more than 1.5× the hook work; or, under path N, the P1b family misses its exit criteria after one re-plan.

## 4. Package decomposition

| Package | Contents | Depends on | Built in |
|---|---|---|---|
| **A. `gatedtrain-contracts`** | Pure Python: step types, declarative field schema, artifact manifests, lineage records, run-contract file formats including the job-start handshake (`issue_epoch` request and response, "do not requeue" exit code 79), the contract emitter library and verb-signature verification, the contract transport interface with its POSIX-directory and object-store (conditional put; optional dependency) implementations, the `ConsumptionRecord` schema and outcome enum (one sizing table), per-primitive capability records. pyarrow optional | Nothing (no torch, Ray, or NeMo RL) | P0 |
| **B. `gatedtrain-denoise`** | Plain-torch denoising-RL math (SDE step log-probs, analytic Gaussian KL, step selection, segment-level ratios, DiffusionNFT loss) with parity tests against flow_grpo, dancegrpo, diffusionnft; torch-typed protocols `ModelAdapter`, `StepForward`, and the loss-kernel signature `(role_outputs: Mapping[str, StepOutputs], batch: StepBatch, cfg, counts: NormCounts) -> (loss, metrics)` (01 §7.1); per-step KL input and segment-level ratios (formerly hooks 2 and 3) | A, torch | P1a |
| **C-up. NeMo RL hooks** | Upstream PRs only (§5); does not import A | NeMo RL | P0 onward |
| **C-ext. `nemo-rl-gatedtrain`** | Adapters and loss wrappers for NeMo RL, emitter wiring for the run contract and provenance, signature verification of verbs | A, B, NeMo RL with C-up | P0 |
| **D. `nemo-rl-diffusion`** | AutoModel `StepForward` implementations, SDE rollout engine, rewards, recipes; pinned to a NeMo RL version, as verl-omni is pinned to verl | B, C-ext | P1b (path N only) |
| **R-agent. `gatedtrain-runtime-agent`** | In-allocation side: the out-of-process run adapter (lease handling, fencing, signing with a per-attempt key, commit gating), A's contract transport; reaches outside only through a narrow authenticated RPC to `gatedtrain-runtime-svc`. Store credentials never enter jobs | A | P2 (minimal), P3 (full) |
| **R-svc. `gatedtrain-runtime-svc`** | Central side: store interface; broker protocol and its single implementation; `ControlAPI`; `issue_epoch`; the `SchedulerPlugin` and `IntentPolicy` interfaces with P2 defaults and one Slurm plug-in; the run record (an R-owned envelope of keys, attempts, epochs, and cut pointers, plus namespaced, separately versioned owner sections); the rendered `WorkloadSpec`, persisted as an A-format artifact at first submission | A | P2 (minimal), P3 (full) |
| **E. `gatedtrain-workflow`** | Workflow spec and IR, thin runner, metric gates, lockfiles, manifests, lineage as OpenLineage-format tables, ledger writer (run-partitioned Parquet) | A, R, E-stats, NeMo-Run | P2 |
| **E-stats. `gatedtrain-gates`** | Statistical gate library wrapping NeMo Evaluator (05); released separately because G0 and P1b need it | A, NeMo Evaluator | P0 |
| **G. `gatedtrain-control`** | Linearizable store client, run supervisor, further `SchedulerPlugin` and `IntentPolicy` registrations, health-service adapter | A, R | P3 |
| **H. `nemo-rl-embodied`** | Fused in-process vectorized execution, light torch policy backend, simulator plug-ins, safety shield with a batched tensor API | A, B; C-ext optional (fused mode needs only the run-contract emitters) | P6a, P6b |
| **F. `gatedtrain-compose`** | Composition as Environment and RewardFn plug-ins plus a loop library; a generic `parent_ref`; partner versions pinned as stage inputs at submission; partner pools placed through the broker | C-ext, E, G | P10 |

**Dependency rules.**

1. NeMo RL (with C-up) never imports A, R, E, G, or F, and never reads leases. A run talks to the outside only through the run contract (02): records it writes (heartbeat with a progress counter, cut manifests, completion manifest) and two verbs it accepts (`pause_and_checkpoint`, `stop`), carried over the authenticated contract transport.
2. Signing is outside the trainer; verification is in A's library. All lease handling, fencing, signing, MAC checks, and commit gating live in the run adapter in `gatedtrain-runtime-agent`, outside the trainer process. The run makes no store calls, and store credentials never enter jobs.
3. A is pure Python. Anything torch-typed lives in B; plug-in math takes tensors, not `KVBatchMeta` or `DataPlaneClient`; data-plane access lives in thin adapters in C-ext or D.
4. E and G both depend on R; G does not depend on E. The run adapter, `ControlAPI`, the plug-in interfaces, and the run-record envelope exist once, in R; G only registers further plug-ins.
5. The placement broker accepts an opaque fencing token and knows nothing about workflows (03).
6. Composition is built from ordinary plug-ins through `parent_ref`; no general contract mentions turns or system samples (04).

## 5. NeMo RL upstream asks

Each ask is a separate small PR with its own tests. Asks are staged so that nothing lands upstream before a user needs it (R8-3). Hook numbers are kept stable from revision 7; retired numbers are not reused.

**Before G0 closes** (only these three):

- **Hook 1.** `STEP_LOGPROB` as a new `LossInputType`, `MetricNormalizer.STEPS`, and a `GlobalNormCounts` argument that the loss wrapper passes **only to losses that declare `STEP_LOGPROB`** in their input types. `ClippedPGLossFn` stays token-only and unchanged: its `__call__` (`nemo_rl/algorithms/loss/loss_functions.py:698`) accepts no `**kwargs` and applies the `[:, 1:]` shift in its body, and `wrapper.py:156-161` passes keyword arguments, so an unconditional new keyword would break every LLM recipe. Hook 1 is **path-N only**: its only consumer is package D.
- **Hook 4.** Optional field registration: partitions that contain only `TOKEN` steps carry none of the new fields.
- **Hook 5.** An `unclassified` result from `classify_rollout_failure` (every caller tests `is FailureClass.INFRA`, so the change is local).

**After G0** (within P0, or in the named phase):

- **Hook 6.** A fleet-health reporting callback (callback only).
- **Hook 6b.** A post-refit per-shard weight digest, opt-in, proposed only after a prototype shows the cost per synchronizer and engine (P4).
- **Hook 8a-1.** Run-contract emitters (heartbeat with a progress counter, completion manifest, cut manifest) and the `stop` verb over the authenticated contract transport; the controller epoch issued at launch is recorded once per cut manifest and ledger-segment header, not per row (by P2).
- **Hook 8d.** A per-step provenance export callback feeding the consumption ledger: consumed rows first (P2); drop reasons, which need instrumentation at several drop sites, later. Hooks 8a-1 and 8d may ship first as C-ext shims against a pinned NeMo RL, so P2 is not blocked by upstream merge pace.
- **Hook 11.** An authenticated data-plane transport, only if the existing transport cannot be configured for it (P1b).

**Later** (each after a prototype or with its phase):

- **Hook 7.** `pre_advantage`, `post_optimizer_step`, and `post_rollout` callbacks inside the optimizer-commit fence (P5, after a DiffusionNFT prototype). Its parts, each prototyped before the PR: (a) refit of a non-policy role, through `WeightSynchronizer` (`sync_weights` has no role parameter, `nemo_rl/weight_sync/interfaces.py:93-99`) or by a package-D adapter swap before a normal refit; (b) save/restore registration for worker-resident state such as EMA, which the checkpoint does not hold today; (c) explicit ordering on `grpo_train_sync` before refit and checkpoint, not reliance on call order.
- **Hook 8a-2.** `pause_and_checkpoint` with `then=hold`/`continue`, which turns 8a into a controller state machine; only after a prototype.
- **Hook 8b.** Contract-mode checkpoint storage: attempt- and epoch-scoped paths, no run-side deletion, restore from an adapter-provided cut (P3).
- **Hook 9.** A generalized `CheckpointParticipant` protocol extracted from the existing Gym checkpoint adapter (P4).
- **Hook 10.** An estimator registry replacing the if-chains in `grpo.py` and `ppo.py` (P5).

**Gated**, raised only in the named phase: a per-row bootstrap value and discount (or step-duration) input for GAE (P6a; NeMo RL's GAE uses a zero terminal bootstrap); context-parallel slice fetch from the data plane if an adapter in D cannot provide it (P4).

**Retired:** hooks 2 and 3 (per-step KL input and segment-level ratios stay in package B); weight-version packing (versions stay NeMo RL integers; lineage identity is `(controller_epoch, counter)`, recorded outside the trainer, D6); in-run lease handling and self-fencing (moved to the adapter in R); the stage-runner refactor of `SingleController`.

Recommended defaults for `min_healthy_shards`, `refit_timeout_s`, and `checkpoint_must_save_by` ship in framework recipes and launchers (07), not as upstream default changes.

### 5.1 External dependencies and site prerequisites

| Provider | Change | Needed by |
|---|---|---|
| NeMo RL (upstream maintainers) | Response to the hook list recorded in G0; no review budget is assumed (planning placeholders: 3–4 hook PRs per month, 5 working days per review round); data-plane trainer parity tracked by the gatedtrain upstream contact | G0 |
| Site KMS/PKI | Issues and holds the signing keys (manifest signing, per-attempt adapter keys; 06) | G0 |
| Site prerequisite | Exclusive nodes and allocation firewall, or dedicated partitions by default (06); per-team Slurm accounts and a QOS for the GPU-hour cap (B14 backstop); a site-local model and data mirror, with direct Hugging Face and object-store egress denied (06 §9) | P1 |
| Site prerequisite | A service node for the per-team P2 runners (not a login node) | P2 |
| Site prerequisite | An administrator-set framework-managed marker (account or QOS) that users cannot set | P3 |
| Slurm query service (optional) | If the site runs a shared Slurm query cache: status queries by job ID for the P2 runner, optionally `sacct` filtered by job name and user; a cached "not found" is mapped to "unknown" (03) | P2 |
| Site prerequisite | Reachability from compute nodes to `gatedtrain-runtime-svc` for the agent's narrow RPC (no store relay: store credentials never enter jobs); a cluster-profile admission check | P3 |
| Cluster health service | Node diagnostics triggered by a framework-emitted event when the framework cancels a job (needed where diagnostics otherwise run only when the remediation logic acts); remediation logic in advisory mode for tagged jobs | P3 |
| Cluster health service | Job-controller guard keyed on the administrator-set marker and verifying identity against the store nonce; per-cluster switch so tagging and `--no-requeue` start only when that cluster's supervisor leaves shadow mode; grouping of resubmitted jobs by step key for per-cycle RCA (NVRx `attempt_index` stays 0 without Slurm requeue); intents through the `ControlAPI` (03 §8) | P3 |
| Cluster health service | Break-glass that also triggers on store unavailability, using single-use, expiring, job-bound signed action IDs | P3 |
| Cluster health service | If the job monitor runs automatic RCA, it stays on until the health-service watch for framework runs is live (a P3 exit criterion) | P3 |
| Operator (P2) | Runs the P2 store in both P2 variants, following 07 | P2 |
| Store operator (P3) | Runs the linearizable store (03): single primary, no asynchronous replicas, with alert response at the site; identified at the P3 entry gate | P3 |
| NeMo-Run | Executor option to omit the hard-coded `sbatch --requeue` (`nemo_run/run/torchx_backend/schedulers/slurm.py:148`, `:180`, `:253`); P2 avoids it by submitting its own `sbatch` (`--no-requeue` in P2-min, `--hold` in P2-full) | P3 |
| NeMo Evaluator | Cluster-count-aware intervals in `clustered_ci` (05) | P1 |

The framework side of the hand-off: the broker refuses tagged placement on any cluster without the guard. Rows needed by P3 are prerequisites of the P3 entry gate (§7.1), not of G0. Not requested: a stage-runner refactor of `SingleController`, variable `dp_size` (only with elastic pools in P12), and anything in packages A, R, E, G, or F.

## 6. Key decisions

| # | Decision | Resolves | Detail |
|---|---|---|---|
| D1 | Contracts and libraries first; services only with a pilot user; G0 decides NeMo RL versus verl-omni for diffusion with a feasibility run and symmetric cost tables | EM-1–4, DA-1, DA-2, DA-4, DA-5, EM2-2, DA2-3 | §3, §4 |
| D2 | Invert the dependency: NeMo RL reports through the run contract and accepts two verbs; leases, budgets, gates, and fencing live outside it, in the run adapter in R | ARCH-1, ARCH-3, ARCH-8, NRL-6, ARCH2-2 | 02, 03 |
| D3 | Upstream asks limited to §5 and staged: only hooks 1, 4, 5 before G0; no stage runner | NRL-1, NRL-4, NRL-5, ARCH-2, ARCH2-1, DA2-1, NRL2-7 | §5, 01 |
| D4 | Algorithms own the KL estimator, loss aggregation, old-policy strategy, step selector, and group sampler; policies declare only what they can compute; stages may be fused | RES-1–5, PERF-3, PERF-4, ARCH-4 | 01 |
| D5 | Flow-GRPO uses rollout log-probs as the ratio denominator when rollout and training share modules and kernels; trainer recompute is used for LLM GRPO and for disaggregated engines. `AlgorithmSpec.rollout_role` (default `policy`) selects which weights are synced to rollout (DiffusionNFT samples with the EMA `old` role); drift beyond threshold is an error unless `on_drift: recompute` is set and recorded | RES-1, PERF-3, RES2-1, RES2-2 | 01 |
| D6 | Checkpoints extend the existing rollout snapshots and Gym protocol; paths are scoped per attempt and epoch; a cut exists only when committed in the store; previous-cut fallback. Weight versions stay NeMo RL integers; lineage identity is `(controller_epoch, counter)` with a new controller epoch per attempt; a rollback is `stop` plus a new attempt from an older cut | NRL-2, ARCH-6, DS-1, DS-5, DS-6, PERF-1, FMEA-3–5, FMEA-8, DS2-5 | 02 |
| D7 | The control plane requires a linearizable store, leases tied to progress, and model-checked protocols before implementation. On lease loss the adapter stops committing, uploading, and deleting; the run keeps training and refitting for at most `fenced_compute_max_s`, is then held in place with its allocation for at most `fenced_hold_max_s` (the adapter sends `pause_and_checkpoint(then=hold)` and releases a hold with `pause_and_checkpoint(then=continue)` after a same-epoch renewal; still two verbs), then stopped; `stop` and safety pauses are always honored; fenced time counts against budgets; `safety_critical` runs use `fenced_compute_max_s = 0`. A replacement is released only when the old job is scheduler-terminal and the fencing bound has elapsed, or the old processes are confirmed dead (`NODE_FAIL` is unconfirmed); store-outage recoveries are an operator-acknowledged staggered bulk recovery; recovery prefers `scontrol requeuehold` or a recovery QOS; hardware-evidenced resubmits have a separate node-scaled budget; fenced and adapter-caused exits never trip crash-loop breakers | DS-2–4, DS-8, DS-9, OPS-1, OPS-3, FMEA-1, FMEA-2, FMEA2-1, FMEA2-2, DS2-B | 03 |
| D8 | The cluster health service owns hardware detection and node verdicts; the framework owns in-run decisions. In P2-min, jobs stay untagged and are submitted with `--no-requeue`, the health service is used for detection and RCA only, and recovery is an operator `retry_stage`; P2-full keeps jobs requeueable with `issue_epoch` refusing stale requeues (D19). From P3, tagging (and `--no-requeue` for P2-full jobs) switches on per cluster only when its supervisor leaves shadow mode; the framework emits a diagnostics-triggering event on cancel; the guard keys on an administrator-set marker and verifies the store nonce; break-glass covers store outages; the job monitor's automatic RCA stays on until the health-service watch is live | OPS-2, DA-3, FMEA-9, OPS2-1, OPS2-2, OPS2-10, FMEA2-8, DA3-1 | 03, 07 |
| D9 | A minimum security baseline lands with the first phase on shared clusters. The research profile on shared clusters enforces safe formats, the code-execution block, the plug-in allow-list, and policy tags on the storage side through launcher-issued, tag-scoped read credentials; its outputs are `unpromotable` without a signed import stage. `safety_critical` data requires a validated workflow; standalone fused runs and the research profile refuse it, while a fused stage inside a validated P2 workflow may use it. The operator role excludes `decide_gate`, release, and safety resume. A GPU-hour cap applies from P1 via Slurm QOS | GOV-1–8, RT-1–8, GOV2-1, GOV2-2, GOV2-5, RT2-4, RT2-8 | 06 |
| D10 | Gates are typed statistical tests from one library wrapping NeMo Evaluator; evaluator independence is graded by backbone family and verified empirically | EVAL-1–3, EVAL-6–8, RES-6, RES-9 | 05 |
| D11 | A consumption ledger makes sample-level lineage durable, with consumption defined by weight-version ancestry and each edge's precision recorded; lineage is emitted in OpenLineage format; reproducibility means statistical re-execution | LIN-1–9, LIN2-1, LIN2-5 | 05 |
| D12 | Fused in-process vectorized execution is a first-class mode with per-iteration guarantees | EMB-1, EMB-4, EMB-7, EMB-8 | 08 |
| D13 | Composition starts with a scripted pilot that must show a gain at equal GPU-hours; it is built from Environment and RewardFn plug-ins; the IR has no turn node | DA-6, ARCH-5, EVAL-4, EMB-5 | 04 |
| D14 | P2 ships a thin workflow layer (D18) that implements the backend contract; a production engine is chosen later with adopters who operate one (P9) | EM-5, EM2-3, DA2-2 | 04 |
| D15 | A research profile runs on one node without the store or workflow engine, with sweeps and per-step diagnostics; manifests are still signed (06 B9) and shared-cluster rules follow D9 | RES-8, GOV2-7 | 01 |
| D16 | Performance targets are per-stage, per-iteration, and per-checkpoint measurements, plus one end-to-end throughput floor for P1b (≥ 0.9× the reference) and registered checkpoint intervals | PERF-10, PERF-1, PERF-2, PERF2-2, PERF2-3 | 08 |
| D17 | **Contract transport and isolation.** The run contract travels over a transport interface with a POSIX-directory and an object-store (conditional put) implementation, all in package A with the contract emitter (the object-store implementation is an optional dependency); the run adapter in R uses them; paths are scoped per attempt and epoch; verbs and pointers are signed by the adapter (Ed25519) and verified by the run; the store accepts outputs only when they match signed committed cuts. Liveness separates unreadable contract files (I/O error: consult storage verdicts, do not revoke) from readable frozen counters; checkpoint phases count I/O progress; phase budgets come from measured p99; more than k simultaneous stalls on one filesystem are treated as infrastructure. The launcher supervises the adapter; adapter releases are canaried per cluster and pinned in lockfiles | DS2-A, RT2-2, ARCH2-6, FMEA2-3, FMEA2-5, OPS2-5 | 02 |
| D18 | **Thin P2.** Spec, stage DAG, metric gates, lockfiles, manifests; lineage as OpenLineage-format tables in the workflow database; ledger as run-partitioned Parquet queried with DuckDB (Marquez and Iceberg in P3); append-only audit log with a WORM anchor. Security: B13 identity, B14 hard GPU-hour cap with a Slurm QOS backstop, B20 policy tags, per-team Slurm accounts and runners with `run_root` isolation, and the authenticated contract transport (no store credential in jobs). The runner is a service on a service node and queries Slurm through a shared Slurm query cache where the site runs one. **P2-min** (default) submits with `--no-requeue` and recovers by operator `retry_stage`; **P2-full** adds its own `sbatch --hold`, `issue_epoch`, and a janitor (D19). The operator of the P2 deployment runs the P2 store in both variants. P2 model releases stay non-public (no external release) until P3 adds re-scoring audits, trigger probes, and membership-inference checks | EM2-3, DA2-2, LIN2-3, RT2-1, RT2-3, DS2-6, FMEA2-7, GOV2-6, OPS2-6, DA3-1, DA3-3 | 04, 06 |
| D19 | **P2-full requeue safety**, built only if the LLM pilot user confirms in writing at G0 that it needs automatic recovery in P2. `issue_epoch(run_id, job_id, restart_count)`, called first in the batch script, succeeds only if the job is the stage's current attempt, the record is `submitted` or `running`, the workflow is not cancelling or cancelled, and the cap is not reached; otherwise the start script exits with the "do not requeue" code 79 (run contract, A) and the job's Slurm requeue limit bounds loops; the janitor releases only holds it placed. The cut to restore is resolved at job start. `retry_stage` requires the previous job to be absent from the queue with no requeue pending. If the runner is unreachable the start script holds the job and exits | DS2-C, FMEA2-6, OPS2-6, RT2-3 | 03, 04 |
| D20 | **Package R and C split.** R sits below E and G and splits by trust zone into `gatedtrain-runtime-agent` (in allocation) and `gatedtrain-runtime-svc` (central), with plug-in interfaces, the run-record envelope, and the persisted `WorkloadSpec` in R (R9-9); C splits into C-up (upstream PRs) and C-ext (`nemo-rl-gatedtrain`); A is pure Python with the emitter, contract transport, ledger schema, and outcome enum; torch-typed protocols live in B; `AlgorithmSpec` names a B-level loss kernel with thin wrappers in C-ext, D, H, and verl | ARCH2-3, ARCH2-4, ARCH2-5, ARCH2-7, ARCH2-8, ARCH2-9, DA2-6, ARCH3-1–ARCH3-4, ARCH3-7 | §4, 01, 03 |

## 7. Phase plan

### 7.1 Phases

The MVP is **G0, P0, P1a, P1b (or P6a for an embodied pilot user), and P2**, delivered in **two quarters**. P1a runs in parallel with P0; P1b and P2 run in parallel after P0. Later phases start only with a pilot user and an owning maintainer. Exit procedures are in 05; G0, P1, and P2 services are non-production (07).

**Staged scope (R9-1).** Increment 1: G0, P0, P1a. Increment 2, at G0 close: P1b and P2 (requires the pilot users and the NeMo RL maintainers' response to the hook list). P3 is a separate scope decision behind the **P3 entry gate**: dated health-service prerequisites (§5.1), an identified P3 store operator, the other P3 rows of §5.1, and the P3 blockers in `open-issues.md`. The end state without P3 is P2 plus the health service as the site runs it.

| Phase | Deliverable | Depends on | Exit criteria |
|---|---|---|---|
| G0 | Decision gate (§3) | — | Four written outcomes; path N or V recorded with both cost tables, including the P1b GPU-hour budget; dated MVP prerequisites or the dedicated-partition default; P2-min or P2-full recorded |
| P0 | Packages A and E-stats; C-ext skeleton; hooks 1, 4, 5, then hooks 6 and 8a-1 after G0; research-profile skeleton; conformance kit v0; P0 security items (06: B9 signed manifests and safe formats, B10, B11, B18, B19, T0 review and provenance) | Hooks 1, 4, 5 may start before G0 closes | LLM GRPO recipes on `grpo_train_sync`: zero new bytes per token on token partitions; per-stage overhead ≤ 2% in interleaved A/B runs (08); NeMo RL CI green; decision-step worked examples (01) pass; hook-1 safety: every existing LLM recipe and loss is unaffected (`ClippedPGLossFn` receives no new argument) |
| P1a | Package B with parity tests | torch only; parallel with P0 | B's parity tests pass against flow_grpo, dancegrpo, and diffusionnft |
| P1b | Path N: package D for the G0 family (synchronous, colocated, on `grpo_train_sync`); research profile complete; Ray and data-plane hardening, hook 11 if needed, plug-in allow-list (06); GPU-hour cap via Slurm QOS within the P1b GPU-hour budget (§3.1). P1 security (06): egress only through the site-local model and data mirror; storage-side policy tags via launcher-issued credentials; research lineage recorded by the launcher; team UID as accepted MVP trust domain with per-run ACLs. Path V: the verl-omni adapter with the same criteria run on verl-omni | P0, P1a, G0 | TOST equivalence to the reference (flow_grpo for SD3.5 Medium; flow_grpo or verl-omni for Qwen-Image) on final reward and on an independent evaluator per 05, with at least five seeds and a power analysis, σ planned from an 80% upper confidence bound, margins registered in advance; per-timestep ratio statistics within the reference's own recompute drift; end-to-end samples/s/GPU ≥ 0.9× verl-omni on the same hardware and configuration over at least three interleaved run pairs; checkpoint interval registered (Young/Daly) with expected loss on failure reported |
| P2 | **P2-min** (default): thin package E and minimal R per D18; `--no-requeue` with operator `retry_stage`; the health service for detection and RCA only; hooks 8a-1 and 8d (consumed rows), as C-ext shims if not yet merged; manifests with per-file hashes; B12 (`find_by_key` bound to team identity and nonce), B13, B14, B20, per-team accounts and runners; injection canaries and a non-LLM blocking evaluator for LLM/VLM judges (a qualified judge may block a constraint clause alone, 05 §6.1). **P2-full** (only on the LLM pilot user's written need at G0): adds `issue_epoch`, own `sbatch --hold`, and the janitor (D19). P2 security (06): adapter-owned cut prefix with copy-on-write restore; runner limited to `sbatch`, `scontrol release`, `scancel` on its team's account; per-principal rate limits and a per-run paid-judge spend cap; owner-set `safety_critical`; WORM anchors at most every 5 minutes; no release of PII- or consent-trained models before P3 | P0 | The LLM pilot user's SFT → GRPO → gate pipeline runs on Slurm with gates reporting power and MDE; an impact query from a data shard to its checkpoints succeeds using weight-version ancestry; a workflow violating gate independence is rejected; restricted samples cannot reach external rewards or telemetry; an approval by the requester is rejected; no job is requeued by Slurm and `retry_stage` refuses while the previous job is queued (P2-full: a requeue after cancel and a stale requeue are both refused by `issue_epoch`); the P2 store is run by the operator of the P2 deployment; model releases stay non-public |
| P3 | R full and package G: linearizable store, run supervisor with safety rules, broker protocol, Slurm plug-in without NeMo-Run local state, Kubernetes plug-in, RBAC, plug-in signing, runtime budgets, health-service adapter and the §5.1 health-service changes; Marquez and Iceberg; chunked Merkle manifests (05 §13.3); hook 8b; re-scoring audits, trigger probes, and membership-inference release checks; P3 operator alert response in place at the pilot site (07 §2); TLA+ models checked before coding, covering terminal-but-alive jobs, requeue of finished jobs, delayed health-service actions, and the contract layer | P2; P3 entry gate and separate scope decision | Model-checked invariants (03); deterministic simulation and chaos tests (07) pass, including the productive-time and tail-recovery chaos criteria; SLOs (07) met for four weeks on the pilot users' runs; health-service watch live (the job monitor's automatic RCA is disabled only after it); operator alert response in place; external releases pass re-scoring-audit, membership-inference, and trigger-probe checks |
| P4 | Asynchronous diffusion and video: hook 9 and `CheckpointParticipant` for the sampler, sampler drain mode, buffer persistence policy, context parallelism within NVLink domains, VLM-judge reward service; hook 6b if the prototype justifies it | P1b, P3 | Per-checkpoint goodput loss ≤ 2% at the registered interval; end-to-end samples/s/GPU on 4 nodes reported against single node; intra-node context-parallel efficiency ≥ 80% |
| P5 | Algorithm breadth: DiffusionNFT prototype, then hooks 7 and 10; DanceGRPO-style train-time sampling; Diffusion-DPO as an offline recipe in D with pair microbatching | P1b | Each algorithm passes TOST against its reference; each new algorithm that fits existing stages is one file |
| P6a | Package H fused continuous control: fused vectorized mode, continuous-control PPO, batched shield, `proposed_action` training rule, discount field and the gated GAE ask | P0; P2 for `safety_critical` data | PPO reward curves equivalent to rsl_rl or rl_games within registered margins; env-steps/s ≥ 80% of the reference library; export equivalence on target hardware |
| P6b | Package H VLA flow heads, with frozen VLM planners as co-located batch services in a declared hybrid mode | P3, P4, a named rollout engine | Equivalence to a named VLA reference within registered margins (08) |
| P7 | Human-feedback connector, preference store, reward-model lifecycle; data-plane DPO trainer with an owning maintainer and a deprecation plan for `dpo.py` proposed upstream | P2, NeMo RL maintainers' agreement | One label → RM → RL cycle; RM calibration reported on on-policy samples; DPO recipes equivalent to `dpo.py` |
| P8 | Sandbox for model-generated code; CAD-as-code family | P3 | Sandbox escape suite passes; code-executing environments blocked until then (06) |
| P9 | Production workflow-engine backend chosen with adopters who operate one | P3 | Backend conformance (04); a composition-sized synthetic loop (6 rounds × 3 nodes with turn gates, toy stages) with zero duplicate launches |
| P10 | Composition: scripted pilot first, then package F; partner pools are separate placement units on the trainer's cluster (cross-cluster partner services are a P11 decision) | P4 or P6a/P6b, P9, pilot gain | Pilot shows a gain at equal GPU-hours on an independent evaluator; then the alternating loop runs six rounds unattended |
| P11 | Multi-cluster placement for stages, services, and batch work | P3, pilot user | Same-hardware migration with replay validation |
| P12 | Elastic capacity, including variable `dp_size` | P3, pilot user | Trainer node failure repaired without restart; rollout pools resize with ≤ 1% rollout-efficiency loss |

### 7.2 Effort estimate

Assumptions: contributors familiar with NeMo RL and Ray; review time of the NeMo RL maintainers is neither counted nor assumed; GPUs available within the declared caps. Numbers are round-2 estimates in engineer-weeks over elapsed weeks, with P2 re-estimated in round 3.

| Phase | Path N | Path V |
|---|---|---|
| G0 | 6–8 engineer-weeks over 3–4 weeks | same |
| P0 | 24–30 engineer-weeks over 8–10 weeks (hooks, A, E-stats, C-ext; data-plane trainer parity is tracked per G0) | same |
| P1 (P1a + P1b) | 30–36 engineer-weeks over 10–12 weeks, including diffusion RL expertise; P1a starts with P0 | 5–12 engineer-weeks over 5–6 weeks (P1a plus the verl-omni adapter) |
| P2 | P2-min: 12–16 engineer-weeks over 6–8 weeks. P2-full: 24–30 engineer-weeks over 8–10 weeks, including checking the P2 requeue TLA+ model before any P2 runner code (03 §12.1) | same |

Critical path: G0 (3–4 weeks, overlapping the start of hooks 1, 4, 5 and P1a), then P0 (8–10 weeks), then P1b in parallel with P2. Under path N this is about 21–26 weeks with peak parallel effort of about 5–6 engineer-weeks per calendar week; under path V the tail after P0 is set by P2 and the peak is about 5. Both fit the two-quarter MVP only with the G0 prerequisites in place and P2-min; without them P1b is re-estimated, and P2-full lengthens the P2 tail by about two weeks and adds about 8–10 engineer-weeks.

P3 is not in the MVP numbers and is a separate scope decision (§7.1). Before P3 leaves non-production, the operating site needs alert response for the store, supervisor, broker, run adapter, and health-service adapter (07 inventory), coordinated with the store operator; 07 §2 gives the guidance for operators, and having it in place at the pilot site is a P3 deliverable and exit criterion (§7.1).

### 7.3 Adoption metrics

Leading metrics, tracked from G0 (baselines measured where not yet known):

| Metric | Baseline | Target | Owner |
|---|---|---|---|
| Hook PRs merged upstream (pace set by the NeMo RL maintainers) | 0 (G0) | Hooks 4, 5 (and 1 under path N) merged before P0 exit | Project lead, with the upstream contact for NeMo RL |
| Trainers passing conformance kit v0 (NeMo RL; verl-omni under path V) | 0 | 1 by P0 exit, 2 by P1b exit under path V | Project lead |
| Weekly runs of pilot users reporting through the run contract | 0 | Every pilot-user run by P2 exit | Project lead |
| Time for a pilot user to go from spec to first gated stage | Measured on the first P2 pipeline | Halved by the second adopter | Workflow maintainer (P2) |
| Time to add an algorithm in the research profile | Measured on DiffusionNFT in P1a | ≤ 1 week for an algorithm that fits existing stages | Algorithms maintainer |

Lagging metrics:

- A second diffusion family is onboarded in at most 3 person-weeks.
- A published algorithm that fits existing stages is added in one file of at most about 200 lines.
- At least two adopters run their pipelines on the workflow layer within two quarters of P2.
- RL-stage goodput of framework-managed runs is at least 90% (07).

## 8. One-way doors and deferrals

- **One-way:** the persisted step-field layout (frozen only after a second real family, an `ACTION` family, is built; until then fields are versioned and marked experimental); the artifact manifest and hash format (per-file hashes in P2, chunked Merkle roots from P3, 05 §13.3); the contract transport format (record layout, attempt- and epoch-scoped paths, Ed25519 signature envelope), which every trainer adapter depends on; R's run-record envelope, shared by E, G, and health-service integration (owner sections are versioned separately); upstream NeMo RL APIs (kept small by staging, and with no version packing, so weight versions remain plain integers upstream); the user-facing workflow YAML; and the metadata-store product (must be linearizable, 03).
- **Reversible:** workflow-engine backend, scheduler plug-ins, placement policy, credit assigners, partner caches, the P2 lineage storage (tables in the workflow database, migrated to Marquez and Iceberg in P3), transport implementations behind the interface.
- **Deferred:** stage runner (retired as an ask), variable `dp_size`, plug-in index service (a digest allow-list is used until P3), composition, joint training, the post-refit weight digest (hook 6b).

## 9. Sub-designs

| File | Covers | Status |
|---|---|---|
| `01-contracts-and-algorithms.md` | NeMo RL baseline; decision step; step fields; `ModelAdapter` and `StepForward` placement in B; primitives and capability records; algorithm plug-ins and `rollout_role`; loss kernel; research profile | Proposed |
| `02-run-contract-and-checkpoints.md` | Run contract and contract transport; provenance; checkpoint cuts; attempt- and epoch-scoped storage; fallback; version lineage; refit verification | Proposed |
| `03-control-plane.md` | Store requirements; leases; fenced compute and hold; supervisor; placement broker; scheduler plug-ins; intents; health-service integration; formal verification | Direction — detailed review after G0 (§1–§3, the P2 subset, normative) |
| `04-workflow-and-composition.md` | Workflow spec, IR, thin runner and backends; gate semantics and pins; `ControlAPI` decision flow; composition; alternating training; joint-training direction | Proposed (P2); P3+ parts Direction — detailed review after G0 |
| `05-evaluation-and-lineage.md` | Gate statistics; evaluator independence; reward-hacking detection; human-feedback design; per-family evaluators; contamination; lineage model and ledger; reproducibility | Proposed |
| `06-security.md` | Threat model; MVP baseline; identities and secrets; plug-in trust; sandbox; egress and residency; consent; safety-critical workloads | Proposed (MVP baseline); P3+ parts Direction — detailed review after G0 |
| `07-operations.md` | Service inventory and operator roles; SLOs and SLIs; failure-mode analysis; chaos tests; runbooks; budgets; upgrades and skew | Direction — detailed review after G0 (P2 operations content normative) |
| `08-families-execution-and-performance.md` | Execution modes including fused vectorized and hybrid; family notes; payload and performance model; placement constraints; embodied specifics; numeric targets | Proposed |
| `open-issues.md` | Deferred review findings with phase gate and owner role; calibration defaults | Maintained during the freeze |
| Round-3 review record (not published) | Round-3 review of revision 8: verdicts, MVP/Deferred classification, decisions R9-1 to R9-10 | Record |

## 10. Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Adopt verl + verl-omni for all non-token families | verl-omni is a plug-in package on pinned verl, the packaging this design uses, with 25+ pipelines (SD3.5, Qwen-Image, Wan2.2, LTX2, Qwen3-TTS, Qwen3-Omni, BAGEL) and FlowGRPO, DiffusionNFT, Diffusion-DPO, DanceGRPO, MixGRPO, GRPO-Guard | A second training stack beside NeMo RL; no workflow, lineage, or gate layer. Evaluated in G0 with a symmetric cost table rather than dismissed |
| NeMo-Run plus gate scripts plus a manifest library | Cheapest option: no new services; teams keep their scripts; manifests and gates come from packages A and E-stats | No requeue safety (NeMo-Run hard-codes `--requeue`), no enforced GPU-hour cap, no identity or separation of duties, no consumption ledger, so impact queries and restricted-data checks cannot be proven. P2-min now equals this option plus gates and lineage (and the cap, identity, and `--no-requeue`) |
| Separate stacks with shared libraries only | Least new infrastructure | No common run contract or lineage unless packages A, B, and E exist, which this design provides in either G0 outcome |
| Purpose-built general workflow engine | Full control | Re-implements durable execution; replaced by a backend contract (04) |
| Stage-runner refactor of `SingleController` first | Uniform algorithm stages | Blocks the 6,685-line, fast-changing controller; retired as an ask (§5) |
| **This design: contracts and libraries, narrow staged hooks, optional packages gated on users** | Useful in either G0 outcome; small upstream footprint | Requires G0 discipline and pilot users |

**P2-min is the cheapest option plus gates and lineage.** It is NeMo-Run-style submission with `--no-requeue` and operator `retry_stage`, metric gates from E-stats, and manifests from A, plus only what the cheapest option cannot provide and the LLM pilot user's exit criteria require: the GPU-hour cap with a QOS backstop, `ControlAPI` identity and separation of duties, per-team accounts and runners, policy-tag checks, and a Parquet ledger. Automatic requeue safety (`issue_epoch`, own `sbatch --hold`, janitor) is P2-full, built only on the pilot user's written need. Lineage services, the linearizable store, the supervisor, and health-service changes are P3. If the LLM pilot user does not need the P2 exit criteria, the cheapest option is the fallback and P2 is not built.

## 11. Open questions

1. Who are the pilot users, and is there an embodied pilot user who takes P6a instead of P1b (G0)?
2. Will the NeMo RL maintainers accept the staged hooks in §5, at what review pace, and who tracks `grpo_train_sync` parity?
3. Path N or path V after the feasibility run and cost tables, and is 2,000 GPU-hours the right G0 cap?
4. Which linearizable store backs the control plane (03), and who operates it at each site (P3 entry gate)?
5. Which service node hosts the per-team P2 runners on each cluster, and can compute nodes reach `gatedtrain-runtime-svc` on every cluster?
6. Which workflow engine do adopters already operate (P9)?
7. Which identity provider backs the `ControlAPI` (06)?
8. Which sandbox technology is approved per cluster type (P8)?
9. Which labeling system and consent process does the human-feedback connector use (P7)?
10. Which rollout engine serves VLA flow heads (P6b)?
11. Is there a pilot user for composition after the pilot (P10)?
12. Which artifact and lineage stores (Iceberg, lakeFS, an OpenLineage backend such as Marquez or DataHub) are already operated? They are P3 dependencies; P2 uses tables in the workflow database (05).

## 12. Finding disposition index

Every finding of the round-1 and round-2 reviews and the MVP findings of the round-3 review (review records not published) is resolved in one primary place; each document ends with disposition tables for its findings.

**Round 1**

| Findings | Primary location |
|---|---|
| EM-1–4, EM-6, EM-8, DA-1, DA-2, DA-4, DA-5, NRL-6, NRL-7, ARCH-2, ARCH-10 | This overview (§3–§8, §10) |
| ARCH-3, ARCH-4, ARCH-7, NRL-1, NRL-3–5, RES-1–5, RES-8, PERF-3, PERF-4, EMB-2, EMB-3, EMB-9 (fields) | 01 |
| ARCH-1, ARCH-6, NRL-2, DS-1, DS-5, DS-6, PERF-1, FMEA-3, FMEA-4, FMEA-5, FMEA-8 | 02 |
| ARCH-8, DS-2–4, DS-8, DS-9, OPS-1–3, FMEA-1, FMEA-2, DA-3, RT-3 (broker), EMB-6 (scheduling), FMEA-9 (health-service contract) | 03 |
| EM-5, ARCH-5, DS-7, DA-6, EVAL-4, RES-7, EMB-5, PERF-9 | 04 |
| EVAL-1–3, EVAL-5–8, RES-6, RES-9, LIN-1–9 | 05 |
| GOV-1–8, RT-1–8, EMB-9 (safety shield) | 06 |
| ARCH-9, OPS-4–7, FMEA-6, FMEA-7, FMEA-9 (SLOs, chaos tests, runbooks) | 07 |
| EM-7, PERF-2, PERF-5–8, PERF-10, EMB-1, EMB-4, EMB-6 (placement), EMB-7, EMB-8 | 08 |

**Round 2**

| Findings | Primary location |
|---|---|
| EM2-1–EM2-8, DA2-1–DA2-6, ARCH2-1, ARCH2-9, NRL2-7 | This overview (round-2 disposition below) |
| ARCH2-4 (protocol placement), ARCH2-5, NRL2-1, NRL2-3, NRL2-5, NRL2-6, RES2-1–RES2-4, RES2-8, PERF2-1 (roles), EMB2-3, EMB2-5 (fields), EMB2-7 (sparse `final_observation`), EMB2-8, GOV2-7 (research signing), RT2-4 and GOV2-2 (research-profile parts) | 01 |
| ARCH2-2, ARCH2-6, DS2-A, DS2-5, NRL2-2, NRL2-4, FMEA2-1, FMEA2-3, OPS2-5, FMEA2-9 (checkpoint age), PERF2-4, PERF2-5 (mechanisms), RT2-2 | 02 |
| ARCH2-3, DS2-B, DS2-C, DS2-6, DS2-7, DS2-8, FMEA2-2, FMEA2-4, FMEA2-5, FMEA2-6, FMEA2-8, OPS2-1–OPS2-4, OPS2-6, OPS2-8, OPS2-9, OPS2-10, RT2-1, RT2-3, RT2-5, RT2-6, GOV2-5, EMB2-4 (bench lease) | 03 |
| DS2-D, ARCH2-8, RES2-5, RES2-6, RES2-7, FMEA2-7, EVAL2-6 (04 parts) | 04 |
| EVAL2-1–EVAL2-8, LIN2-1–LIN2-8, ARCH2-7 | 05 |
| RT2-4, RT2-7, RT2-8, GOV2-1–GOV2-4, GOV2-6, GOV2-8, EMB2-2, EMB2-4 (watchdog and interlock) | 06 |
| OPS2-7, OPS2-11, FMEA2-9 (alarm dependency), R8-19 | 07 |
| PERF2-1 (numbers), PERF2-2–PERF2-6 (targets and arithmetic), EMB2-1, EMB2-5 (hybrid mode), EMB2-6, EMB2-7 (rendering placement) | 08 |

**Round 3** (round-3 review record, not published; MVP findings only, Deferred findings in `open-issues.md`)

| Findings | Primary location |
|---|---|
| EM3-1–EM3-7, DA3-1–DA3-4, ARCH3-1–ARCH3-4, ARCH3-7, ARCH3-8 | This overview (round-3 disposition below) |
| NRL3-1, NRL3-2, NRL3-6, RES3-1, RES3-3–RES3-5 (01); NRL3-3–NRL3-5, LIN3-2, FMEA3-6 (02) | 01, 02 |
| DS3-3, FMEA3-4 (03); DS3-4, DS3-5 (04) | 03, 04 |
| EVAL3-1 (counting rule and validator), EVAL3-2–EVAL3-4, LIN3-1, LIN3-3 | 05 |
| RT3-1–RT3-6, GOV3-1, GOV3-2 (MVP part), GOV3-4–GOV3-6 | 06 |
| PERF3-1, PERF3-5 | 08 |
| All Deferred IDs | `open-issues.md` |

## 13. Glossary

- **Decision step:** the unit at which a policy acts and, where defined, has a log-probability (token, denoise step, action).
- **Decision segment:** a contiguous run of steps of one type inside a row; field `decision_segment_id`, distinct from NeMo RL's physical `segment_index` (01).
- **Run contract:** the records a run writes and the verbs it accepts, through which every outer component observes and controls it (02).
- **Contract transport:** the interface (in A, with the contract emitter) that carries the run contract, with a POSIX-directory and an object-store (conditional put; optional dependency) implementation, both in A and used by R's run adapter; paths are scoped per attempt and epoch, and verbs and pointers are signed by the adapter (02).
- **Run adapter:** the out-of-process component in `gatedtrain-runtime-agent` that holds the lease, fences, signs, and gates commits for one run; the trainer never sees leases (02, 03).
- **Cut:** a consistent checkpoint, committed only when its manifest is recorded in the store under the current controller epoch (02).
- **Controller epoch:** a store-issued number for one attempt that fences that attempt's side effects; every attempt gets a new one at launch (03). Lineage identity of a weight version is `(controller_epoch, counter)`, where the counter is NeMo RL's integer version (02).
- **`issue_epoch`:** the endpoint in `gatedtrain-runtime-svc` that a start script calls first to obtain a controller epoch; it refuses stale or post-cancel requeues, and the script then exits with code 79 (P2-full, D19; P3).
- **Fenced compute / fenced hold:** after lease loss, the run may keep training and refitting for `fenced_compute_max_s` without committing, then is held in place with its allocation for `fenced_hold_max_s` (entered with `pause_and_checkpoint(then=hold)`, released with `pause_and_checkpoint(then=continue)`; default in 03 §4.3), then stopped (D7, 03).
- **`rollout_role`:** the `AlgorithmSpec` field naming which weights are synced to rollout (default `policy`; DiffusionNFT uses its EMA `old` role) (01).
- **Precision (lineage):** the granularity of a consumption edge: `row`, `step_range`, `iteration`, or `stage` (05).
- **`StepForward`:** the per-step forward and primitive interface of a model family. **`ModelAdapter`:** construction, sharding, name maps, export. Both are torch-typed and live in package B (01).
- **Package R:** `gatedtrain-runtime-agent` (in-allocation run adapter, A's contract transport, per-attempt signing key) and `gatedtrain-runtime-svc` (store, broker, `ControlAPI`, `issue_epoch`, `SchedulerPlugin` and `IntentPolicy`, run-record envelope, persisted `WorkloadSpec`); below E and G (§4).
- **C-up / C-ext:** C-up is the set of upstream NeMo RL PRs (no import of A); C-ext (`nemo-rl-gatedtrain`) is the framework-side adapters, wrappers, and emitter wiring for NeMo RL (§4).
- **Gate:** a typed statistical test over evaluator results with declared error rates (05).
- **Consumption ledger:** a durable per-sample record of what each optimizer step consumed and why rows were dropped (05).
- **Path N / path V:** the G0 outcome for non-token families: build on NeMo RL, or adopt verl-omni.

## 14. Sources

- NeMo RL local clone at `4f39fd8`; files cited in each sub-design.
- verl-omni local clone: `pyproject.toml` (pinned verl dependency), `verl_omni/pipelines/` (25+ pipelines), about 74,600 lines of Python; verl `verl/__init__.py` (`verl.plugins` entry points).
- flow_grpo, dancegrpo, diffusionnft, nemo-evaluator, nemo-curator, nemo-run, and nvidia-resiliency-ext local clones; web sources for workflow engines as listed in 04.
- Panel reviews: the round-1, round-2, and round-3 review records and the history of earlier rounds (maintainers' working material, not published).

## Round-1 disposition

| ID | Section | Resolution |
|---|---|---|
| EM-1 | §3 | G0 requires LLM and diffusion (or embodied) pilot users with written success criteria; otherwise the project stops after P0. |
| EM-2 | §3, §5, §5.1 | G0 records the NeMo RL maintainers' response to the hook list (no review budget is assumed); without acceptance hooks land as a plug-in against a pinned NeMo RL. |
| EM-3 | §7.1, §7.2 | The MVP is G0, P0, P1a, P1b (or P6a), and P2 over two quarters (rev 8); later phases need a pilot user. |
| EM-4 | §3, §3.1 | G0 runs a verl-omni feasibility comparison and symmetric cost tables before any family code. |
| EM-6 | §7.3 | Leading and lagging adoption metrics with baselines and owners (rev 8). |
| EM-8 | §7.2 | Effort per phase and per path, re-estimated in round 2. |
| DA-1 | §1, §4, §3.3 | Workflow, control plane, embodied, and composition packages are built only for a pilot user; kill criteria are reviewed each phase. |
| DA-2 | §10, §14 | verl-omni is described as a plug-in package on pinned verl with 25+ pipelines and is evaluated in G0. |
| DA-4 | §3, §5.1 | Every external dependency has a dated prerequisite or a stated fallback (plug-in against pinned NeMo RL, dedicated partitions). |
| DA-5 | §1, §4 | Contracts and libraries ship first; family-specific work stays in optional packages. |
| NRL-6 | §4 | Workflow (E), runtime (R), control plane (G), composition (F), and embodied (H) are separate packages; EMA roles are algorithm roles in B and C-ext (01). |
| NRL-7 | §7.1 P7 | The data-plane DPO trainer has an owning maintainer and a deprecation plan for `dpo.py`. |
| ARCH-2 | §5, §7.1 | Flow-GRPO (P1b) needs neither a stage runner nor the workflow layer; P1b runs in parallel with P2. |
| ARCH-10 | §8 | One-way doors are listed, with the step-field layout frozen only after an `ACTION` family. |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| EM2-1 | §7.1, §7.2 | The MVP is two quarters, with P1a parallel to P0 and P1b parallel to P2 after P0. |
| EM2-2 | §3 | G0 runs a 1–2 seed feasibility comparison at a fixed sample count under a declared GPU-hour cap; full TOST moves to P1b. |
| EM2-3 | §6 D18, §7.1 P2 | P2 is thinned: lineage as tables in the workflow database, ledger as Parquet with DuckDB, manifests with per-file hashes; Marquez, Iceberg, Merkle manifests, B15, and release checks moved to P3. |
| EM2-4 | §3, §5.1 | G0 records the NeMo RL maintainers' response and the dated site prerequisites (cluster and health service); without them P1 and P2 run on dedicated partitions. |
| EM2-5 | §3, §5, §7.2 | Only hooks 1, 4, 5 start before G0, and P0 is re-estimated at 24–30 engineer-weeks over 8–10 weeks. |
| EM2-6 | §7.2 | Effort is given separately for path N and path V. |
| EM2-7 | Header, §9 | 03, 07, and the P3+ parts of 04 and 06 carry "Direction — detailed review after G0". |
| EM2-8 | §7.3 | Leading metrics with baselines, targets, and owners are added beside the lagging metrics. |
| DA2-1 | §5 | Hooks are staged; only hooks 1, 4, 5 land before G0, and hooks without a path-V user wait for their phase. |
| DA2-2 | §10 | The cheapest option is listed, and P2 is rescoped to that option plus only what its exit criteria require. |
| DA2-3 | §3.1 | The path rule compares written cost tables for both paths symmetrically, together with user requirements. |
| DA2-4 | §3, §5.1, §11 | An identified store operator is a G0 dependency for P3, and P3 does not start without one. |
| DA2-5 | Header, §9 | Detailed review of P3+ material is deferred until after G0 by document status. |
| DA2-6 | §3.2 | Under path V, package A serves gates, lineage, the run contract, and the workflow layer; nothing consumes `StepForward`. |
| ARCH2-1 | §4, §5 | Fencing, signing, MAC checks, and version handling stay out of NeMo RL; the upstream list is narrow, staged, and explicit about retired asks. |
| ARCH2-9 | §4 | E depends on E-stats, H depends on C-ext only optionally, and A is pure Python with torch-typed protocols moved to B. |
| NRL2-7 | §3, §5, §5.1 | The NeMo RL maintainers' response to the hook list is recorded in G0, and only hooks 1, 4, 5 land before G0 closes. |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| EM3-1 | §3, §7.1 | G0 closes on MVP prerequisites only; health-service, P3 store-operator, and other P3 prerequisites form the P3 entry gate. |
| EM3-2 | §7.2 | P2 is re-estimated: P2-min 12–16 engineer-weeks over 6–8 weeks, P2-full 24–30 over 8–10 weeks. |
| EM3-3 | §3, §5, §5.1 | The planning placeholder for upstream review pace is 3–4 hook PRs a month (not a commitment of the NeMo RL maintainers), and hooks 8a-1 and 8d may ship first as C-ext shims. |
| EM3-4 | §3, §5.1 | A site KMS/PKI for signing keys is a G0 prerequisite. |
| EM3-5 | §7.1, §7.2 | P3 is a separate scope decision; the end state without P3 is P2 plus the health service as the site runs it. |
| EM3-6 | §3.1, §7.1 | The P1b GPU-hour budget is extrapolated from the G0 run and recorded in the cost tables. |
| EM3-7 | Header, §7.1 | Scope is staged in increments, and the document set is frozen until G0 closes. |
| DA3-1 | §6 D8, D18, D19, §7.1, §10 | P2-min (`--no-requeue`, operator `retry_stage`) is the default; P2-full is built only on the LLM pilot user's written need at G0. |
| DA3-2 | §5 | Hook 1 is marked path-N only. |
| DA3-3 | §5.1, §6 D18 | The operator of the P2 deployment runs the P2 store in both variants. |
| DA3-4 | Header | Revision 9 is the closing revision; later changes come only through G0 outcomes or `open-issues.md`. Amended 2026-11-11 (maintainer decision, R10-12): design defects found during implementation are fixed in the design with a revision bump and a row in the change table ("Revision 10 changes"). |
| ARCH3-1 | §4 | R splits into `gatedtrain-runtime-agent` (in allocation) and `gatedtrain-runtime-svc` (central); store credentials never enter jobs. |
| ARCH3-2 | §4 | R defines `SchedulerPlugin` and `IntentPolicy` with P2 defaults and one Slurm plug-in; G only registers more. |
| ARCH3-3 | §4, §8 | The run record is an R-owned envelope plus namespaced, separately versioned owner sections. |
| ARCH3-4 | §4 | The rendered `WorkloadSpec` is persisted as an A-format artifact at first submission. |
| ARCH3-7 | §4, §6 D19 | The job-start handshake (`issue_epoch`, exit code 79) is part of the run contract in A; the P2 restore's reliance on NeMo RL's checkpoint listing is covered by a conformance test and removed by hook 8b (02); restore uses copy-on-write copies (06). |
| ARCH3-8 | §4 | Dependency rule 2 reads "signing is outside the trainer; verification is in A's library"; the dispatcher's `loop_state` input is in 04. |

All Deferred round-3 IDs are recorded, with phase gate and owner role, in `open-issues.md`.

## Revision 10 changes

Approved by the project lead on 2026-11-11. Sources G-x are the gaps in the maintainers' prototype gap record (not published); G-22 was found in the revision-10 review.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R10-1 | 05 §11.2, §12.1, §12.2 | `ConsumptionRecord.precision` gains `iteration`; in iteration records `data_ref.row`, `sample_id`, `chunk`, `rewards`, and `seed_inputs` are null and `outcome` is `consumed`; row keys are required except in iteration records | §11.2 allowed `iteration` edges, but the schema had no record shape for them | G-1 |
| R10-2 | 04 §3 (example), §4.1; 05 §3.7 | A gate's pre-registration is an artifact of type `registration`, bound to the gate's `registration` input role by content hash; a blocking gate without one is rejected at validation | No spec field referenced a registration | G-2 |
| R10-3 | 05 §3.1, §3.6; 04 §4.2 rule 1 | Achieved power at the registered planning effect with the observed SD of paired differences (ddof = 1); superiority, non-inferiority, and TOST power and n formulas; `underpowered` on low power, n below plan, or power not computable; `sd_zero` flag | Observed versus planned σ was undefined; no equivalence power formula; σ̂_d = 0 undefined | G-3, G-12, G-13 |
| R10-4 | 05 §3.8; 04 §4.2 rule 1 | Statuses PASS, FAIL, INCONCLUSIVE, MISSING defined; `combine` precedence for `and`/`and_mean` and `or`; only PASS opens a blocking gate; INCONCLUSIVE and MISSING follow `on_fail` | MISSING inside `and` had no defined result | G-9 |
| R10-5 | 05 §6.1, §3.7 | Scorer validity: evaluator hash covers task, prompt template, scorer, and container; hash mismatch is MISSING; scorer error rate measured on baseline outputs before registration; floor, ceiling, or excess error refused; shared parser with a reward is T0; evaluator hash resolved when the registration is signed | A strict scorer scored a format, not the training effect (baseline 0.0); a registration could carry a null evaluator hash | G-21, G-6 |
| R10-6 | 04 §4.3 | Gate evaluation uses the registration's `eval_seeds`, not seeds derived from the step key; raw results stay keyed by the step key | 04 and 05 §3.3 disagreed on evaluation seeds | G-22 (review finding) |
| R10-7 | 04 §3, §4.2; 05 §3.2, §3.7 | The registration is authoritative for α, power, margin or bar, MDE, sampler, and evaluator; a differing spec value is rejected; spec statistical fields are optional; α must be in the registration or come from a family; 05 §3.2 defaults are `plan()` drafting values | Spec and registration could both set the same values; 04 required α while 05 gave a default | G-4, G-8 |
| R10-8 | 05 §3.4, §3.7; 04 §4.4 | Missingness check: `pairing.max_unpaired` (default 0), complete pairs only, per-arm missing counts and reasons, MISSING above the bound | No tolerance for unpaired items; 04's reference had no target | G-18 |
| R10-9 | 05 §3.7 | A registration amendment after any candidate-arm result was observed marks the gate `post_hoc: true`; it cannot PASS (INCONCLUSIVE, reason `post_hoc`); amendments reference the prior registration by hash. Refined by R11-1 | Amendments were unrestricted after unblinding | Review finding |
| R10-10 | 05 §3.6, §3.7 | `min_paired` (default 10, as `gate_runs`); fewer complete pairs or a zero-width interval is INCONCLUSIVE, never PASS | A gate could pass on a handful of pairs or a degenerate interval | Review finding |
| R10-11 | Implementation plan (not published) §2.2 rule 13, §4 G0 rule; overview §3 | Controlled prototype exception before G0: `prototype`/`unpromotable` outputs that count toward nothing; maintainer-set GPU-hour cap and host scope in a decision record; GPU work may be scheduled within them without further approval; interim gap decisions by the maintainers, each written back to the design. First use: the walking-skeleton prototype's decision record (not published) | The walking-skeleton prototype ran outside the G0 rule without a recorded exception | Maintainer decision 2026-11-09, 2026-11-11 |
| R10-12 | Header, DA3-4 | Design defects found during implementation are fixed in the design with a revision bump and a row in this table | Revision 9 allowed changes only through G0 outcomes or `open-issues.md` | Maintainer decision 2026-11-11 |

## Revision 11 changes

Approved by the project lead on 2026-11-11. Sources P-n, V-n, and MM-n are items of the revision-10 triage; C-n are the project lead's choices on its open items.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R11-1 | 05 §3.7 | `post_hoc` is triggered only by results of the candidate checkpoint the gate judges (a raw-results artifact created before the amended registration counts as observed), not by other checkpoints or other loop candidates; a `post_hoc` clause whose pass rule holds is INCONCLUSIVE (`post_hoc`), other statuses unchanged. Refines R10-9 | R10-9 blocked every later candidate of a loop gate and changed FAIL and MISSING to INCONCLUSIVE | P1, P2 |
| R11-2 | 05 §3.4 | Missingness defined: expected items (item set × K, × training seeds when that is the unit) per arm; missing means absent, evaluator error, or out of range, with reason assigned by the gate library; `max_unpaired` count or fraction over all arms; an item missing in both arms counts once per arm. Refines R10-8 | "Unpaired item" and the bound's denominator were undefined | V3, V4 |
| R11-3 | 05 §3.6 | `min_paired` counts complete pairs in units of analysis, the n of the power formulas; a complete pair is an item with at least one repeat in both arms, averaging the repeats present in both | n was not tied to the unit; partial repeats had no rule | V5, C1 |
| R11-4 | 05 §3.7 | Evaluator manifest hashes are resolved when the registration is written; validation refuses a null or unresolved hash whether or not the registration is signed. Refines R10-5 | An unsigned registration escaped the check | V6 |
| R11-5 | 05 §3.8 | p-values come from the interval procedure by test–interval duality; TOST reports the larger one-sided p; the decision comes from the interval; none for a zero-width interval or no pairs | `evaluate` returned p-values with no definition | V7 |
| R11-6 | 05 §3.7; 04 §4.2 | The registration is authoritative for every clause field it holds and for sampler and evaluator; a differing mapped spec field is rejected. Refines R10-7 | R10-7 listed only some fields | V8 |
| R11-7 | 05 §3.8 | Records with a repeat outside range(K) or a seed other than `eval_seeds[repeat]` make the clause MISSING; fewer than K records are missing items | Malformed records had no status | V10 |
| R11-8 | 05 §3.6 | TOST n formula writes z_{1−β/2}; a planning effect with \|θ\| ≥ δ is refused by `plan()` and rejected at validation | Ambiguous quantile; no finite n for \|θ\| ≥ δ | V11 |
| R11-9 | 05 §3.8 | Per-item record in package A (`item`, `repeat`, `seed`, `score`, `error`, `sub_metrics`, `output_ref`; per arm the model, evaluator, and sampler hashes); score versus error declared by the evaluator manifest; NeMo Evaluator records converted | `evaluate` had no record shape | MM-1 |
| R11-10 | 05 §3.1 | Clauses state `direction`; lower-is-better metrics are negated, all margins and effects are in oriented units, Δ > 0 favours the candidate; report shows natural-unit difference beside the oriented one | Orientation relied on an upstream helper | MM-2 |
| R11-11 | 05 §6.1 | The evaluator manifest declares the score range; out-of-range scores are missing items (`out_of_range`); floor and ceiling are its bounds | Floor and ceiling were undefined | MM-3 |
| R11-12 | 05 §11.3 | A `stage` edge returns every checkpoint the consuming stage generated as possibly consumed, precision `stage` | Stage-precision edges have no records or rollups | MM-5 |
| R11-13 | 05 §11, §11.1 | Checkpoint `base_ref` for adapter-only checkpoints; an unmerged adapter that is evaluated, exported, or deployed gets a `lora_compose` activity and a model entity covering base and adapter hashes | Unmerged adapters had no deployable identity | MM-7 |
| R11-14 | 05 §3.3, §11; 02 §2.4 | `model_manifest_hash` over deployable components (including normalizer statistics and `base_ref`) names a checkpoint for gates, exports, and deployments; the cut manifest records it; evaluation records name it and a mismatch with the pinned checkpoint is MISSING | The cut manifest hash mixed resume state with the deployable model | MM-8, MM-9 |
| R11-15 | 08 §2.4 item 1 | M5 adapters write a cut manifest only for a checkpoint complete under a per-library-version completeness rule; conformance test with an interrupted write | A partial library checkpoint could be committed | MM-14 |
| R11-16 | 05 §12.4, §13.2 | Superseded and orphaned results: record counts with precision for row-level questions, step ranges per segment and shard for membership questions | "Both counts" was undefined for rollup-only answers | C2 |
| R11-17 | 05 §12.1; 08 §6.1 | Environment-based RL without a dataset materializes environment parameters as a `DatasetSnapshot`, one shard per `env_params_version` | Consumption records had no anchor without a dataset | C3 |
| R11-18 | 05 §11.2, §12.2 | `ConsumptionRecord.derivation` (`observed` or `reconstructed`); reconstructed records are limited to `iteration` precision unless the adapter passes a replay conformance test; answers show the derivation | Reconstructed provenance could claim row precision unverified | C4 |
| R11-19 | 05 §1 E2, §2, §3.1, §3.3–§3.8, §4.1, §5, §6.2, §7, §10.1, round-1 disposition; 04 §3 and §10 examples, §4.2; 08 §6.1; implementation plan §5 (report generator); verification plan §5.2 | The analysis unit "prompt" is renamed `item`, defined once in 05 §3.3 per family; `unit: item`, `samples_per_item`, item-clustered intervals, 04 `cluster_by: item`, gate input role `items`, evaluation-set artifact type `item_set` (training prompt datasets keep `prompt_dataset`); the registration's prompt-set hash becomes `eval_distribution` (item-set hash, plus `env_version` and `env_params_version` for environments). Literal prompts, prompt templates, and training prompts keep "prompt" | Speech, control, and VLA evaluation units are not prompts | C5 |
| R11-20 | 05 §6.1 | A learned continuous scorer's validity is its agreement rate with human pairwise preferences on baseline outputs (§8 process), with a registered lower bound; below it the deciding metric is refused; a scorer that is a training reward remains T0 | Learned scorers have no error rate | C6 |
| R11-21 | 05 §3.1, §3.4, §3.6 | Ratio-of-sums metrics (for example corpus WER) may decide: records carry numerator and denominator; Δ is the difference of arm ratios; item-clustered bootstrap of the ratio difference; power from delta-method per-item contributions | Corpus metrics are not means of per-item scores | C7 |
| R11-22 | Header; implementation plan status line | Revision 11; the implementation plan names design revision 11 | — | Maintainer decision 2026-11-11 |

## Revision 12 changes

Approved by the project lead on 2026-11-11 (multimodal prototype decision record, not published; constraint clauses and blocking judges).

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R12-1 | 05 §3.1, §3.7, §3.9 (new); 04 §4.2 | Constraint clause: a threshold or non-inferiority clause with `constraint_set`; metric is the per-item violation indicator v_i (any repeat violates), `direction: lower_is_better`; constraint texts (numbered), per-constraint checker (programmatic tool or judge with model ID and version, prompt template, decoding, parser), and canary set are in the registration and inside the evaluator manifest (scorer) hash; programmatic checkers preferred; violated constraint IDs and judge rationale in `sub_metrics`, never deciding; unparseable judge response is an evaluator error | Generated outputs could not be gated on registered constraints | Maintainer decision 2026-11-11 |
| R12-2 | 05 §3.9 | Threshold form: exact (Clopper-Pearson by default) upper (1−α) bound of the violation rate below `bar`, binomial power and n; report gives count, n, and exact bound; zero observed violations do not mean rate 0 (bound 1 − α^(1/n)) and are not a zero-width interval. Non-inferiority form: upper bound of the rate difference below δ. Combined with quality clauses by `and`; a constraint FAIL fails the gate | Hypothesis forms and the zero-violation case were undefined | Maintainer decision 2026-11-11 |
| R12-3 | 05 §1 E5, §6.1, §3.8; 04 §4.2 rule 2; 06 §4 B26, §12.1; overview §7.1 (P2 row) | For constraint clauses only, an LLM or VLM judge may be the sole blocking evaluator when (a) its miss and false-alarm rates against §8 human labels on baseline (and, when available, earlier different-checkpoint candidate) outputs are stated with upper bounds and the miss-rate bound (optionally false-alarm bound) is met, (b) it is T2 or higher and of a backbone family different from the policy and every training reward, (c) registered injection canaries are in every judged batch and a missed canary is MISSING, (d) decoding is deterministic, responses are cached in `JudgeResponseCache`, model ID and date are pinned, and a scorer-hash mismatch is MISSING, (e) an external-API judge has 06 egress approval and a pinnable version, else it only supports. Quality clauses keep the non-LLM rule | Programmatic checks cannot cover every natural-language constraint; human evaluation per item is too costly | Maintainer decision 2026-11-11 |
| R12-4 | 04 §4.2 | No new gate type: `gate.threshold` and `gate.noninferiority` take an optional `constraint_set` (mapped to the clause field, authoritative in the registration); a constraint gate blocks beside the quality gate | The hypothesis forms already exist; only the metric and the evaluator rule differ | Maintainer decision 2026-11-11 |
| R12-5 | 06 §9; implementation plan task cards (05 §6.1 reading list, validator reject list); verification plan §4.4 SEC-B26, §4.5 STAT-06, STAT-12 (new), §8 VV-GATE-1 | Cross-references only: an external judge receiving candidate outputs is an `upload` destination (05 §6.1 (e)); B26 and the validator tests admit the qualified constraint-clause judge and reject an unqualified one | Keep 06 and the verification plan consistent with R12-3 | Maintainer decision 2026-11-11 |
| R12-6 | 08 §3.6 (new); 05 §11 | Deployable components per family (LLM/VLM; image and video diffusion; speech; control and VLA) listed in 08 §3.6, which 05 §11 cites for `model_manifest_hash` | 05 §11 cited a component list in 08 that did not exist (left from R11-14) | Revision-11 leftover |
| R12-7 | 05 §13.1 | `derivation` shares the 1-byte code with `precision`; record size unchanged (about 76 bytes) | R11-18 added `derivation` without sizing it | Revision-11 leftover |
| R12-8 | Header; implementation plan status line | Revision 12; the implementation plan names design revision 12 | — | Maintainer decision 2026-11-11 |

## Revision 13 changes

Approved by the project lead on 2026-11-11 (NI method and canary scope chosen by the maintainers; the rest are definitions the text left open). Found by the prototype's rev-12 test round (`tests/README.md` §11.17 in the `gatedtrain` framework repository).

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R13-1 | 05 §3.9 (non-inferiority form), §3.1 interval list, §3.4 table | Non-inferiority constraint clause: exact upper (1−α) Clopper-Pearson bound of c/n (candidate-only violations over complete pairs), `interval: clopper_pearson_discordant`, below δ; conservative, valid because (c − b)/n ≤ c/n; report b, c, n, difference, bound; positive width at c = 0, so not a zero-width case | "Exact McNemar mapped to non-inferiority" conditions on b + c and does not bound the difference for δ > 0; at zero violations in both arms it gave a zero-width interval | Maintainer decision 2026-11-11 |
| R13-2 | 05 §3.9 | Power and n of the non-inferiority constraint clause: binomial, as the threshold form, with δ for `bar` and planning rate of c/n = δ − planning effect (no new field) | Only the threshold form had a power rule | Follows from R13-1 |
| R13-3 | 05 §6.1 (a) | The judge's miss and false-alarm bounds are exact Clopper-Pearson upper (1−α) bounds at the constraint clause's α | The α and method of the validity bounds were unstated | Definition left open by R12-3 |
| R13-4 | 05 §6.1 (c), §3.8 | Per-batch injection canaries (and MISSING on a missed canary) apply to a judge that blocks a constraint clause alone; a supporting judge needs only 06 B26 conformance canaries | Unclear whether supporting judges also needed per-batch canaries | Maintainer decision 2026-11-11 |
| R13-6 | 05 §3.4 | A record without an evaluation seed is a missing item, reason `no_seed` (a seed other than the registered one stays a MISSING clause, §3.8) | The status of a seedless record was unstated | Prototype rev-11 test round |
| R13-7 | 05 §3.7 | An unresolved evaluator hash is anything other than `sha256:` plus 64 hex digits; the registration records `created_at` (UTC) so the `post_hoc` raw-results comparison is auditable | "Unresolved" was undefined, and the audit rule of R11 had no registration time to compare with | Prototype rev-11 test round |
| R13-8 | 05 §3.4 table, §3.6 | Ratio-of-sums interval is BCa with the item jackknife; num_i and den_i are sums over the repeats used in the analysis | Bootstrap variant and per-item aggregation were unstated (they differ when repeat counts differ) | Prototype rev-11 test round |
| R13-9 | 05 §6.1 (scorer validity) | The floor/ceiling refusal uses the pilot baseline mean stated in the registration | The registration held no baseline score to test | Prototype rev-11 test round |
| R13-10 | 05 §11.3 | A `stage` edge reports derivation as null; the impact closure continues across runs through `used` model edges with the coarsest precision on the path | Derivation of a `stage` answer and the cross-run closure were implicit | Prototype ext-collect test round |
| R13-5 | Header; implementation plan status line | Revision 13; the implementation plan names design revision 13 | — | — |

## Revision 14 changes

Definition the text left open, found by the prototype's ext-collect re-review (`docs/findings.md` in the `gatedtrain` framework repository, finding L-e).

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R14-1 | 05 §11.2 | An impact answer reached by several paths takes the finest of the per-path (coarsest-edge) precisions and lists every path | §11.2 defined the precision of one path only; the prototype downgraded a row-level "consumed" answer to "possibly consumed" when a stage path also reached it | Prototype ext-collect re-review |
| R14-2 | Header; implementation plan status line | Revision 14; the implementation plan names design revision 14 | — | — |

## Revision 15 changes

Definition the text left open, found while choosing the deciding metric of the multimodal prototype's SD1.5 area (multimodal prototype decision record, not published, area 3; maintainer decision 2026-11-12).

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R15-1 | 05 §6.1 | "Same programmatic tool" means the scoring engine (codec, OCR engine, parser, detector, simulator) and its version, not the wrapper library; a T3 metric from a different engine that measures a reward's construct is "construct-aligned" and the registration and report say so | T0 named "the same programmatic tool" without defining it; a WebP size metric beside a JPEG compressibility reward could be read as T0 (both through Pillow, same construct) or T3 (different codec, like a different OCR engine) | Multimodal prototype, area 3 |
| R15-2 | Header; implementation plan status line | Revision 15; the implementation plan names design revision 15 | — | — |

## Revision 16 changes

Definition the text left open, found while choosing the deciding metric of the multimodal prototype's Whisper area (multimodal prototype decision record, not published, area 4; maintainer decision 2026-11-13).

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R16-1 | 05 §6.1 | "Shares training data or labelers" covers scorers that learned from the training labels and human or judge panels drawn from their labelers; held-out references from the same annotation process, consumed by no training stage, leave a programmatic scorer at its own tier, with the shared convention disclosed | WER against held-out AMI transcripts beside SFT on AMI transcripts could be read as T0 (same labelers) or T3 (programmatic, held-out) | Multimodal prototype, area 4 |
| R16-2 | Header; implementation plan status line | Revision 16; the implementation plan names design revision 16 | — | — |

## Revision 17 changes

Design defect found while renaming the prototype package before its first public release (new package name with version 0.2.0; maintainer decision 2026-11-17).

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R17-1 | 05 §3.7 | A version-only amendment (only `analysis_code_version` changes, from a recognized pre-rename version to the renamed framework's current version; `amends`, `created_at` and `post_hoc` may differ, every other field equals the prior registration, checked mechanically) is not post hoc and the report marks it; any other amendment follows the `post_hoc` rule, and declaring one that crosses the rename not post hoc is refused | Package rename before the public release: every registration records the old analysis code version, the renamed code accepts only the new one, and under R10-9/R11-1 the amendment that only renames the version would have made every historical gate post hoc (PASS reported INCONCLUSIVE) without any change to its hypothesis | Maintainer decision 2026-11-17 |
| R17-2 | Header; implementation plan status line | Revision 17; the implementation plan names design revision 17 | — | — |
| R17-3 | 05 §3.7 | A version-only amendment may also change an evaluator manifest hash whose scorer changed only by the rename: per changed evaluator, the scorer code hash before the rename and the pre-rename scorer file(s) shipped next to the registration; checked mechanically (the evaluator identity with the old code hash is the prior manifest hash and with the current one the amended hash; the files hash to the old code hash; the files with the published old-to-new name map applied give the current code hash byte for byte); anything else follows the `post_hoc` rule | Package rename before the public release: the recipes' scorer scripts write the renamed arm schema id, so their code hash, and with it the evaluator manifest hash every registration records, changes without any change to how items are scored | Maintainer decision 2026-11-17 |

## Revision 18 changes

Design defect found while porting the TTS recipe's pre-registered gate (two blocking clauses on two metrics, a cluster bootstrap by speaker, percentile bounds) to gatedtrain: the registration held one clause decided by one evaluator, with an item bootstrap and BCa bounds only (maintainer decision 2026-11-17).

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R18-1 | 05 §3.1, §3.7 | A registration may list several blocking clauses in `clauses` instead of `clause`: each entry has the §3.1 fields plus a unique `name` and its deciding `evaluator` (one evaluator per clause); `sigma_pilot` maps clause names to pilot values; constraint clauses keep the single form; a registration with one `clause` is unchanged | A gate with a quality clause and a guard clause on another metric (TTS: Qwen3-ASR WER superiority and speaker-similarity non-inferiority) could not be registered | TTS recipe, 2026-11-17 |
| R18-2 | 05 §3.5 | `multiplicity: {plan: all_clauses_must_pass}`: intersection-union, each clause at its own α, no α split; status by the §3.8 `and` precedence (FAIL, then MISSING, then INCONCLUSIVE, else PASS), `post_hoc` and evaluator hash mismatch per clause first; per-clause power as registered, joint power of independent clauses (the product) stated in plan and report | §3.5 described AND gates, but no registration field or combination rule carried one | TTS recipe, 2026-11-17 |
| R18-3 | 05 §3.4, §3.6 | `cluster_key`: resample clusters of complete pairs defined by an item-set field (both arms together, clusters in sorted value order, ratio of the drawn clusters' sums, leave-one-cluster-out jackknife for BCa); an item without the field refuses the gate; fewer than two clusters is a zero-width interval; achieved power uses the observed design effect (cluster-robust variance over item variance, at least 1) | Items sharing a voice prompt are not independent; the item bootstrap understated the interval | TTS recipe, 2026-11-17 |
| R18-4 | 05 §3.4 | `bootstrap_variant: percentile` beside the BCa default: α and (1−α) quantiles of the replicates, p-value the share of replicates below the null; the pass rules stated in natural units for both directions | The pre-registered analysis fixed one-sided percentile bounds | TTS recipe, 2026-11-17 |
| R18-5 | 04 §4.2 | A gate over several clauses lists one evaluator per clause in clause order; `uses` names the first clause's type; a stated `test` field must equal every clause's value | The spec named one evaluator and one clause type per gate | TTS recipe, 2026-11-17 |
| R18-6 | 05 §3.7 | The report of a gate with several clauses gives each clause's fields with its name and evaluator, and the combined status | Reports carried one clause | TTS recipe, 2026-11-17 |
| R18-7 | Header; implementation plan status line | Revision 18; the implementation plan names design revision 18 | — | — |
| R18-8 | 05 §6.1 | Pre-run baseline (smoke) check: a score exactly on the finite bound at the metric's best end (lower bound if lower is better, upper bound if higher is better) cannot be the value that makes up half of the sample; the worst end counts as before; the share stays a share of the whole sample; at least a quarter of the sample (and at least two scores) must lie off the best bound; with several clauses every clause's evaluator is checked; a sample without scores on the best bound is judged as before | A lower-is-better WER with many exact zeros (14 of 20 TTS baseline scores) failed the check although the evaluator worked; only the first evaluator was checked | TTS recipe, 2026-11-17 (maintainer decision B) |

## Revision 19 changes

Design defect found while migrating the recipes to the renamed framework: the revision 17 rule exempted a version-only amendment from `post_hoc` without saying what happens when the prior registration is itself `post_hoc: true`, so an amendment that only renamed the version could declare `post_hoc: false` and turn an INCONCLUSIVE gate into a PASS.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R19-1 | 05 §3.7 | An amendment across the rename (version-only or not) keeps a prior `post_hoc: true`; declaring it false is refused | The version-only exemption must not clear a `post_hoc` the prior registration already carried | Recipe migration, 2026-11-18 |
| R19-2 | Header | Revision 19 | — | — |

## Revision 20 changes

Design gap found while planning multi-seed gates for the recipes: 05 §3.3 names the training seed as the unit of a recipe claim and §3.4 a two-stage bootstrap for it, and §4.2 tests per-seed means with a `t` test, but no section says how the runs of an arm are registered and matched, what the seed-level score is, which of the two analyses decides, or how power and the minimum sample count at the seed level; the prototype refused `unit: training_seed`. The reruns of 2026-11-18 showed why it matters: GRPO training is not bit-reproducible, and one recipe's baseline arm moved widely between training cuts.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R20-1 | 05 §3.3 | Training-seed unit defined: candidate arm = S runs differing only in the training seed, baseline = one fixed checkpoint or S_b runs; `training_seeds` (and `baseline_training_seeds`) registered before any run, runs matched by the launch record's training seed, an unregistered seed refused and a registered seed without a usable run makes the clause MISSING; seed-level score y_r over the items present in every run | §3.3 named the unit without defining arms, matching or the score; registering the seeds prevents leaving out runs after the fact | Roadmap 2026-11-19, workstream 4 |
| R20-2 | 05 §3.3, §3.4 | `seed_analysis`: `seed_t` (default; one-sample `t` on d_r with a fixed baseline, Welch's `t` with baseline runs, paired `t` with `seed_pairing: paired`) or `two_stage_bootstrap` (runs, then items within runs, 10,000 replicates, percentile bounds; paired runs are resampled as pairs sharing one item draw) | §3.4 and §4.2 described two seed-level analyses without saying which decides; both are kept, chosen at registration (maintainers' decision 2026-11-19) | Maintainers' decision 2026-11-19 |
| R20-3 | 05 §3.6 | Seed-level achieved power by the noncentral `t` with SE = s_seed/√S (or Welch's SE and df); with the training-seed unit `min_paired` counts runs per trained arm, defaults to 5, and fewer than 5 registered seeds are refused; missingness applies within each run | §3.6's power and `min_paired` were stated for items; §4.2 already refuses three seeds per arm | Roadmap 2026-11-19, workstream 4 |
| R20-4 | 05 §3.1, §3.7 | The clause example and the registration field list name `seed_analysis`, `training_seeds`, `baseline_training_seeds` and `seed_pairing` | Consistency with R20-1 to R20-3 | — |
| R20-5 | Header, status lines | Revision 20 | — | — |

## Revision 21 changes

New capability rather than a defect: a goal and a small labelled dataset in, a trained model and a report out, with the evaluation data sealed from the training side. 05 §18 defines it and fixes the rule ids AN-1 to AN-11 that tests cite. The keeper daemon and the alpha-investing variant are not decided here; they stay open for step 2.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R21-1 | 05 §18.1 | Project (one goal, one directory, `project.yaml`) and the `--json` result shape (`schema`, `ok`, `result` or `problems`, nothing else on stdout) as AN-1 and AN-2 | Agents must not parse human text; the project is the unit of the budget and of resumption | AI-native design, 2026-11-20 |
| R21-2 | 05 §18.2 | Stage protocol, pipeline spec, scenario pack, the fixed order of the trust stages `intake_seal`, `preregister`, `evaluate_sealed`, `gate`, `report`, and idempotent resumable stages as AN-3 to AN-5; `StageContext` refers to a `TrustBackend` and an `AlphaLedger` | Pipelines are customised only through the spec, so the machine itself must refuse a spec that drops or reorders a trust stage | AI-native design, 2026-11-20 |
| R21-3 | 05 §18.3 | `TrustBackend` interface with `sealed_guarantee`; step 1 stand-in `InProcessTrust` ("unsealed (step 1): the operator process can read the sealed items"); intake and seal, preregistration with user-confirmed metric, `evaluate_sealed`, gate and report rules as AN-6 to AN-10 | The seal and the report must be defined by what a backend guarantees, so a stand-in cannot be read as the full guarantee | AI-native design, 2026-11-20 |
| R21-4 | 05 §18.4, §18.5 | Scorer library with a pilot sigma per scorer (0.5 for `exact_match`); `AlphaLedger` interface; step 1 stand-in `FixedAlphaLedger` (alpha 0.05, one attempt) | Intake needs a variance to plan power; one fixed test is the only repeat-testing rule that needs no further choice | AI-native design, 2026-11-20 |
| R21-5 | 05 §18.6, §18.7 | Keeper daemon, signed records and the alpha-investing variant with its false-discovery guarantee stay open (step 2); drivers and the local backend that executes jobs itself are later steps | These need choices the maintainers have not made | AI-native design, 2026-11-20 |
| R21-6 | Header, status lines | Revision 21 | — | — |
| R21-7 | 05 §18 (AN-7) | Optional registration field `base_model: {model_ref_sha256}` that binds the registered base model; registrations without it are unchanged | `evaluate_sealed` must refuse a baseline that is not the registered base model, and the registration schema had no field for it | AI-native design §5.5 ("records the base model") |
| R21-8 | 05 §18.3 (`TrustBackend`, AN-8), §18.5 | One sealed evaluation per project: `evaluate_sealed` claims it through the backend (`claim_sealed_evaluation`, `sealed_evaluation`) for the candidate's model reference hash and the registration before it scores any item; the same candidate may be scored again, another candidate is refused. §18.5 states that the alpha is per project in step 1 and that a new test needs new sealed data | Scoring several candidates on the sealed set and gating the best one selected on the test data without spending alpha, against §18.5 ("repeated testing is not allowed"); a new project on the same data would get a fresh alpha on the same sealed split | final review AI-native step 1 (C1, I1) |
| R21-9 | 05 §18.3 (`sealed_guarantee`, AN-7, AN-10) | `InProcessTrust`'s `sealed_guarantee` also states that every step 1 stage can read and rewrite the trust records and the attempt log; the report's known limits are a list (the guarantee, the per-project alpha, the `metric_confirmed` flag, the number of `evaluate_sealed` attempts); AN-7 states that the step 1 confirmation is the decision flag `metric_confirmed`, with no separate channel to the user | The step 1 guarantee and the report's known limits understated what the stand-in leaves open | final review AI-native step 1 (I2) |
| R21-10 | 05 §18.4 | The planning standard deviation for `exact_match` is that of the paired per-item difference, the bound for two independent 0/1 scores at p = 0.5: sqrt(0.5^2 * 2) = sqrt(0.5), about 0.71; it replaces the pilot sigma 0.5 of R21-4 | The gate tests the paired difference, whose standard deviation can be sqrt(2) times one score's; planning with 0.5 sized the sealed part at half of what the stated power needs | final review AI-native step 1 (I3); user decision, option b, 2026-11-20 |
| R21-11 | 05 §18.2 (`StageContext`, AN-3) | A stage writes its outputs under its name in the pipeline, which `StageContext` carries; an output written under another stage's name is refused | Records written under the stage type collided when a pipeline used one type twice, and the stage machine hashed a directory the stage did not write | final review AI-native step 1 (I4) |
| R21-12 | 05 §18.3 (AN-7) | `preregister` refuses a base model reference that names no generator an installed scenario pack serves, and the registration records the pack and the generator | `evaluate_sealed` needs the generator for both arms; refusing only there came after training | final review AI-native step 1 (I5) |
| R21-13 | 05 §18 intro, §18.1 (AN-1, AN-2, new AN-11), §18.3 (intro, `TrustBackend`, AN-9, AN-10), §18.8 | Text aligned with the step 1 code: the `--json` keys (`schema`, `schema_version`, `command`, `ok`, `exit`, `data`, `problems`, `warnings`; no `result`) and the `USAGE` and `INTERNAL` envelopes; a stage record carries a schema id and version and is tied to its project by the project directory, not by a project id; `latest_registration` in the `TrustBackend` listing; the stage `evaluate_sealed` calls the backend method `run_sealed`; a refused gate attempt carries its rule and reason, and the alpha granted is in the ledger's entries; the AN-10 items step 1 does not deliver are marked "step 1: not yet"; the project commands become rule AN-11 with an index row; long lines rewrapped | The design text and the step 1 code disagreed, and the code already cited AN-11 | final review AI-native step 1 (I6) |
| R21-14 | 05 §18.3 (AN-8, AN-10) | Once the sealed-test claim exists, `evaluate_sealed` refuses every further scoring, also for an unchanged model reference; a done stage with unchanged inputs is still reused. A model reference must content-address its weights; step 1 does not check it, and the report's known limits say so | R21-8 let the same reference be rescored; a reference to a mutable checkpoint path keeps its hash across retrains, so best-of-N selection on the sealed set and a gate PASS were possible | final review AI-native step 1 (re-review of C1) |

## Revision 22 changes

New capability rather than a defect: an external NeMo RL run preempted by the scheduler resumes from its last committed cut, and `ext-collect` counts consumption per segment. 05 §12.4 already defines segments; this revision implements it for linear chains in external runs and fixes the rule ids EXT-RES-1 to EXT-RES-7 that tests cite.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R22-1 | 02 §2.1 | External runs use `attempt_id = <run_id>.<A>` (A = 1, 2, ...) and `controller_epoch = A`; the parent of attempt A+1 is `(A, branch_step)` | An external run has no scheduler job id to build the attempt id from | Resumable runs design, 2026-11-20 |
| R22-2 | 05 §12.4 | External runs implement segment lineage for linear chains: attempt A+1 restores the newest committed cut of attempts 1..A; a rollback (any other restore) is refused; records after a non-final attempt's last cut are `orphaned` and excluded from rollups and impact answers | Preempted runs must resume without double-counting or losing consumption; branching trees need choices not yet made | Resumable runs design, 2026-11-20 |
| R22-3 | tests/README.md (framework) | Rule ids EXT-RES-1 to EXT-RES-7 | Later tests cite the rules by id | Resumable runs design, 2026-11-20 |
| R22-4 | Header, status lines | Revision 22 | — | — |

## Revision 23 changes

New capability rather than a defect: step 2 of the AI-native path makes the seal hold against an agent that runs as the user with a shell. A keeper daemon under its own OS user holds the sealed items, registrations, claims, per-item records, the alpha ledger and an Ed25519 key; every trust record is signed and `gatedtrain verify` checks them against a trust anchor; alpha-investing replaces one sealed scoring per project, so each sealed evaluation spends alpha first (AN-8 and AN-9 change); the scorer library grows to five scorers. 05 §18.9 defines the keeper and the rule ids AN-12 to AN-19 that tests cite.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R23-1 | 05 §18 intro and heading | The section covers revisions 21 and 23; revision 23 adds the keeper (§18.9), signed records, the alpha-investing rule and the scorer library, and the rule ids AN-12 to AN-19 | Later tasks cite the new rules by id | `AI-NATIVE-STEP2-DESIGN.md`, 2026-11-21 |
| R23-2 | 05 §18.1 (project) | The project is the unit of the alpha budget on `InProcessTrust` only; under the keeper the wealth belongs to the sealed set | Alpha-investing is keyed by the sealed set, so a new project on the same data shares the wealth | `AI-NATIVE-STEP2-DESIGN.md` §5.4, 2026-11-21 |
| R23-3 | 05 §18.3 (intro, `TrustBackend`) | `mode` (`in_process` or `keeper`); `claim_sealed_evaluation` spends alpha first and grants one attempt; `sealed_evaluation` is the latest claim; the full design's backend is `KeeperTrust` (§18.9); `InProcessTrust` stays for tests and users without a keeper; signing belongs to the keeper (AN-14) | The step 1 text named the keeper as open and the claim as the project's only one | `AI-NATIVE-STEP2-DESIGN.md` §2, §5.4, 2026-11-21 |
| R23-4 | 05 §18.3 (AN-6) | A sealed label may also be a list (a set-valued answer, as `set_f1` scores it); under the keeper the split runs in the keeper and the power plan uses the planning alpha 0.02375 | `set_f1` needs list labels, which intake refused; the keeper takes the split out of the project | `AI-NATIVE-STEP2-DESIGN.md` §5.2, §5.6, 2026-11-21; plan pre-flight finding 2 |
| R23-5 | 05 §18.3 (AN-7) | Under the keeper the registered alpha is the fixed planning alpha (the test level alpha_j comes from alpha-investing) and the confirmation is the keeper's signed record (AN-18) | The ledger's alpha changes with the wealth; the step 1 flag is replaced by a keeper record | `AI-NATIVE-STEP2-DESIGN.md` §5.3, §5.4, 2026-11-21 |
| R23-6 | 05 §18.3 (AN-8) | "One sealed scoring per project" becomes "each sealed evaluation spends alpha first": the claim asks the ledger for alpha_j and is recorded before any item is scored; attempts differ only in the candidate and the baseline is scored once per sealed set; under the keeper the keeper scores and keeps the per-item records and the driver sees only status and alpha; on `InProcessTrust` `FixedAlphaLedger` still allows one scoring | Alpha-investing allows several attempts; without a spend per evaluation an agent could score many candidates and gate the best; per-item results would let an agent hill-climb on the sealed set | `AI-NATIVE-STEP2-DESIGN.md` §5.4, 2026-11-21 |
| R23-7 | 05 §18.3 (AN-9) | The gate runs at the attempt's alpha_j, computed and signed by the keeper; a granted attempt always settles (crash, MISSING and INCONCLUSIVE count as not passed and cost their alpha); an attempt open at a keeper restart is gated if its complete per-item records are stored, else settles not passed; a PASS ends the project's attempts; an amendment after the first sealed attempt makes later results `post_hoc` | A retry must never be free, and a later FAIL must not hide a PASS | `AI-NATIVE-STEP2-DESIGN.md` §5.4, 2026-11-21; plan interpretation 5; pre-flight finding 8 |
| R23-8 | 05 §18.3 (AN-10) | The headline is the PASS attempt's, else the latest attempt's; under the keeper the keeper builds and signs `report.json`, the report carries the key fingerprint and closes the sealed set, and the keeper's known limits are listed; `report` verifies every signature | Per-item records stay in the keeper, so the report must be built there | `AI-NATIVE-STEP2-DESIGN.md` §5.4, §5.5, 2026-11-21; plan interpretation 5 |
| R23-9 | 05 §18.4 | Five library scorers with evaluator ids, scores, direction (higher is better) and planning SD sqrt(0.5) for 0/1 and [0, 1] scorers; recipe scorers registered by a scenario pack by reference; new rule AN-17 | The library of the AI-native design §5.5; intake sizes before the scorer is chosen | `AI-NATIVE-STEP2-DESIGN.md` §5.6, 2026-11-21; plan interpretation 3 |
| R23-10 | 05 §18.5 | `AlphaLedger.planning_alpha`; Foster-Stine alpha-investing in the keeper keyed by the sealed set: W0 = 0.05 * (1 - 0.05) = 0.0475, alpha_j = W_{j-1} / 2, cost alpha_j / (1 - alpha_j) when not passed, payout 0.05 on a PASS, refusal when alpha_j < 0.005, planning alpha W0 / 2 = 0.02375, mFDR at level 0.05; new rule AN-15; `FixedAlphaLedger` stays | Revision 21 left the alpha rule to the maintainers (approach A chosen) | `AI-NATIVE-STEP2-DESIGN.md` §4, §5.4, 2026-11-21 |
| R23-11 | 05 §18.7 | "Open in revision 21" becomes "Open after revision 23": the keeper, signatures, the alpha rule and the scorer library are decided; containers (step 3, with the local backend and model-file access for real packs), MCP server and orchestrator (step 4), hosted keeper, multi-host, Windows/macOS service installation, key rotation and a second confirmation channel stay later | Step 2 decides what revision 21 left open | `AI-NATIVE-STEP2-DESIGN.md` §2, 2026-11-21; plan interpretation 6 |
| R23-12 | 05 §18.8 | Index rows AN-12 to AN-19; subjects of AN-8, AN-9 and AN-10 updated | Later tests cite the rules by id | `AI-NATIVE-STEP2-DESIGN.md`, 2026-11-21 |
| R23-13 | 05 §18.9 (new) | The keeper: process and protocol (AN-12), root-owned environment not writable by the owner uid and model file formats (AN-13), `setup keeper` prints only (AN-19), intake over the socket with items tracked by canonical input hash, byte-identical reuse, released training inputs forced into the training part of later splits, and a report closing the sealed set (AN-16), metric confirmation record (AN-18), signatures and the trust anchor (AN-14), and `KeeperTrust.sealed_guarantee` with its same-user variant | The seal must hold against an agent that runs as the user with a shell | `AI-NATIVE-STEP2-DESIGN.md` §5.1-5.7, 2026-11-21; plan interpretations 1, 2, 4; pre-flight findings 12, 21 |
| R23-14 | tests/README.md (framework) | Rule table for AN-12 to AN-19 and the extended AN-8, AN-9, AN-10 | Later tests cite the rules by id | `AI-NATIVE-STEP2-DESIGN.md` §7, 2026-11-21 |
| R23-15 | Header, status lines | Revision 23 | — | — |
| R23-16 | 05 §18.3 (AN-8) | The driver-visible statuses include MISSING | The keeper returns MISSING for an attempt whose records are missing; the list named three statuses | final review of AI-native step 2, 2026-11-21 |
| R23-17 | 05 §18.3 (AN-10) | The keeper's known limits add the intake and registration oracles and the assumption behind the mFDR guarantee | The report must state every limit the keeper has (§18.5, §18.9 AN-16) | final review of AI-native step 2, 2026-11-21 |
| R23-18 | 05 §18.4 | "0 for a prediction that is not canonical JSON" applies only to a prediction passed to a scorer; under the keeper a generator value that is not canonical JSON is a crash (AN-9), settled not passed | The keeper checks the generator's output before any scorer sees it | final review of AI-native step 2, 2026-11-21 |
| R23-19 | 05 §18.5 | The bid is capped at W / (1 + W) (identical to W / 2 up to W = 1); the mFDR guarantee is stated with its assumption (each test valid given the earlier outcomes; the sealed items reused, only each status revealed) as a known limit | Above W = 1 a bid of W / 2 would cost more than the wealth; the guarantee's assumption was unstated | final review of AI-native step 2, 2026-11-21 |
| R23-20 | 05 §18.9 (AN-13) | The start-up check accepts code owned by root or the keeper's uid, so a keeper-owned environment is owner-proof only under a separate keeper user (in same-user test mode it is the owner's); the socket's directory must not be writable by another uid | The text claimed a keeper-owned environment is never writable by the owner uid; a writable socket directory lets another uid replace the socket | final review of AI-native step 2, 2026-11-21 |
| R23-21 | 05 §18.9 (AN-19) | `setup keeper` refuses a path holding `%`, `$`, whitespace, a quote or a backslash | systemd rewrites or splits those in `ExecStart=` | final review of AI-native step 2, 2026-11-21 |
| R23-22 | 05 §18.9 (AN-16) | Known limit: intake refusals are an unthrottled membership oracle and registration refusals reveal label-type counts | Opening a project is free and a refused intake stores nothing | final review of AI-native step 2, 2026-11-21 |
| R23-23 | 05 §18.9 (AN-18) | The sentence shown is "the one plain sentence the user is asked to confirm" | AN-7 has no such sentence to refer to | final review of AI-native step 2, 2026-11-21 |
| R23-24 | 05 §18.9 (AN-14) | `verify` against a same-user keeper is consistency only (exit 1); the driver syncs its receipts with the keeper's list after evaluate, gate and report and when the report is printed; a non-refusal scoring error after the grant returns the spend's envelopes; the missing-record message names both causes | An honest project failed verify after an interrupted evaluation, with a message blaming tampering; a same-user keeper's key is readable by the owner | final review of AI-native step 2, 2026-11-21 |

## Revision 24 changes

Defect found while preparing the next package release: 05 §3.7 required every registration to record the analysis code version and the framework refuses one that differs from the running code, but the text did not say what that version names. The prototype used the package version, so every package release would have invalidated every existing registration, also when the analysis code had not changed.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R24-1 | 05 §3.7 | The registration's `analysis_code_version` names the analysis code (gate statistics and registration semantics), not the package release; it changes only when that code's behaviour changes, and a package release that leaves it unchanged leaves existing registrations valid | Decouple the analysis code version from package releases | maintainers, 2026-11-22 |
| R24-2 | Header, status lines, README | Revision 24; the prototype (`gatedtrain` 0.4.0) implements revision 24 except the training-seed unit of revision 20 (05 §3.3) | — | — |

## Revision 25 changes

New capability rather than a defect: step 3a of the AI-native path. Step 2's separate-user keeper holds the seal only while the user's account cannot reach a rootful container daemon, and a member of the `docker` group can. Revision 25 runs the user's jobs and the keeper's sealed evaluation on rootless Docker, detects rootful access and downgrades the guarantee when it finds it, shares one rootless variant of the launch rules between `ext-launch` and a new local backend that runs the AI-native path's jobs itself, and runs a scenario pack's batch generator in the keeper's inference container on a copy of the model files the keeper hashed. 05 §18.10 defines it and the rule ids AN-20 to AN-25 and EXT-263 that tests cite.

| ID | File and section | Change | Reason | Source |
|---|---|---|---|---|
| R25-1 | 05 §18 intro and heading | The section covers revisions 21, 23 and 25; revision 25 adds rootless execution, rootful-access detection, the local backend and the keeper's inference container (§18.10) and the rule ids AN-20 to AN-25 and EXT-263 | Later tasks cite the new rules by id | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 |
| R25-2 | 05 §18.3 (AN-8) | Under the keeper a per-item generator runs in the keeper's process and a batch generator in its inference container (§18.10) | The keeper gains a second way to run the candidate | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 |
| R25-3 | 05 §18.6 | "Changes the principle that gatedtrain never starts a process for this path only" becomes "gatedtrain starts processes on the AI-native path only (the local backend and the keeper's inference container)"; the `ext-*` commands keep printing commands; budget and approvals are cooperative; the local backend is defined in revision 25 | The local backend and the keeper's container start processes; the principle must say where | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.4, §5.8.3, §7 |
| R25-4 | 05 §18.7 | "Open after revision 23" becomes "Open after revision 25": step 3a decided (rootless Docker and host setup, detection, EXT-263, the local backend, batch generators in the keeper's container, model hand-off); step 3b is the text pack, the GPU end to end and one recipe smoke under the rootless rules; the later list is otherwise unchanged | Step 3a decides what step 2 left for step 3 | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §6 |
| R25-5 | 05 §18.8 | Index rows AN-20 to AN-25 and EXT-263; subjects of AN-8, AN-9, AN-10, AN-13, AN-14 and AN-19 extended | Later tests cite the rules by id | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 |
| R25-6 | 05 §18.9 (heading, sealed evaluation, AN-8, AN-9 and AN-10 extended) | A batch generator's sealed evaluation runs only in the keeper's inference container and the in-process path or a keeper without an inference image refuses it before any spend (AN-8); a non-zero exit, a timeout or malformed predictions after the grant are a crash (AN-9); `report.json` gains a `compute` block (keeper: signed; backend: the driver's job records, cooperative) and the new model reference limit (AN-10) | The keeper runs candidates in a container; the report states GPU time from both sides | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.5, §5.8.3; plan interpretation 6 |
| R25-7 | 05 §18.9 (AN-13) | `.jinja` model files; the keeper copies each listed model file into its own state before the grant (no link, regular files only, the opened path checked, hash of the copy) and runs only the copy; with an inference image the environment record adds the image ID, the eval trees' file hashes and the Docker CLI's hash, and the keeper refuses to start on another image ID, an eval tree open to another uid or with a link leaving the trees, an unchecked Docker CLI or `nvidia-smi`, or a daemon that is not rootless or whose socket the keeper does not own | The hash in the claim must be the hash of what runs; the inference environment is code the keeper runs | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.5, §5.8.1, §5.8.4; pre-flight findings 3, 10, 15, 20, 25, 27; ruling on finding 10 |
| R25-8 | 05 §18.9 (AN-14) | `hello` and `keeper fingerprint` report `owner_has_root_docker` and its reasons, per project when a project is named; `verify` exits 1 on rootful access, on a report stating the downgraded guarantee and on a report stating the revision 23 guarantee text | A downgraded keeper must not verify as sealed; reports signed before the check did not check it | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.2; pre-flight findings 2, 6, 18 |
| R25-9 | 05 §18.9 (AN-19) | `setup keeper` prints the inference options (`--inference-image`, `--eval-tree`, `--docker-host`) into `ExecStart=`; the host script's drop-in uses the same serve command and orders the keeper after the keeper user's service manager | The keeper's daemon must be up before the keeper checks its image | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.5; pre-flight findings 17, 20 |
| R25-10 | 05 §18.9 (guarantees) | New `KeeperTrust.sealed_guarantee` text (container, copy of the model files, the escape to the keeper's account, no rootful access found, sudo not checked); the downgraded text when detection is positive now or was on the sealed set; the same-user text takes precedence; the revision 23 text is no longer stated; the guarantee is per project | The guarantee must state exactly what the host setup provides | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.2, §5.5, §5.8.2; plan interpretation 5; pre-flight findings 2, 12, 18; ruling on finding 12 |
| R25-11 | 05 §18.10 (new) | Host setup printed as one root script (AN-20) with the group removal before lingering and the CDI check; rootful-access detection with its scope, fail-closed reading and the sticky per-set downgrade (AN-21); the rootless launch rules (EXT-263); the local backend: jobs, reattach, per-attempt output directories, job records, GPU accounting, budget, retries, GPU wait and the GPU log row (AN-22, AN-23); stage access (AN-24); batch generators and the keeper's inference container with its uid, options, eval tree and prediction contract (AN-25); model hand-off; the GPU on one host; budget and approvals as cooperative | Step 3a of the AI-native path: the seal must hold on a real host, and the AI-native path runs its own jobs | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.1-§5.5, §5.8; plan interpretations 1-4, 7; pre-flight findings 1, 3, 4, 7-9, 11-14, 21, 22, 25-28; rulings on findings 10 and 12 |
| R25-12 | 06 §3.12 (new) | AI-native threat paragraph: root-equivalent access by the user's account downgrades the keeper's guarantee; an escape from the keeper's inference container gives the keeper's account | The security document did not cover the AI-native trust boundary | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 §5.2, §5.8.2 |
| R25-13 | tests/README.md (framework) | Rule table for AN-20 to AN-25, EXT-263 and the extended AN-8, AN-9, AN-10, AN-13, AN-14, AN-19 | Later tests cite the rules by id | `AI-NATIVE-STEP3-DESIGN.md`, 2026-11-22 |
| R25-14 | Header, status lines | Revision 25; the prototype (`gatedtrain` 0.4.0) still implements revision 24 except the training-seed unit of revision 20 (05 §3.3) | — | — |
