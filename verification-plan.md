# gatedtrain: Verification Plan

**Status:** Companion document, **not part of the frozen design**; the design set is at revision 28, and this plan was last changed in revision 12. The prototype (`gatedtrain` 0.7.1) implements revision 28 except the training-seed unit of revision 20 (05 §3.3). This plan changes no decision in `design.md`, 01–08, or `open-issues.md`. Where it finds a gap that needs a design decision, it proposes an entry in Appendix A; it does not edit `open-issues.md`.

**Owner:** the P0 maintainer (role). **Revision:** draft 1 (2026-11-05), written against design revision 9. Amended 2026-11-05 only to register the test IDs cited by the project's implementation plan (not published), revisions 2 to 4.1 (§4.6), to state the P2-min simulation subset explicitly (§3.4), and to state the P1 scope of SEC-B3 (§4.4).

**Numbers:** every number taken from the design cites its section. Every other number is marked *(placeholder)* or *(estimate)* and is confirmed at the phase gate that first needs it.

## 1. Purpose and scope

1. **Purpose.** Organize how each claim of the design is shown to hold: which test, in which layer and environment, owned by which role, run on which trigger, and what its failure blocks.
2. **Scope.** The MVP (G0, P0, P1a, P1b or P6a, P2-min or P2-full; design §7.1) in full detail; P3 and later only to the depth needed to keep test IDs stable (03, 07, and the P3+ parts of 04 and 06 are "Direction — detailed review after G0").
3. **Relation to the design.** The design states requirements, invariants, exit criteria, SLOs, the security baseline, and statistical procedures. This plan maps each to tests (§4) and never redefines them. If a test cannot be written without a new decision, the gap goes to Appendix A.
4. **How this plan changes.** The P0 maintainer revises it at every phase gate (G0 close, P0 exit, P1a/P1b exit, P2 exit, P3 entry). Each revision records: test IDs added or retired (IDs are never reused), placeholders replaced by measurements, waivers opened or closed (§11), and Appendix A entries accepted into `open-issues.md` by the project lead.

## 2. Verification principles

1. **Claims are falsifiable.** Every requirement maps to at least one check with a stated pass rule. A claim that no test can fail (for example "runs keep training while fenced" without a measured productive-time target) is reported as unverified, not as passed.
2. **Tests before code for protocols.** TLA+ models (03 §12.1) and their deterministic-simulation conformance tests (03 §12.2) are written and checked before the runner, broker, janitor, or supervisor code they constrain. P2-full: the P2 requeue model is checked before any P2-full runner code (design §7.2).
3. **Deterministic first, then real clusters.** Each fault runs in seeded simulation before staging (07 §9). A staging failure that the simulator did not predict produces a new simulation case before the fix merges.
4. **Every test has an owner and a blocking target.** A test without an owner role, a trigger, and a named thing it blocks (merge, release, phase exit, cluster enablement) is not part of the plan.
5. **Verify the verifiers.** Statistical gates, detectors, contamination checks, lineage queries, security controls, and fencing guards are themselves tested with seeded faults whose correct answer is known (§8).
6. **Evidence is a lineage artifact.** Phase-exit reports, gate reports, and campaign logs are stored as immutable artifacts with manifests (05 §3.7), so the verification of a release is itself traceable.
7. **No silent waivers.** A skipped or failing blocking test needs a written waiver with owner and expiry (§11).

## 3. Test layers

| Layer | Owner role | Environment | Trigger | Failure blocks |
|---|---|---|---|---|
| L1 Static checks and model checking | Control-plane maintainer (models); project lead (lint, typing, pickle ban) | CI CPU | Every merge request touching protocol code or models; nightly | Merge; protocol code for the modelled phase |
| L2 Unit and property tests | Package maintainer | CI CPU (A, E-stats, R-svc, E); CI GPU for B, C-ext, D, H kernels | Every merge request | Merge; package release |
| L3 Conformance kit | Project lead | CI CPU and CI GPU | Every merge request to a plug-in or trainer adapter; nightly supported-combination matrix (01 §9.2) | Plug-in admission; trainer adapter release; adoption metric "trainers passing kit v0" |
| L4 Deterministic simulation | Control-plane maintainer | CI CPU, one seeded process (03 §12.2) | Every merge request to R or E; nightly with fresh seeds | Merge; staging chaos run |
| L5 Single-node integration | Algorithms maintainer | Local toy, CI GPU | Merge requests to B, C-ext, D; nightly | Release of B, C-ext, D |
| L6 Parity and statistical acceptance | Evaluation maintainer with algorithms maintainer | Training cluster (dedicated or staging partition) | Phase exit (P1b, P5, P6a), registered in advance (05 §3.7) | Phase exit |
| L7 Performance | Performance maintainer | Training cluster, interleaved A/B on the same nodes (08 §7) | P0 exit, P1b exit, P4 exit; weekly trend *(placeholder)* | Phase exit; C-ext release |
| L8 Security | Security maintainer | CI, staging partition, red-team exercise | Merge (B9, B10 scanners); phase entry on shared clusters (06 §4) | Running the phase on a shared cluster |
| L9 Chaos and game days | Operations maintainer (P3); project lead (P1b, P2) | Simulation, then staging Slurm partition | Phase exit per 07 §9 "Phase exit" column; quarterly game days from P3 | Phase exit; cluster enablement |
| L10 End-to-end acceptance | P0 maintainer with the pilot users | Staging partition, then pilot-user partition | Phase exit (§10) | Phase exit |
| L11 Production verification | Operations maintainer (P3); project lead (P2 report-only) | Pilot users' runs | Continuous; canary per release | Release widening; P3 exit (four weeks of SLOs, 07 §7) |

### 3.1 L1: static checks and model checking

- **TLA+ models (03 §12.1):** P2 requeue (P2-full only: `NoTrainingAfterCancel`, `NoTrainingOfFinishedStage`, `OneEpochGrantPerRestart`, `HeldJobsResolved`); stage lifecycle (`OneLiveGangPerStage`, `NoLaunchAfterCancel`, `NoSubmitOnUnknown`); lease, epoch, clock, fenced phases (`NoSideEffectFromSupersededEpoch`, `SingleLeaseHolder`, `NoReleaseBeforeFenceBound`); cut commit, restore, GC, gate pins (`RestoreOnlyCommittedCurrentLineage`, `NoGCOfPinnedOrReferencedCut`); gate and publish (`AcceptedVersionAtMostOncePerTurn`, `GateOutcomeByCommitOrder`). Each with its liveness property.
- **Fault model in every spec:** message loss, duplication, reordering, process pause, bounded drift, store failover, scheduler accounting lag, plus the revision-8 additions (terminal-but-alive jobs, requeue of finished jobs, delayed health-service actions, the contract layer).
- **Checkers:** TLC on bounded instances (3 stages, 2 recoveries, 2 supervisor leaders) in CI; Apalache for the timing model (03 §12.1). A model change and its conformance test change land in the same merge request.
- **Static rules:** package A imports without torch (01 §9.2 schema row); any pickle load other than `torch.load(weights_only=True)` fails CI (B9); secret scanning (B10); NeMo RL with C-up never imports A, R, E, G, F (design §4 rule 1), checked by an import-graph test.

### 3.2 L2: unit and property tests per package

| Package | Focus | Property-based targets |
|---|---|---|
| A `gatedtrain-contracts` | Manifests, lineage records, run-contract formats, `issue_epoch` request/response and exit 79, emitter, Ed25519 verification, both transports, `ConsumptionRecord` and outcome enum | Readers reject unknown major and accept unknown minor fields (07 §11); signatures fail on any byte flip; create-only races and atomic replace on both transports (02 §10); golden files for one-way-door formats (design §8) |
| B `gatedtrain-denoise` | SDE log-probs, analytic KL, step selection, segment ratios, DiffusionNFT loss | `gaussian_exact = gaussian_mean_sq × d` at equal variances (01 §9.2); invariance of losses to row permutation; finite gradients across the timestep range |
| C-ext `nemo-rl-gatedtrain` | Loss wrappers, emitter wiring, verb verification, 8a-1 and 8d shims | `TOKEN`-only partitions add zero fields and bytes; `ClippedPGLossFn` receives no new argument (hook-1 safety); 8d segment rows equal consumed rows |
| R-agent | Signing, MAC per segment, commit gating, contract handling, restart from store and namespace | Commits and verbs idempotent by `cut_id` and `action_id`; restarted adapter resumes fencing timers from the last recorded renewal (02 §2.5) |
| R-svc | Store interface, broker, `ControlAPI`, `issue_epoch`, plug-in interfaces, run-record envelope | Record-first ordering; `Unknown` never submits; allow-lists derived server-side; owner sections versioned independently |
| E `gatedtrain-workflow` | Spec and IR validation, thin runner, gates, lockfiles, ledger writer | Validator rejects every rule violation of 04 §4.2 and 05 §3.2; resume returns an existing gate result without writes (04 §4.3) |
| E-stats `gatedtrain-gates` | `plan`, `reestimate`, `evaluate`, `combine`, `family_level`, `independence` (05 §3.8) | Interval coverage and type-I error by simulation (STAT-02); `reestimate` never lowers n |
| G, H, F | P3, P6a, P10 | Defined at their phase kickoff; IDs reserved (§4) |

### 3.3 L3: conformance kit

The kit of 01 §9.2 is versioned with package A. Each extension point (`StepForward`, B loss kernels, `ModelAdapter`, rollout engine, environment and reward, algorithm, hook-7 callbacks, estimator/selector/sampler/filter, schema) has the required tests listed there; each test gets an ID `CONF-<point>-<n>`. Additions from other sections: the P2 restore conformance test pinning `get_latest_checkpoint_path` at the baseline commit (02 §5.7); the refit dead-peer test for samplers (02 §8); both contract transports pass one suite (02 §10); injection canaries for every LLM or VLM judge (B26). The supported-combination matrix (family × backend × rollout engine) runs nightly (01 §9.2).

### 3.4 L4: deterministic simulation

One seeded process with a simulated store, clock, and scheduler (03 §12.2). Faults include `sbatch` timeout after creation, ghost jobs, runner crash between CAS and release, accounting lag and cached not-found, cancel during submit, delayed requeue and requeue after cancel, runner unreachable at job start, `NODE_FAIL` with live processes, store outages around the fenced windows, lossy store failover, supervisor failover, hung controller, unreadable contract files, correlated stalls, adapter crash loops, forged comments and nonces, replayed break-glass IDs, and gate decision/timeout races. Each maps to the named conformance test in 03 §12.2 (`broker/submit-ambiguous` … `gate/commit-order`); those names are the `SIM-*` test IDs of this plan. P2-min runs the subset without requeue, held submission, and janitor; P2-full adds `p2/requeue-refused`, `p2/requeuehold-on-unreachable`, `broker/ghost-held`, and `janitor/*`.

**P2-min subset (explicit).** `broker/submit-ambiguous`, `broker/cancel-race`, `find/unknown-on-lag`, `find/spoof-rejected`, `p2/retry-precondition`, `gate/commit-order`, the proposed `runner/stop-at-cap` (§4.6; the P2 variant of `fence/stop-honored`, which stays a P3 case), `liveness/io-error-not-hang` (INV-12, P2), and the proposed `broker/ghost-unheld` (§4.6). `p2/retry-precondition` belongs to both variants (03 §3.3): in P2-min the case is `retry_stage` while the previous job is still queued or not yet absent from the controller; in P2-full it adds a pending requeue (EXIT-P2-6).

### 3.5 L5: single-node integration (research profile and local mode)

- Research profile end to end on one node: Ray local, adapter in local mode (pointer file updated by atomic rename under a lock, 02 §2.5), signed manifests on shared clusters, `unpromotable` outputs, non-blocking exploratory gates (01 §8.1).
- `--capture-batch N` and `--replay` agree bitwise or within the declared tolerance (01 §8.3).
- Each plug-in's single-GPU smoke config (01 §9.1 rule 6).
- Toy models only (§5.3), so the layer fits CI GPU.

### 3.6 L6: parity and statistical acceptance

Procedures are those of 05 §4: G0 feasibility (descriptive only, never "parity"); P1b full parity with at least five seeds per arm, σ planned from the 80% upper bound, TOST on final reward and an independent T2+ non-LLM evaluator, all metrics passing by intersection-union; per-timestep ratio statistics by procedure (b); throughput ratio from 08. Registrations are written before any run and their SHA-256 enters lineage (05 §3.7).

### 3.7 L7: performance

08 §7 and §8 own the procedures: per-stage overhead by interleaved A/B over at least 50 pairs, target ≤ 2% with zero new bytes per token (P0); end-to-end samples/s/GPU ≥ 0.9× verl-omni over at least three interleaved run pairs, lower bound of the 95% interval ≥ 0.9 (P1b, path N); registered checkpoint interval with both loss metrics (P1b, P4). Each must-measure item of 08 §8 is a test `PRF-M<n>` that blocks its phase until recorded.

### 3.8 L8: security

Each baseline item B1–B30 (06 §4) becomes a check with a positive test (the control works) and a red-team test (an attempt to bypass it fails); see §4.4. Controls by phase follow 06 §16: P0 items before P0 exit; B1–B8, B14 QOS, B17, B20 inputs, B24 before any P1 run on a shared cluster; P2-min and P2-full items before P2 runs.

### 3.9 L9: chaos and game days

C1–C24 (07 §9), simulation first then staging. Every run also asserts zero duplicate gangs and zero infrastructure exits counted as run faults (07 §9). Injection mechanisms are in §9. Game days are quarterly from P3, one 07 §3.2 cascade each.

### 3.10 L10: end-to-end acceptance

Executable scenarios per phase (§10). Each produces a verification report artifact.

### 3.11 L11: production verification

- **P2:** no SLOs and no pager (07 §2); this plan records report-only SLIs (07 §8.2 subset) on pilot-user runs, without targets (proposed in Appendix A, VP-OI-7).
- **Canaries:** adapter releases per cluster for 48 h *(estimate, 07 §11)*, widening only if the `adapter_fault` rate is no worse than the previous version; reward and judge releases canaried on a calibration set (07 §3.1).
- **Shadow mode:** supervisor per cluster (03 S5) and reward-hacking detectors (05 §7) record "would act" decisions; promotion criteria are those of the design.
- **P3:** all SLOs of 07 §7 for four consecutive weeks; nightly skew matrix (07 §11 item 6).

## 4. Traceability matrix

Test IDs are stable and never reused. Layer codes are those of §3. "Phase" is the phase whose exit or entry the test blocks.

### 4.1 Invariants (03 §13)

| ID | Invariant | Tests | Layer | Phase |
|---|---|---|---|---|
| INV-01 | At most one released gang per stage key | TLA `OneLiveGangPerStage`; SIM `broker/submit-ambiguous`, `broker/ghost-held`, `janitor/release-held` | L1, L4, L9 | P2 |
| INV-02 | P2-min jobs never requeued (pending R9-OI-1) | Staging: `scontrol show job` requeue flag off, `SLURM_RESTART_COUNT` stays 0; E2E-P2MIN step 6 | L9, L10 | P2-min |
| INV-03 | P2-full requeue trains only as current attempt of a live, uncancelled, under-cap stage | TLA P2 requeue model; SIM `p2/requeue-refused`; C18, C19 | L1, L4, L9 | P2-full |
| INV-04 | P2-full requeue never runs with runner unreachable | SIM `p2/requeuehold-on-unreachable`; C21 | L4, L9 | P2-full |
| INV-05 | Janitor never releases a foreign or early hold | SIM `janitor/release-held`; C21 (administrator hold left alone) | L4, L9 | P2-full |
| INV-06 | One computing gang per run across P3 recoveries | TLA lease model; SIM `supervisor/fence-bound-before-release`, `supervisor/terminal-before-resubmit`; C6, C20 | L1, L4, L9 | P3 |
| INV-07 | No launch after cancel (P2-min: no commit after cancel) | TLA `NoLaunchAfterCancel`; SIM `broker/cancel-race`; E2E-P2MIN step 7 | L1, L4, L10 | P2 |
| INV-08 | Superseded controller causes no durable side effect | TLA `NoSideEffectFromSupersededEpoch`; 02 §10 partition test (8b, P3); epoch-CAS unit tests | L1, L2, L4 | P2 (CAS), P3 |
| INV-09 | Cancellation, budget, safety stops reach a fenced run | SIM `fence/stop-honored` (P3); SIM `runner/stop-at-cap` (P2, §4.6); C14 | L4, L9 | P2 (cap), P3 |
| INV-10 | Store outage within fenced windows causes no resubmission | SIM `lease/fenced-hold-resume`; C1, C2 | L4, L9 | P3 |
| INV-11 | Store outage beyond them causes no resubmit storm | SIM `supervisor/bulk-recovery-ack`; C3 | L4, L9 | P3 |
| INV-12 | Filesystem brownout does not stop runs | SIM `liveness/io-error-not-hang`, `liveness/correlated-stall-hold`; C15 | L4, L9 | P3 (P2: no run killed by the runner) |
| INV-13 | Hung but renewing controller is recovered | SIM `lease/progress-liveness`; C6 | L4, L9 | P3 |
| INV-14 | Adapter bugs do not mark cuts suspect or trip breakers | SIM `supervisor/adapter-fault-not-counted`; C16, C17 | L4, L9 | P2 (C16), P3 |
| INV-15 | Hardware recoveries neither starved nor unbounded | SIM bucket tests; calibration in shadow mode | L4, L11 | P3 |
| INV-16 | Bad nodes excluded while the remediation logic is advisory | C22; node-list transition check of the verification harness (§9.1) | L9 | P3 |
| INV-17 | The cluster health service cannot restart a tagged job outside the framework | SIM `guard/nonce-mismatch`, `find/spoof-rejected`; skew-matrix guard cell | L4, L11 | P3 |
| INV-18 | No recoverer gap during rollout | Per-cluster switch audit; automatic RCA (where the site runs it) still on until health-service watch | L11 | P3 |
| INV-19 | Break-glass not replayable, cannot resume safety pauses | SIM `guard/breakglass-single-use`; SEC-B29 | L4, L8 | P3 |
| INV-20 | Gate timeout and decision cannot both win | TLA `GateOutcomeByCommitOrder`; SIM `gate/commit-order` | L1, L4 | P2 |
| INV-21 | Callers act only within role; operator cannot decide gates, release, resume safety | SEC-B13, SEC-B27 | L8 | P2 |
| INV-22 | Actions applied at most once; unknown failures never retried silently | Action-ID CAS unit tests; hook 5 `unclassified` test | L2 | P0 (hook 5), P2 |

### 4.2 Phase exit criteria (design §7.1)

| ID | Criterion | Tests | Layer |
|---|---|---|---|
| EXIT-G0-1..4 | Four written outcomes; path and both cost tables with P1b budget; MVP site prerequisites or dedicated partitions; P2-min or P2-full recorded | Document review checklist (E2E-G0) | L10 |
| EXIT-P0-1 | Zero new bytes per token on token partitions | Byte-count test on recorded partitions; CONF-schema | L2, L3 |
| EXIT-P0-2 | Per-stage overhead ≤ 2% (interleaved A/B) | PRF-M3 | L7 |
| EXIT-P0-3 | NeMo RL CI green | Upstream CI result on each hook PR; DRIFT-N1 | L2 |
| EXIT-P0-4 | Decision-step worked examples pass | CONF-algorithm worked-example tests (01 §4.4) | L3 |
| EXIT-P0-5 | Hook-1 safety: every LLM recipe and loss unaffected | Recipe sweep with argument-capture on `ClippedPGLossFn` | L2, L5 |
| EXIT-P1a-1 | B parity vs flow_grpo, dancegrpo, diffusionnft | CONF-kernel parity on recorded tensors | L3 |
| EXIT-P1b-1 | TOST on final reward and independent evaluator, ≥ 5 seeds, power analysis | STAT-03 | L6 |
| EXIT-P1b-2 | Per-timestep ratio statistics within reference drift | STAT-04 | L6 |
| EXIT-P1b-3 | Samples/s/GPU ≥ 0.9× verl-omni, ≥ 3 interleaved pairs | PRF-M7 | L7 |
| EXIT-P1b-4 | Checkpoint interval registered (Young/Daly), expected loss reported | PRF-M8 | L7 |
| EXIT-P6a-1..3 | Reward-curve TOST; env-steps/s ≥ 80%; export equivalence | STAT-03 (PPO variant); PRF-M14; export test (08 §6.5) | L6, L7 |
| EXIT-P2-1 | SFT → GRPO → gate pipeline on Slurm, gates report power and MDE | E2E-P2MIN; STAT-01 | L10 |
| EXIT-P2-2 | Impact query from shard to checkpoints via weight-version ancestry | VV-LIN-1 plus live query in E2E-P2MIN | L2, L10 |
| EXIT-P2-3 | Workflow violating gate independence rejected | VV-GATE-1 | L2 |
| EXIT-P2-4 | Restricted samples cannot reach external rewards or telemetry | SEC-B20 red-team cases | L8 |
| EXIT-P2-5 | Approval by the requester rejected | SEC-B27 | L8 |
| EXIT-P2-6 | No Slurm requeue; `retry_stage` refuses while the previous job is queued | INV-02; SIM `p2/retry-precondition`; E2E-P2MIN | L4, L10 |
| EXIT-P2-6F | P2-full: requeue after cancel and stale requeue refused | C18, C19 | L9 |
| EXIT-P2-7 | The P2 operator runs the P2 store | Operations checklist (runbooks, backup restore drill) | L10 |
| EXIT-P2-8 | Releases private only | SEC-B25 | L8 |

### 4.3 SLOs (07 §7; apply from P3)

| ID | SLO | Verification |
|---|---|---|
| SLO-01, SLO-02 | Store availability 99.95%; per-cluster reachability 99.9% | Probes from each cluster and a compute-node canary; C1–C3 confirm probes detect outages |
| SLO-03, SLO-04 | Zero duplicate gangs; zero infrastructure exits counted as run faults | Asserted in every chaos run (07 §9) and continuously in production |
| SLO-05 | Productive time while fenced ≥ 90% *(estimate, 07)* | C1 measurement of `gatedtrain.fenced=true` spans against commits |
| SLO-06, SLO-07 | Detection to action p95 ≤ 30 s in-run; ≤ 2 min from health-service verdict | Intent-log timestamps in C6, C12, C22 |
| SLO-08, SLO-09 | Restart to first step p90 ≤ 15 min (≤ 64 nodes); bulk recovery tail | C3, C6, C20 timing; game days |
| SLO-10, SLO-11 | Checkpoint freshness; restore success | C8, C24; restore-event counters |
| SLO-12 | RL-stage goodput ≥ 90% | nemo-lens buckets on pilot-user runs |
| SLO-13, SLO-14 | Serving shards below desired < 5%; rollout efficiency ≥ 85% *(estimate)* | C13; ledger outcome counts |
| SLO-15, SLO-16 | Shared reward/judge p99; shared-service idle GPU ≤ 15% | C10 (P4); gateway metrics |

### 4.4 Security baseline (06 §4)

| ID | Positive check | Red-team case (must fail) | Phase |
|---|---|---|---|
| SEC-B1, B2 | Reward actor writes only `rewards.<name>`; controller actor unnamed | Plug-in actor writes advantages; looks up controller by name | P1 |
| SEC-B3 | Ledger segments MACed by the agent's user. **P1 scope** (no store or runner yet): MAC key unreadable by the team UID; the agent's local-mode commit is the acceptor and records the MAC list in the B9-signed cut manifest; later consumers (import stage, P2 ledger writer and commit RPC) re-check it (open-issues PI-23, decided by the maintainers; decision record not published) | Team-UID process reads the MAC key or forges a segment accepted at commit; a segment modified after acceptance passes the re-check | P1 (local-mode acceptor), P2 (store acceptance) |
| SEC-B4 | Run writes only its attempt/epoch prefix, reads declared inputs | Write to a sibling run's prefix; read an undeclared input; find a store credential in the job environment | P1 |
| SEC-B5, B6 | Ray dashboard, job API, client server off; prolog firewall | Port scan from another allocation reaches Ray or data-plane ports | P1 |
| SEC-B7, B8 | Ray TLS and token; CURVE data-plane transport | Unauthenticated client joins Ray or TransferQueue | P1 |
| SEC-B9 | Signed manifests verified before load; safe formats | Tampered manifest or pickle payload loads | P0 |
| SEC-B10 | Secret scanning in CI and image build | Planted test secret passes the scanner | P0 |
| SEC-B11 | Adapter-owned committed prefix; copy-on-write restore | Team UID modifies a committed cut; restore via hard link | P0, P2 |
| SEC-B12 | Runner limited to `sbatch`, `scontrol release`, `scancel` on its account; adoption needs UID and nonce | Runner cancels another team's job; adoption of a job with forged comment | P2 |
| SEC-B13, B27 | `ControlAPI` authenticates every call; role allow-lists | Unauthenticated call; operator calls `decide_gate`, release, or safety resume; requester approves own gate | P2 |
| SEC-B14 | QOS cap and runner cap | Submission over cap; job continues after runner is down (QOS must stop it, C14) | P1, P2 |
| SEC-B15 | Per-principal rate limits; paid-judge spend cap | Burst of `retry_stage` calls; judge spend past cap | P2 |
| SEC-B16 | Hash-chained audit, WORM anchor ≤ 5 min | Rewrite an audit row without breaking the chain check | P2 |
| SEC-B17 | Digest allow-list via reviewed merge request | Lockfile with an unlisted digest validates | P1 |
| SEC-B18 | Code-executing environments blocked | Manifest with `sandbox: none` and code execution admitted | P0 |
| SEC-B19 | Capture off; export only through the exporter | Model copied out of the run prefix to an external location | P0, P2 |
| SEC-B20 | Tag-scoped read credentials; export and sink checks | Restricted row read by a research run; restricted sample sent to an external reward or telemetry sink | P1, P2 |
| SEC-B21 | Per-team accounts, `run_root` 0700, per-run ACLs | Cross-team read of `run_root`; shared framework account usable | P2 |
| SEC-B22 | Signed verbs and pointers; store accepts only signed committed cuts | Stale attempt writes a pointer; replayed verb; unsigned cut accepted | P2 |
| SEC-B23 | `issue_epoch` with job-bound token and authoritative `squeue -j` | Token reused for another job; cached not-found authorizes | P2-full |
| SEC-B24 | Research profile enforcement on shared clusters | Unsigned research cut; `unpromotable` output promoted without import stage | P1 |
| SEC-B25 | Private (operator-controlled) destinations only | Export to an external destination; release of a PII-trained model | P2 |
| SEC-B26 | Injection canaries; non-LLM blocking evaluator, except a qualified constraint-clause judge (05 §6.1) | Judge-only blocking gate validates outside a qualified constraint clause; canary prompt alters judge score unnoticed; a missed canary in a judged batch does not make the constraint clause MISSING | P2 |
| SEC-B28 | `safety_critical` refused outside validated workflows | Research or standalone fused run reads `safety_critical` data | P0, P2 |
| SEC-B29, B30 | Break-glass IDs single-use; separate OS user for T1 plug-ins | Replay of a break-glass ID; plug-in reads run credentials | P3 |

### 4.5 Statistical guarantees (05)

| ID | Guarantee | Test | Layer | Phase |
|---|---|---|---|---|
| STAT-01 | Gate reports carry power, MDE, `underpowered`, tiers, hashes (05 §3.7) | Report schema test on every gate run | L2 | P0, P2 |
| STAT-02 | Intervals reach nominal coverage; TOST type-I ≤ α at the margin | Monte Carlo on synthetic paired data per interval type (05 §3.4) *(10,000 replicates, placeholder)* | L2 | P0 |
| STAT-03 | Phase-exit TOST parity | Registered run per 05 §4.2 | L6 | P1b, P5, P6a |
| STAT-04 | Per-timestep ratio statistics, procedure (b) | Registered run per 05 §4.3 | L6 | P1b |
| STAT-05 | Family α (Bonferroni) and online FDR wealth | Replay of simulated turn sequences with known true/false improvements | L2 | P2 |
| STAT-06 | Independence tiers and non-LLM blocking rule, with the constraint-clause judge exception (05 §6.1) | VV-GATE-1 | L2 | P2 |
| STAT-07 | Transfer test (05 §6.2) | Probe set per training reward; stored artifact cited in gate reports | L6 | P2 |
| STAT-08 | Detector per-step false-alarm rate, ARL₀, ARL₁ | VV-DET-1, VV-DET-2 | L6 | Promotion out of shadow mode |
| STAT-09 | Contamination at registered precision with known recall | VV-CON-1 | L2, L6 | P1b, P2 |
| STAT-10 | Validator computes required human labels from gate configuration | Configurations from 05 §8 reproduce 783 / 1,304 comparisons | L2 | P2 |
| STAT-11 | G0 never reports parity | Report linter rejects "parity" or "equivalence" in G0 reports | L2 | G0 |
| STAT-12 | Constraint clause exact bound (05 §3.9) | Monte Carlo on synthetic violation indicators: coverage of the exact upper bound at the registered α, including zero observed violations; a zero-violation sample below the n for `bar` does not PASS | L2 | P2 |

STAT-01 is implemented where the gate report is built (task T1-09 of the implementation plan, not published); the clause and interval functions (task T1-07) are covered by L2-E-stats and STAT-02.

### 4.6 Package suites, conformance points, and proposed simulation cases

IDs cited by revision 2 of the project's implementation plan (not published; milestone IDs such as M0b are that plan's). Conformance IDs name an extension point of 01 §9.2 (§3.3); individual tests within a point are `CONF-<point>-<n>`. Proposed IDs are confirmed or retired at the phase gate that first needs them.

| ID | Definition | Layer | Phase |
|---|---|---|---|
| L1-static | Static rules of §3.1: import graph, pickle-load ban, secret scanning, each with a failing fixture | L1 | P0 |
| L2-A | Package A unit and property suite of §3.2 (golden files for one-way-door formats, unknown-major rejection, signature byte flips, path scoping) | L2 | P0 |
| L2-B | Package B unit and property suite of §3.2 (Gaussian KL identity, permutation invariance, finite gradients, selector masks) | L2 | P1a |
| L2-C-ext | C-ext unit suite of §3.2 (wrappers, verb verification, emitter wiring; zero new fields on token partitions) | L2 | P0 |
| L2-E-stats | E-stats unit and property suite of §3.2 (clause status mapping, `plan`, `reestimate` never lowers n) | L2 | P0 |
| L2-E | Package E unit suite of §3.2 (spec and IR validation, resume without writes, lockfiles) | L2 | P2 |
| L2-R-svc | R-svc unit suite of §3.2 (store CAS, owner sections versioned independently, record-first, cached not-found mapped to `Unknown`) | L2 | P2 |
| L2-R-agent | R-agent unit suite of §3.2 (launch and supervision with backoff, timer resume from the last renewal, per-segment MAC, signing, idempotent commits and verbs) | L2 | P1 (launch, MAC), P2 |
| L2-RPC | Agent↔svc authenticated RPC: a token for another key is refused; a stale-epoch commit fails the CAS; a commit with a mismatched manifest hash, MAC list, or signature is refused (02 §2.5; 03 §3.3, §3.4; 06 §5.2) | L2 | P2 |
| L5-research | Research profile end to end on one node (§3.5) | L5 | P0 (skeleton), P1b |
| L5-replay | `--capture-batch` and `--replay` agree bitwise or within the declared tolerance (01 §8.3) | L5 | P1b |
| L5-smoke | A plug-in's single-GPU smoke configuration (01 §9.1 rule 6) | L5 | P1b |
| L5-agent | `gatedtrain-runtime-agent` launched by the launcher under a second OS user (or on a service node) on one node; restart and `adapter_down_max_s` kill path (02 §2.5; 06 §5.2 rule 1) | L5 | P1b |
| CONF-schema | Step-field schema point: unknown major rejected; `TOKEN`-only partitions declare zero fields (01 §5, §9.2) | L3 | P0 |
| CONF-transport | Both contract transports pass one suite: create-only races, atomic replace, I/O error distinct from not-found (02 §2.2, §2.6, §10) | L3 | P0 |
| CONF-contract | Run-contract emission through hook 8a-1 or its C-ext shim: invalid signature rejected, duplicate `action_id` acted on once, hung trainer shows advancing `seq` with frozen progress, `stop` honored; shim and upstream hook emit identical records (02 §2, §10) | L3 | P0, P2 |
| CONF-contract-sft | CONF-contract and CONF-8d rows rerun on the SFT stage (`sft_train`) for the P2 pipeline (design §7.1 P2 exit) | L3 | P2 |
| CONF-8d | Hook 8d or its shim: segment rows equal the consumed rows; segment hashes listed in the next cut; epoch only in the segment header (02 §3; 05 §12) | L3 | P2 |
| CONF-P2-restore | P2 restore conformance pinning `get_latest_checkpoint_path` and the `remove_old_checkpoints` early return at the baseline commit; restore only from a signed pointer (02 §5.7) | L3 | P2 |
| CONF-kernel | B loss kernels: parity against flow_grpo, dancegrpo, diffusionnft on recorded tensors; C-ext and D wrappers equal a direct kernel call (01 §7.1, §9.2) | L3 | P1a (P0 wrapper row) |
| CONF-algorithm | Algorithm point: decision-step worked examples (01 §4.4) and, in P6a, the intervention-rule variants | L3 | P0, P6a |
| CONF-ModelAdapter | `ModelAdapter`: build and shard; the name map round-trips; `rollout_role` weights reach the engine (01 §6.1) | L3 | P1b |
| CONF-StepForward | `StepForward`: frozen-weight rollout/train log-ratio below the family threshold; fused equals unfused (01 §6.2, §7.6) | L3 | P1b |
| CONF-rollout | Rollout engine: `store_step_mask` and `sampler_config_digest` written, `GroupPlan` honored, refit dead-peer test (01 §6.4; 02 §8) | L3 | P1b |
| CONF-env | Environment point for fused vectorized execution: per-iteration guarantees and resets (08 §2.2, §6) | L3 | P6a |
| E2E-SKEL | *(Proposed, pre-P0 exit, subject to open-issues PI-25.)* Walking skeleton: one approved LLM recipe on unmodified NeMo RL runs from a thin spec (a strict subset of the 04 §3 YAML) through NeMo Evaluator on the baseline and trained checkpoints and an E-stats gate on per-item paired scores (05 §3.3) under a registration signed by the LLM pilot user (report carries power, MDE, `underpowered`) to manifests, a lockfile, a Parquet ledger at precision `iteration`, and an impact query that returns the stage's checkpoint with its precision; every stub is marked `skeleton-stub` | L10 | M0b |
| L2-E-skel-compat | *(Proposed.)* Fixture written by the verifier before the walking skeleton: the skeleton's workflow spec files are a strict syntactic subset of the 04 §3 workflow YAML and satisfy the 04 §4.2 required-field rules for a one-stage, one-gate workflow; the same files, unchanged, validate under package E's validator once it exists | L2 | M0b (fixture), P2 (re-run) |
| UX-01 | *(Proposed.)* Time to first gated run for a new user, from onboarding docs to a gate report; measured by at least one human from the pilot user (not a gatedtrain maintainer) at M0b and before P0 exit, then on each new user; target ≤ 1 day *(placeholder)* (open-issues PI-26; design §7.3) | L10, L11 | M0b, P0, P2 |
| UX-02 | *(Proposed.)* Weekly runs by the pilot user completed without maintainer help, reported weekly from P2 (baseline at the skeleton review) | L11 | P2 |
| UX-03 | *(Proposed.)* Pilot-user dry run: an automated pilot with only the onboarding docs reaches a gate report on a CPU path (tiny random model, recorded `results.jsonl` pairs) or a recorded-output path (replay of recorded training outputs and evaluation results); transcript and stumbles filed; run before each increment exit. Complements, never replaces, the human UX-01 measurement | L10 | P0 (increment 1), P2 (increment 2) |
| UX-04 | *(Proposed.)* Error-message checklist: every validator and CLI error names the rule, the field, and the fix | L2 | P0, P2 |
| SIM `runner/stop-at-cap` | *(Proposed, P2.)* The P2 variant of `fence/stop-honored`: when a stage reaches its GPU-hour cap, the runner issues `stop` through the run contract (04 §8.1) and the run stops; the Slurm QOS backstop still stops it if the runner is down (06 B14, §13; INV-09 P2 cap) | L4 | P2 |
| SIM `broker/ghost-unheld` | *(Proposed, P2-min.)* A ghost job created by an ambiguous `sbatch` appears after `Found([])` and a retry; with no hold and no janitor, the runner must not end with two live gangs for one stage. Expected to fail until open-issues PI-14 is decided | L4 | P2-min |

## 5. Environments and budgets

### 5.1 Environments

| Environment | Description | Used by | Provider |
|---|---|---|---|
| Local toy | Workstation or one GPU; research profile, local-mode adapter | L2, L5 development | Contributor |
| CI CPU | Container runners; TLC, simulation, unit tests | L1–L4 | Maintainers (project CI) |
| CI GPU | Nightly pool; assumed budget **64 GPU-hours per night** *(placeholder)*: conformance matrix, kernel parity, upstream drift, smoke runs | L2, L3, L5, §6 | Project lead (Appendix A, VP-OI-3) |
| Staging Slurm partition | **8 exclusive nodes** *(placeholder)*, prolog firewall, per-team test accounts and QOS, service node for a test runner, the site's Slurm query cache if it runs one, staging health-service deployment, a scratch filesystem that can be degraded | L7 (A/B), L8, L9, L10 | Operator's site (proposed G0 prerequisite, VP-OI-1) |
| Pilot-user partition | The pilot users' production allocation | L6, L10, L11 | Pilot users per G0 |
| Kubernetes staging | Needed for the Kubernetes plug-in and P3 services | P3 | Operator's site, at P3 entry |

### 5.2 Fixtures and lineage

- **Models:** small open models only in CI (a sub-1B LLM for GRPO recipes and a small diffusion model for B and D smoke tests) *(choice at P0 kickoff)*, served from the site-local mirror (06 §9); full-size references (SD3.5 Medium or Qwen-Image, 05 §4.2) only in L6 and L7.
- **Prompt sets and evaluation item sets:** frozen, content-addressed, registered as A manifests; registrations reference item sets by hash in `eval_distribution` (05 §3.7).
- **Recorded tensors:** reference tensors from flow_grpo, dancegrpo, diffusionnft for B parity, stored with the reference commit.
- **Synthetic ledgers and lineage graphs** with known answers for VV-LIN (§8).
- **Every fixture is a lineage artifact;** a test result cites fixture hashes, so a fixture change invalidates cached results.

### 5.3 Cost estimate per layer *(estimates)*

| Layer | Compute | Notes |
|---|---|---|
| L1, L2 (CPU), L4 | CPU only | TLC bounded instances may need a large-memory runner |
| L2 (GPU), L3, L5 | Within the nightly CI GPU budget | |
| L6 | Set by the G0 cost tables: the P1b GPU-hour budget (design §3.1, EM3-6) | G0 cap 2,000 GPU-hours (design §3, placeholder in the design) |
| L7 | P0 A/B ≥ 50 pairs; P1b ≥ 3 run pairs × 2 arms × ≥ 20 iterations | About 1 staging-day each *(estimate)* |
| L8 | Mostly CPU and staging; one red-team week per phase on shared clusters *(estimate)* | |
| L9 | Staging partition; about 2–3 staging-days per phase exit for the P1b and P2 subsets *(estimate)* | |
| L10 | Staging, then pilot-user partition; small models except EXIT-P2-1 pipelines | |

## 6. Upstream drift

| Job | Target | Contents | Trigger | Triage owner |
|---|---|---|---|---|
| DRIFT-N0 | Pinned NeMo RL of each release (baseline `4f39fd8`) | Full C-ext suite, conformance kit, P2 restore test | Every C-ext merge request and release | Project lead |
| DRIFT-N1 | NeMo RL `main` nightly | Hooks 1, 4, 5, 6, 8a-1, 8d tests; C-ext shims for 8a-1 and 8d; hook-1 safety (`wrapper.py:156-161` keyword passing, `ClippedPGLossFn` signature); P2 restore test (`get_latest_checkpoint_path`, `checkpoint.py:682-691`); `remove_old_checkpoints` no-op with retention keys unset | Nightly | Upstream contact for NeMo RL |
| DRIFT-V0 | Pinned verl and verl-omni (path V) | verl plug-in emitter, `_save_checkpoint` wrapper, `resume_mode = resume_path`, synchronous-save requirement (02 §2.7), conformance kit | Every adapter merge request | Path-V adapter maintainer |
| DRIFT-V1 | verl-omni `main` and its pinned verl nightly (path V) | Same as DRIFT-V0 plus `verl.plugins` entry-point loading | Nightly | Path-V adapter maintainer |

**Triage rule.** A DRIFT-N1 or DRIFT-V1 failure is triaged within two working days *(placeholder)* into: (a) our shim or test is wrong (fix in C-ext or the adapter); (b) an intentional upstream change (adapt and, if a hook contract changed, raise it with the upstream maintainers; no upstream review capacity is assumed); (c) flaky (quarantine with a waiver, §11). Nightly failures never block a release built on the pinned version; they block the next pin bump.

## 7. Release qualification

| Package | Must pass before tagging |
|---|---|
| A | L1 static rules; L2 incl. golden files for one-way-door formats; both transports on one suite; N/N−1 reader compatibility (07 §11 item 1); no torch import |
| B | L2; CONF-kernel parity against the three references; VLA worked example (P1a) |
| C-ext | DRIFT-N0; hook-1 safety; zero bytes per token; PRF-M3 on the release candidate when a hook or wrapper changed; P2 restore test |
| R-agent | Contract conformance (02 §10); SIM suite for its phase; adapter restart and timer resume; C16 on staging; per-cluster canary for 48 h *(estimate, 07 §11)* with `adapter_fault` rate no worse than the previous version; lockfile pin and rollback rehearsal (runbook R9) |
| R-svc | SIM suite; TLA+ models of its phase checked; SEC-B12, B13, B15, B21, B22, B27 (and B23 for P2-full); backup-restore drill of the P2 store |
| E | Reference workflows produce identical lineage and gate outcomes (the conformance oracle, 04 §8.1); VV-GATE-1; VV-LIN-1 |
| E-stats | STAT-01, STAT-02, STAT-05, STAT-10; upstream fixes to NeMo Evaluator tracked separately (05 §3) |
| D | DRIFT-N0 against its pinned NeMo RL; conformance kit; P1b exit evidence for the first release, smoke parity on recorded tensors after |

**Plug-in admission (allow-list).** A digest enters the allow-list (B17) only through a reviewed merge request whose CI shows: static manifest validated without importing code (01 §9.1 rule 2); conformance kit for its extension point; single-GPU smoke config under the research profile; safe formats (B9); secret scan (B10); no code execution (B18); for T0 code, two-person review, SBOM, and build provenance (06 §16); for LLM or VLM judges, injection canaries (B26).

## 8. Verifying the verifiers

| ID | Verifier | Seeded fault | Required outcome |
|---|---|---|---|
| VV-STAT-1 | TOST parity (STAT-03) | Known parity-breaking changes on a toy diffusion recipe: KL term sign flipped, timestep index off by one, σ of the SDE step scaled by 1.1 *(choices at P1a)* | TOST fails on at least the registered evaluator, at the planned power, in simulation replay and in one real run per change |
| VV-STAT-2 | TOST type-I control | Candidate equals reference exactly (A/A) and a candidate at Δ = δ | A/A passes at ≥ planned power; Δ = δ passes at most at rate α |
| VV-STAT-3 | B parity tests | Mutation testing of B kernels (operator, constant, and index mutations) | Every surviving mutant is reviewed; kill rate target ≥ 90% *(placeholder)* |
| VV-DET-1 | Reward-hacking detectors | Known hack in a toy run (KL disabled, or a reward that pays for rendered-text area) | Alarm within the registered ARL₁ bound; detection delay reported |
| VV-DET-2 | Detector false alarms | Replay of human-labelled unhacked reference runs (≥ 5, 05 §7) | Per-step rate and ARL₀ meet target; no run-level claim from fewer than about 30 runs |
| VV-CON-1 | Contamination check | Planted exact, NFKC-variant, near-duplicate (MinHash), perceptual-hash, and template-slot items at known rates in a training artifact | Recall at least the calibrated recall; precision at least the registered 0.95 (05 §10.2) |
| VV-LIN-1 | Impact query | Synthetic ledger and lineage graph with retries, new epochs, rollbacks, dropped and exposed rows, and a known answer set | Exact set match for `consumed`; `include_exposed: true` adds exactly the exposed rows; edge precision reported correctly |
| VV-GATE-1 | Workflow validator | Specs violating each rule: T0/T1-only blocking evaluator, LLM-only blocking judge (on a quality clause, or on a constraint clause failing any of 05 §6.1 (a)–(e)), missing margin, `alpha` inside a family, uncorrected peeking, unfunded `labeling_usd` | Every spec rejected with the named rule |
| VV-SEC-1 | Security baseline | Red-team cases of §4.4 | Every case fails to bypass; a case that succeeds is a phase blocker |
| VV-FENCE-1 | TLA+ models | Remove one guard at a time (for example the current-attempt check in `issue_epoch`, the epoch CAS on commit, `not_before` on recovery holds, the pin check before GC) | TLC reports a violation of the matching invariant; a guard whose removal violates nothing is reported as either redundant or under-modelled |
| VV-FENCE-2 | Deterministic simulation | Same guard removals in code under the SIM suite | Each named conformance test fails; seeds that trip it are kept as regression seeds |
| VV-PERF-1 | Overhead harness | A/A run (hook off vs hook off) and a seeded 5% slowdown in one stage | A/A interval contains 0; seeded slowdown detected |

## 9. Fault injection

### 9.1 Fault-injection capabilities assumed of a site

- **Software faults:** fault points that hang a collective (all-reduce, send/receive), crash a rank, or make one rank a compute straggler (optionally followed by a hang), armed after a configurable delay on a chosen rank; a launcher that supports held submission and requeue or dependency-based resume.
- **Hardware faults:** injected interconnect faults (for example link-down errors) and node reboots.
- **Syslog injection:** synthetic fatal-error messages of the kind the cluster health service detects.
- **Verification harness:** polls the scheduler (`sacct`, `scontrol show job --json`) for state, `restart_cnt`, and the requeue flag; checks fault signatures in logs; checks the node-list transition (a previously allocated node appears in the next attempt's exclusion list) against the health service's per-attempt status; covers requeue and release scenarios and random faults.
- **Health-service coupling:** test jobs can be marked so that a staging health service acts on them (for example for automatic resume), and its detection delays can be shortened for faster tests.

### 9.2 Mapping chaos tests and FMEA rows to mechanisms

| Test | Mechanism | Site capability (§9.1) or add |
|---|---|---|
| C5 IB flap during refit | Hardware-injected link-down on one node during refit; software proxy: NCCL hang fault point armed in the refit collective | Site hardware-fault capability; add an IB link fault and a refit-scoped fault point |
| C6 SIGSTOP Ray head | `srun --overlap kill -STOP` on the head | Add |
| C7 Kill a storage-unit node | Reboot the node hosting storage units | Site capability |
| C8 Truncated latest cut | Truncate one shard in the committed prefix (test identity with adapter-user rights) | Add |
| C9 Poisoned reward plug-in | Unlisted digest in lockfile; listed plug-in returning shifted rewards | Add |
| C11 Run to the time limit | Short `-t` in the sbatch script | Add (configuration only) |
| C12, C22 Node fault seen by the health service | Hardware-injected NVLink faults; check the node-list transition | Site capability |
| C13 Fleet collapse to 25% shards | Segfault fault point in rollout workers on 75% of shards | Add (segfault fault point in the test-only module) |
| C14 Budget 100% | Tiny QOS limit on the test account; runner stopped mid-stage | Add |
| C15 Lustre brownout | Storage administration throttles a scratch MDS; fallback: delay-injecting FUSE layer over the contract directory *(choice at P2)* | Add (needs the site's storage administrators) |
| C16, C17 Adapter crash and bad release | `kill -9` loop on the adapter; fault-injecting adapter build pinned on the canary cluster | Add |
| C18, C19 Requeue after cancel; stale requeue | `scancel` workflow then `scontrol requeue <job>`; requeue of a finished attempt | Add (P2-full) |
| C20 `NODE_FAIL` while alive | Administrator stops `slurmd` on one node while processes run | Add (needs site administrator rights) |
| C21 Runner down plus administrator hold | Stop the runner service; `scontrol hold` on an unrelated job | Add (P2-full) |
| C23 Relay outage | Stop the relay processes | Add (only where a relay is used) |
| C24 Storage outage at 4× interval | Make the checkpoint prefix unwritable or exhaust quota | Add |
| C1–C4 Store blackout, slurmdbd down | Firewall drop to the store endpoint; stop `slurmdbd` | Add (P3; C4 needs site administrator rights) |
| C10 Shared judge outage | Proxy injecting latency and errors in front of the judge | Add (P4) |
| FMEA: crash in a stage (P2-min) | Segfault fault point from the test-only module on a NeMo RL rank | Add; expect stage failure, RCA, operator R14 |
| FMEA: hung run (P2) | NCCL all-reduce hang fault point from the test-only module; frozen progress counter | Add; expect runner `stop` beyond the declared timeout (04 §8.1) |
| FMEA: `issue_epoch` refusal | Call with stale `restart_count` or a superseded job | Add (P2-full) |
| FMEA: straggler | Compute-straggler fault point from the test-only module | Add; expect no recovery action, only telemetry |

**Test-only fault points.** Fault points armed inside a training framework's own collectives do not reach NeMo RL or Ray workers. Proposed: a test-only C-ext module configured through documented environment variables, armed only in test lockfiles and never allow-listed for production (VP-OI-4).

### 9.3 Keeping fault-injection runs from triggering the health service unexpectedly

In P2-min framework jobs are untagged and `--no-requeue`, and the health service is for detection and RCA only (D8); but nothing yet stops the health service or its remediation logic from acting on them (open-issues R9-OI-1), and how a site's health service treats jobs that carry no automatic-resume marker must be confirmed per site.

1. Run campaigns only on the staging partition against a staging health-service deployment; never shorten detection delays (§9.1) on a shared production health service.
2. Framework jobs never carry a marker that asks the health service to resume them automatically.
3. Announce each campaign to the health-service operator with its job IDs; keep the job list as a campaign artifact.
4. Every test asserts: `restart_cnt` 0 and requeue flag off (`scontrol show job --json`), no health-service exclusion or requeue recorded for the job in the job monitor, and NVRx `attempt_index` 0 (07 §8.3).
5. Hardware faults need reboot rights on test nodes; grant them only on staging nodes.

## 10. End-to-end acceptance scenarios

Each scenario stores its report, logs, and registrations as lineage artifacts and fails on any unmet check.

### 10.1 E2E-G0
- **Preconditions:** pilot users identified; G0 feasibility registration (05 §4.1) signed.
- **Steps:** run both arms to N_fix or the cap; fill both cost tables; record the dated site prerequisites.
- **Checks:** feasibility report per 05 §4.1 (descriptive, STAT-11); cap respected; P2-min or P2-full recorded; R9-OI-1 Slurm behavior confirmed per target cluster.
- **Pass:** EXIT-G0-1..4 checklist recorded by the project lead. **Artifacts:** feasibility report, cost tables, prerequisite register.

### 10.2 E2E-P0
- **Preconditions:** hooks 1, 4, 5 merged or pinned as plug-ins; A, E-stats, C-ext skeleton released to staging.
- **Steps:** run every existing LLM GRPO recipe on `grpo_train_sync` with and without hooks; run the A/B harness; run the conformance kit v0 on NeMo RL; run the P0 security checks.
- **Checks:** EXIT-P0-1..5; SEC-B9, B10, B11, B18, B19, B28; STAT-01, STAT-02; VV-PERF-1.
- **Pass:** all blocking checks pass. **Artifacts:** A/B report with per-stage intervals, conformance report, security checklist.

### 10.3 E2E-P1a
- **Preconditions:** recorded reference tensors for the three references.
- **Steps:** run B parity and kernel conformance; run VV-STAT-3.
- **Checks:** EXIT-P1a-1; worked VLA example; mutation report reviewed.
- **Pass:** all parity tests pass. **Artifacts:** parity report with reference commits.

### 10.4 E2E-P1b (or E2E-P6a)
- **Preconditions:** registration signed (05 §4.2); P1 security items on the target cluster (SEC-B1–B8, B14, B17, B20, B24); GPU-hour QOS within the P1b budget.
- **Steps:** seeds per the power plan for both arms; per-timestep statistics; throughput pairs; chaos C5 (watchdog), C7, C8, C9, C11, C13 on staging; VV-STAT-1 change set.
- **Checks:** EXIT-P1b-1..4; PRF-M4–M10; chaos pass criteria; zero duplicate gangs.
- **Pass:** all metrics pass TOST by intersection-union; throughput lower bound ≥ 0.9. **Artifacts:** parity report, throughput report, interval registration, chaos log. P6a replaces EXIT-P1b with EXIT-P6a and adds the 08 §7.5 simulator-failure test.

### 10.5 E2E-P2MIN
- **Preconditions:** per-team runner on a service node; P2 store run by the staging operator; P2-min security items; staging health service configured per §9.3.
- **Steps:** (1) submit the LLM pilot user's SFT → GRPO → gate workflow; (2) gates evaluate with power and MDE; (3) run the impact query from a training shard; (4) submit a workflow violating gate independence; (5) attempt a restricted-sample route to an external reward and to telemetry; (6) inject the segfault fault point (test-only module, §9.2) into the GRPO stage and confirm no Slurm requeue; (7) cancel during submit; (8) call `retry_stage` while the old job is still queued, then after it leaves the queue; (9) requester approves own gate; (10) attempt an external export.
- **Checks:** EXIT-P2-1..8; INV-01, INV-02, INV-07, INV-20; C14, C16; SEC P2 items.
- **Pass:** all checks pass; the health service took no remediation action. **Artifacts:** workflow lineage, gate reports, impact-query output, audit log extract with WORM anchor.

### 10.6 E2E-P2FULL
- **Preconditions:** E2E-P2MIN passed; P2 requeue TLA+ model checked before runner code; `MaxBatchRequeue` verified on the cluster (03 §3.3).
- **Steps:** C18, C19, C21; Slurm requeue of a running stage after a hang from the test-only fault-point module; runner stopped during requeue.
- **Checks:** EXIT-P2-6F; INV-03..05; SEC-B23; no training step from a refused job.
- **Pass:** all refusals exit 79 before `srun`. **Artifacts:** `issue_epoch` decision log, janitor audit.

## 11. Reporting

**Verification dashboard.** Panels: pass/fail per test ID and phase; nightly trend for CI GPU, conformance matrix, DRIFT-N1/V1; flaky-test quarantine list; open waivers by expiry; chaos-test coverage (C-number × simulation × staging); must-measure items recorded versus pending (08 §8); GPU-hours spent per layer against the placeholders of §5.

**Per-phase verification report** (an immutable artifact attached to the phase-exit decision): scope and design revision; tests run with environment, fixture hashes, and results; exit criteria with evidence links; placeholders replaced by measurements; waivers; Appendix A changes; sign-off by the P0 maintainer and the phase's maintainer role.

**Waivers.** Each waiver names the test ID, the reason, the risk, a compensating control, an owner role, and an expiry no later than the next phase gate. Waivers are not accepted for INV-01, INV-03, INV-06, INV-08, SLO-03, or any SEC red-team case that succeeded.

## 12. Schedule and effort

Consistent with design §7.2; verification effort is an *(estimate)* inside those numbers, not additional (see VP-OI-8).

| Window | Verification work | Effort *(estimate)* |
|---|---|---|
| Increment 1: G0 (3–4 weeks) | E2E-G0; this plan revised; staging partition and CI GPU budget requested; plan for the test-only fault-point module | about 1 engineer-week |
| Increment 1: P0 (8–10 weeks) and P1a | L1 static rules; L2 for A, E-stats, C-ext; conformance kit v0; A/B harness and PRF-M1–M3; STAT-02; DRIFT-N0/N1; P1a parity and VV-STAT-3; stage-lifecycle TLA+ model drafted | 4–10 engineer-weeks |
| Increment 2: P1b (parallel with P2) | L6 parity runs; PRF-M4–M10; P1 security checks; test-only fault points; staging chaos subset; DRIFT-V* if path V | 5–12 engineer-weeks plus diffusion expertise for registrations |
| Increment 2: P2-min (6–8 weeks) | SIM suite before runner code; E2E-P2MIN; VV-LIN-1, VV-GATE-1, STAT-05–STAT-10; SEC P2 items; C14, C16 | 3–8 engineer-weeks |
| Increment 2: P2-full (+about 2 weeks) | P2 requeue TLA+ model before runner code; C18, C19, C21; E2E-P2FULL | +about 1 engineer-week |

## Appendix A. Open-issues entries (recorded in open-issues.md)

| ID | Issue | Gate | Owner | Proposed direction |
|---|---|---|---|---|
| VP-OI-1 | No staging Slurm partition is a G0 site prerequisite; chaos tests (07 §9) and P2 acceptance need one with exclusive nodes, a degradable filesystem, and administrator actions (`slurmd` stop, `slurmdbd` stop) | G0 | G0 maintainer with the operator's site | Add a staging partition (size placeholder 8 nodes) to the G0 site prerequisites, or record that P1b/P2 chaos runs use the dedicated partition |
| VP-OI-2 | Fault-injection campaigns on framework jobs can trigger health-service remediation; extends R9-OI-1 to testing. How a site's health service treats framework jobs, which carry no marker requesting automatic resume, must be confirmed per site | Before the first staging campaign (P1b) | Health-service operator with the P0 maintainer | A staging health service in detection-only mode for the staging partition, or an explicit health-service exclusion for framework test accounts |
| VP-OI-3 | The design states nightly matrices (01 §9.2, 07 §11) but no CI GPU budget | P0 entry | Project lead | Declare a nightly CI GPU budget (placeholder 64 GPU-hours) in the increment-1 scope |
| VP-OI-4 | Fault-injection tooling that arms faults inside a training framework's own collectives does not reach NeMo RL or Ray workers | P1b | Project lead | Test-only fault-point module in C-ext with documented configuration, excluded from production allow-lists |
| VP-OI-5 | Ground truth for verifying detectors and contamination (≥ 5 human-labelled unhacked runs, about 30 for run-level claims, calibration pair sets) has no owner or budget in the MVP | Detector promotion (P1b–P4) | Evaluation maintainer | Budget labelling and runs per family, or keep detectors in shadow mode with that stated |
| VP-OI-6 | 03 §12.1 says P2-min is "covered by the stage-lifecycle model" but does not require that model to be checked before P2-min runner code | P2-min code start | Control-plane maintainer | Require the stage-lifecycle model without requeue checked before P2-min runner code |
| VP-OI-7 | P2 has no SLOs (07 §2), so P2 production behavior has no recorded baseline for P3 | P2 | Operations maintainer | Record a subset of 07 §8.2 SLIs as report-only in P2, without targets or paging |
| VP-OI-8 | Design §7.2 effort estimate does not separate verification effort; staging operation and red-team weeks may not fit | G0 (increment 2 sizing) | Project lead | Confirm verification effort is inside the estimates, or add it explicitly |
| VP-OI-9 | No supported NeMo RL version window is stated for C-ext in the MVP (07 §11 covers P3 skew) | P0 exit | Upstream contact for NeMo RL | State the pinned version per release and how many older pins C-ext supports |

## Appendix B. Sources

**Design set (revision 9):** `design.md` §3 (G0), §4 (packages), §5 (hooks), §5.1, §6 (D8, D18, D19), §7.1–§7.3; `01-contracts-and-algorithms.md` §3.1, §4.4, §8, §9.1–§9.2; `02-run-contract-and-checkpoints.md` §2.5–§2.7, §5.6–§5.7, §6, §7, §8, §10; `03-control-plane.md` §1.3, §3, §12.1, §12.2, §13; `04-workflow-and-composition.md` §4.2, §4.3, §8.1; `05-evaluation-and-lineage.md` §3–§8, §10, §11.3, §16; `06-security.md` §4, §16; `07-operations.md` §2, §3, §4.8–§4.9, §7, §8, §9, §10, §11; `08-families-execution-and-performance.md` §7, §8; `open-issues.md`.

**Operations context:** the cluster health service roles of 03 §10.

**Fault-injection tooling:** the capabilities of §9.1, which a site provides.
