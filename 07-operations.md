# 07. Operations: Services, Reliability, SLOs, and Runbooks

**Status:** The design set is at revision 21; this document's design content was last changed in revision 9. A prototype (`gatedtrain` 0.3.0) implements revision 19. Frozen until G0 closes; deferred items in `open-issues.md`. This document remains **Direction — detailed review after G0** (R8-20), except that the P2 operations content is normative for the MVP: the P2 rows of §2, §4.8, the P2 chaos tests in §9, and the P2 runbooks in §10. Everything else describes P3 and later. Overview: `design.md`.

**Primary findings:** ARCH-9, OPS-4, OPS-5, OPS-6, OPS-7, FMEA-6, FMEA-7, FMEA-9 (round 1, §14); OPS2-7, OPS2-11, FMEA2-9, and the operational parts of R8-5, R8-6, R8-7, R8-11 (round 2, §15); the operational consequences of R9-1, R9-2, R9-8, and R9-9 (round 3, last section).

**P2-min and P2-full (R9-2).** P2-min is the default (§4.8): `--no-requeue`, operator `retry_stage` (R14), the health service for detection and RCA only, no `issue_epoch` or janitor. Rows and tests marked "P2-full" apply only if the LLM pilot user confirms in writing that it needs automatic recovery in P2.

**Baseline verified against:** NeMo RL `4f39fd8` (local clone, 2026-11-05) and nvidia-resiliency-ext `86cf39c7` (local clone, 2026-10-29). Numbers marked *(estimate)* are planning values to be replaced by measurements from the pilot users' runs.

## 1. Scope and principles

This sub-design says who runs each service, how it fails, what service targets an operator can set for it, and how operators recover it. Protocols (leases, fencing, supervisor rules, `issue_epoch`, the contract transport, health-service detect/decide/act) are defined in 02 and 03 and only summarized here.

1. **Phase-honest.** P2 services (per-team workflow runner, the P2 store, lineage tables, retention job; in P2-full also `issue_epoch` and the janitor) are **non-production**: no paging, no SLO, manual runbooks, notifications to the run owner. The P2 store has a designated operator, the framework operator, in both variants (R9-2). Their behavior under failure is still specified and tested (§4.8, §9). Production SLOs and paging start with P3, whose exit criterion is "SLOs met for four weeks on the pilot users' runs" (overview §7.1).
2. **Fail-static with a bounded fence.** If a control-plane service is down, running workloads continue; no new placements, stage transitions, or migrations happen. When a run's adapter loses its lease it stops commits, uploads, and deletes; compute and refit continue for at most `fenced_compute_max_s`; the adapter then pauses the run in place with `pause_and_checkpoint(then=hold)`, keeping its allocation, for at most `fenced_hold_max_s` (released by `pause_and_checkpoint(then=continue)` after a same-epoch renewal); then it stops the run. `stop` and safety pauses are always honored. Fenced time counts against budgets (§6). `safety_critical` runs use `fenced_compute_max_s = 0` (03, R8-5).
3. **One hardware owner.** The cluster health service owns hardware detection and node verdicts; the framework owns in-run decisions (overview D8). Nothing in this file adds a second node-exclusion path.
4. **Infrastructure is not a run fault.** Exits caused by the store, the adapter, a filesystem stall, or fencing never count toward crash-loop breakers and never mark a cut `suspect` (R8-6, R8-7).
5. **Reuse telemetry.** All signals are built on `nemo_rl/telemetry` span groups and nemo-lens; there is no new telemetry contract (§8, ARCH-9).
6. **Measured before promised.** Every SLO target below is an initial target, reviewed after the first four weeks of P3 data.

## 2. Service inventory

Owner roles are roles at the operator's site, not names. "Framework operator" means whoever runs the gatedtrain services at a site; "Platform" means whoever operates the site's shared cluster services (overview §11). Packages: R = `gatedtrain-runtime-agent` (in-allocation adapter) and `gatedtrain-runtime-svc` (store, broker, `ControlAPI`, `issue_epoch`) per R9-9, E = workflow layer, G = P3 control plane (overview §4).

| Service | Owner role | Deployment | HA approach | Backup / PITR | RTO / RPO | Phase |
|---|---|---|---|---|---|---|
| Linearizable metadata store (leases, epochs, intents, fencing tokens, cut commits, workflow state) | Platform (store operator; a P3 entry-gate prerequisite, R9-1); framework operator owns the schema | Platform-run linearizable store, 3 or 5 voters across failure domains (03 §2) | Quorum replication; clients retry with bounded backoff | Continuous WAL/snapshot, PITR 7 days; daily logical export | 15 min / 0 committed writes | P3 |
| P2 workflow database / P2 store (workflow records, epochs, cut commits, lineage tables) | Framework operator (designated store operator in P2-min and P2-full, R9-2) | Single primary, synchronous durability, no asynchronous replicas (03 linearizability rules, R8-18) | None; restart | Daily backup plus binlog PITR 7 days | Best effort / 0 committed writes | P2 |
| Workflow runner (P2), with `issue_epoch` endpoint in P2-full | Framework operator | One runner per team as a service on a **service node** (not a login node), Slurm privileges limited to `sbatch`, `scontrol release`, `scancel` on the team's account (06 B12); submits `--no-requeue` (P2-min) or its own `sbatch --hold` (P2-full); queries Slurm status through the site's Slurm query cache where one exists, except that `issue_epoch` authorizes with a direct `squeue -j` | None; restart and resume from the database | Database is the record | Best effort / last completed stage | P2 |
| Broker janitor (P2-full inside the runner; P3 inside the broker) | Framework operator | Periodic task: releases only holds the framework placed (`held_by_broker` in the record and a matching Slurm hold reason), never before a recovery hold's `not_before`; releases a self-held job only if `issue_epoch` would succeed; cancels ghost jobs whose nonce matches no record; reports other holds (03 §3.2, §6.2) | Runs with its host; idempotent | None (stateless over the store) | Best effort (P2-full); 15 min (P3) | P2-full |
| Run adapter (`gatedtrain-runtime-agent`) | Framework operator | One process per run attempt on the Ray head under its own OS user, holding no store credential and reaching `gatedtrain-runtime-svc` only through its authenticated RPC (R9-9); **supervised by the launcher** (restart on exit); version pinned in the lockfile; releases canaried per cluster (§11) | Launcher restart; re-acquires the lease only with the same epoch (03 §4.5) | n/a | Restart ≤ 1 min *(estimate)* | P2 (contract, verbs), P3 (lease) |
| Contract relay (**unnecessary once the agent uses the `gatedtrain-runtime-svc` RPC**, R9-9; not built for P3) | Framework operator with platform | Service-node process, or a site query service, for P2 clusters whose compute nodes cannot reach the runner; reachability is a cluster-profile admission check (R8-11) | 2 replicas per cluster behind one address | Stateless | 15 min / 0 | P2 (only where needed) |
| Workflow layer services (P3+) | Framework operator | K8s Deployment, leader election via store lease | Active/standby; reconcile from store on start | State lives in the store | 10 min / 0 | P3 |
| Placement broker | Framework operator | K8s Deployment, 2 replicas, store lease; refuses tagged placement on clusters without the health-service guard | Active/standby; fail-static when down | State in the store | 15 min / 0 | P3 |
| Run supervisor | Framework operator | K8s Deployment outside every run's Ray cluster, one per cluster, 2 replicas, store lease | Active/standby; rate-limited reconciler (03 §5) | State in the store | 5 min / 0 | P3 |
| Budget service | Framework operator | P2: hard GPU-hour cap computed in the runner, Slurm QOS backstop; P3: service beside the workflow layer emitting burn alerts and budget intents (§6) | P2: QOS enforces even when the runner is down; P3: stateless over the store | Usage rows in the store and lineage | P2 cap: n/a (QOS); P3: 30 min / 0 | P2 (cap), P3 (runtime) |
| Health-service adapter (remediation plan → framework intent; diagnostics trigger on framework cancel) | Health-service operator (prerequisite at the P3 entry gate, R9-1), with the framework operator for the intent schema | Health-service component that consumes remediation plans and framework cancel events | The health service's own (stateless consumer) | Redelivery by the site's event transport | 30 min / 0 | P3 |
| Health-service job-controller guard | Health-service operator (P3 entry gate) | Wherever the site's job controller runs, before it dispatches any action; verifies the admin-set marker (account or QOS) and the store nonce (03 §10.2) | As the job controller | n/a | As the job controller | P3 |
| Store-side retention job | Framework operator | Scheduled job per artifact root; **dry-run mode** prints the delete set; every delete first moves the cut to `retiring` by compare-and-set (skipping pinned cuts, 04 §4.3) and re-checks the cut history, gate pins, and published-artifact references at delete time | Single instance per root with a store lock | Deletion log in lineage; versioned buckets where available | n/a / 0 wrong deletes | P2 (dry-run only), P3 (deleting) |
| Plug-in allow-list (P1–P2) / plug-in index (P3+) | Security owner (06) with framework operator | P1–P2: digest list changed by reviewed merge requests; P3+: index service, 2 replicas | List: Git; index: stateless over the artifact store | Git history; index rebuildable from signed manifests | List: n/a; index: 30 min / 0 | P1, P3 |
| Artifact store (checkpoints, manifests, cross-stage artifacts) | Platform (storage) | Object store or Lustre, content-addressed layout (02) | Storage system's redundancy | Versioned buckets; immutable manifests | 1 h / 0 for committed cuts | P2 |
| Lineage store and OpenLineage events | Framework operator (schema, writer, query); platform (storage) | P2: OpenLineage-format tables in the workflow database, ledger as run-partitioned Parquet queried with DuckDB; P3: Marquez and Iceberg (05) | P2: database and object store; P3: table snapshots | P2: database backup and Parquet in versioned storage; P3: PITR 30 days | 4 h / 15 min events; 0 committed ledger files | P2, P3 |
| Audit log | Security owner | Append-only table with a periodic WORM anchor (06) | As the workflow database | WORM anchor | 4 h / last anchor | P2 |
| Partner services | Composition user team | Ray Serve or Triton pools per partner version | ≥ 2 replicas; scale to zero during gate waits | Stateless; pinned by digest | 30 min / n/a | P10 |
| Shared reward and judge services | Reward-service owner per service | Inference pools behind a gateway, versioned by digest | ≥ 2 replicas; circuit breakers (§5.3) | Stateless | 30 min / n/a | P4 (P2 for external rewards with injection canaries, 06) |
| Preference store | Human-feedback owner (P7) | Transactional DB with consent fields (05, 06) | Primary/replica | PITR 30 days; deletion log | 4 h / 5 min | P7 |
| Sandbox for model-generated code | Security owner with platform | Per-cluster approved sandbox (06) | Stateless pool | n/a | 1 h; code-executing envs blocked while down | P8 |

**P2 statement.** P2 services carry no SLO and no pager; runbooks are manual (§10). A run that needs production guarantees waits for P3. Releases from P2 are private only (overview §7.1).

**P3 scope and the no-P3 end state (R9-1, EM3-5).** P3 is a separate scope decision. Without P3, the end state is P2 plus the health service as the site runs it: no production SLOs, no paging for the framework services, and operator-driven recovery. The operator's P3 responder arrangement, the store operator, and the health-service adapter and guard prerequisites are checked at the P3 entry gate, not at G0.

**P3 alert response (guidance for operators).** An operator who runs the P3 services in production needs a primary and a secondary responder for the framework services, with an interrupt load that is highest in the first P3 quarter and falls as runbooks are automated. The store operator's alert response is part of the store prerequisite at the P3 entry gate. The health service's alert response is unchanged except for the guard and adapter, which the health-service operator runs. This effort is outside the MVP estimate in overview §7.2, and supervisor enforcement on a cluster (03 S5) does not start until the responder arrangement is in place.

## 3. Failure-mode analysis (FMEA)

S = severity (1 cosmetic … 5 data loss or duplicate gang), L = likelihood per month at P3 scale (1 rare … 5 frequent). S×L ≥ 12, or S = 5, requires a tested automatic response and a chaos test (§9). Values are *(estimates)*.

### 3.1 Per component

| Component | Failure mode | Detection | Blast radius | Automatic response | Residual risk | S×L |
|---|---|---|---|---|---|---|
| Metadata store | Unavailable | Store probes from every cluster | All control actions; runs keep computing | Adapters fence: no commits, uploads, deletes; compute and refit continue ≤ `fenced_compute_max_s`; pause in place ≤ `fenced_hold_max_s`; then stop. Supervisor holds all resubmits; recovery after return is an operator-acknowledged staggered release (§4.2) | Outage longer than compute + hold window stops runs | 5×2 = 10 |
| Metadata store | Latency spike | p99 latency SLI | Renewals near TTL | Renewal margin ≥ 3× p99; alarm at 50% of TTL | False expiry if margin mis-set | 4×3 = 12 |
| Contract relay (P2 only where used; no relay from P3, R9-9) | Down on a cluster | Per-cluster reachability SLI | All runs on that cluster | Treated as a store outage for that cluster; new placements refused by the admission check | Single relay pair per cluster | 4×2 = 8 |
| Run supervisor | Resubmit storm after a blip | Resubmits/h SLI per bucket | Many runs restart together | Two signals plus fencing bound (03 S1, S7); separate buckets: ordinary, bulk store-outage recovery (operator-acknowledged), hardware-evidenced (scaled by node count) | Slow recovery under real mass failure | 5×2 = 10 |
| Run supervisor | Old job scheduler-terminal but processes alive (e.g. `NODE_FAIL` while computing) | Scheduler state vs heartbeat file | Two gangs for one run | Release a replacement only after the fencing bound (lease expiry + `fenced_compute_max_s` + `fenced_hold_max_s` + `stop_grace_s` + pause bound) or confirmed-dead processes; `NODE_FAIL` is unconfirmed until then | Delay equal to the bound | 5×2 = 10 |
| Run supervisor | `find_by_key` false negative | `unknown` count | Duplicate gang | `unknown` blocks resubmit (03 §6.4) | Long accounting outage delays recovery | 5×2 = 10 |
| Run controller | Hung but adapter alive | Readable heartbeat with frozen progress counter beyond the phase's p99 budget | One run | Adapter withholds renewal after a stack capture; supervisor recovers | Slow phases outside the measured p99 | 3×3 = 9 |
| Shared filesystem | Lustre metadata brownout: contract files unreadable, saves slow | Contract-file read-failure rate; I/O progress counters; > k stalls on one filesystem | Every run on that filesystem | Unreadable files are an I/O fault, not a hang: the adapter keeps the lease and consults storage verdicts; checkpoint phases count I/O progress; > k simultaneous stalls is an infrastructure incident that suspends recoveries on that filesystem (§4.6) | A true hang during a brownout is detected late | 4×3 = 12 |
| Run adapter | Crash | Launcher restart count; heartbeat age at the store | One run | Launcher restarts it; it re-acquires the lease with the same epoch; exits caused by the adapter are classified `adapter_fault` and excluded from crash-loop counting | Crash during a commit leaves an orphan, never a bad commit | 4×3 = 12 |
| Run adapter | Bad release (every run on a version misbehaves) | Canary failure; `adapter_fault` rate by version | Every run pinned to that version | Per-cluster canary before widening; lockfile pins; rollback by pin change and in-place adapter restart (runbook R9); no cut marked `suspect` from adapter faults | Defect that only appears at scale | 5×2 = 10 |
| Contract channel | Stale attempt writes heartbeat, completion, or cut pointers | Signature or path mismatch counts | One run | Paths scoped per attempt and epoch; verbs and pointers signed by the adapter and verified; the store accepts only outputs matching signed committed cuts (02, 06) | Key compromise (06) | 5×1 = 5 |
| Run controller | Crash loop on a bad cut | Restore failures per cut | One run | Fallback ladder (02 §6.2); S4 breaker counts only run faults, never fenced, adapter, or filesystem exits | All retained cuts bad | 4×2 = 8 |
| Run controller | Restore hits a permission or read-only error (FMEA3-6) | Restore error class | One run | Classified `restore_infra`, not `cut_corrupt`: retried on the same cut at most twice, then escalated without marking the cut bad (02, R9-8); runbook R3 | Mis-set prefix ownership delays the restart | 3×2 = 6 |
| P2 runner (P2-full) | Unreachable during a Slurm requeue | `issue_epoch` failures from start scripts | Requeued jobs of that team | Start script holds the job and exits; no restart loop; the janitor releases it only if `issue_epoch` would succeed; the GPU-hour cap does not fail open because no job runs without an epoch (R8-9, R9-8) | Recovery waits for the runner | 3×3 = 9 |
| P2 runner (P2-full) | Delayed requeue of a finished or cancelled job | `issue_epoch` refusals | One stage | `issue_epoch` runs as the first line of the batch script, before `srun` and container start, so a refusal ends the job before the gang starts (FMEA3-4); it refuses unless the job is the stage's current attempt and the workflow is live; exit with the "do not requeue" code; Slurm's requeue limit bounds loops | The refused job still briefly holds its allocation | 5×2 = 10 |
| P2 runner (P2-min) | Stage fails; jobs are `--no-requeue` | Stage result; run-owner notification | One stage | None automatic; operator `retry_stage` (R14), allowed only when the previous job is absent from the queue | Recovery waits for an operator | 3×3 = 9 |
| Broker janitor (P2-full) | Down or lagging | Oldest held job age | Jobs held after a crash between record and release | Held jobs never run unrecorded; alarm; janitor resumes idempotently | Queue position lost while held | 3×2 = 6 |
| Broker janitor (P2-full) | Releases a hold it did not place (DS3-3) | Released-hold audit | One job released early | Ownership rule in the janitor row of §2 (03) | Hold reason reused by another tool | 4×1 = 4 |
| Placement broker | Down | Health probe, intent queue age | New placements, cross-cluster recovery | Fail-static; intents queue | Backlog after long outage | 3×2 = 6 |
| Rollout fleet | Shards retire one by one | Serving fraction (§5.1; **requires hook 6**) | One run's throughput | In-process restarts; recipe `min_healthy_shards` = 50% | Before hook 6, no alarm; only the recipe minimum | 4×3 = 12 |
| Refit | Collective hangs on dead peer | Refit duration; `RefitAborted` | One run | Refit watchdog when `refit_timeout_s` is set (recipes set it) | Watchdog disarmed in user config | 4×3 = 12 |
| Checkpointing | Saves fail; last cut ages | Checkpoint-age SLI (§5.4) | One run's RPO | Alarm; at 4× interval pause with a cut if one is possible, otherwise stop with reason `cut_impossible` (§5.4) | Cluster-wide storage outage | 4×3 = 12 |
| Checkpointing | Time limit before saving | Time-to-limit vs save duration | One interval of work | `checkpoint_must_save_by` from the launcher (§5.4) | Save slower than estimate | 3×3 = 9 |
| Data plane | Storage-unit node lost | Data-plane RPC errors | Rows in flight | Redispatch regenerable rows; persisted buffers per 02 | Stochastic payloads lost (by design) | 3×3 = 9 |
| Shared reward / judge | Outage or slow | Latency, errors, breaker state | Every consumer | Breaker; mask within budget; pause when exceeded | Multi-run goodput loss | 4×3 = 12 |
| Shared reward / judge | Bad release | Canary on calibration set; distribution shift (05) | Every consumer of that version | Pin by digest; canary; rollback by lineage (R4) | Drift below canary power | 5×2 = 10 |
| Plug-in | Poisoned or broken | Allow-list rejection; held-out disagreement (05) | Runs using that digest | Allow-list; quality failure pauses and gates | Allowed but wrong plug-in | 5×2 = 10 |
| Workflow layer | Duplicate stage launch | Duplicate step key | Two gangs | Store CAS on step key; `retry_stage` requires the old job absent with no requeue pending (R8-9) | Bugs in a new backend | 5×1 = 5 |
| Retention job | Deletes a needed cut | Dry-run diff review; deletion log | One run or published artifact | Dry-run first; CAS to `retiring` before delete (04 §4.3); delete-time re-check against cut history, gate pins, published artifacts; never deletes while a cut is `bad` or `suspect` or the only good one | Bug in the check itself | 5×1 = 5 |
| Budget enforcement | Exhausted mid-stage | Burn SLI (§6) | One run or tenant | `pause_and_checkpoint`, release; P2 QOS backstop | Pause fails during storage outage | 3×3 = 9 |
| Health-service adapter | Framework cancel without node diagnostics | Node-diagnostics run count vs framework cancels | Recovery lands on the same bad nodes | Framework emits the diagnostics-triggering event on every cancel (§4.9) | Event lost; supervisor applies the cached bad-node list | 4×2 = 8 |
| Health-service guard | Missing or outdated on a cluster | Skew matrix (§11); broker cluster-profile check | Tagged jobs requeued outside epochs | Broker refuses tagged placement on clusters without a compatible guard | Guard bypass by an unregistered tool | 5×1 = 5 |

### 3.2 Top cascading failures

Goodput losses are *(estimates)* for a 64-node RL run with a 30-minute cut interval and 10-minute restart-to-first-step; game days replace them (§9).

| Cascade | Chain | Containment | Goodput impact *(estimate)* |
|---|---|---|---|
| Cluster-wide fabric flap | IB flap → refit hangs in many runs → watchdogs fire → many restart requests | Refit watchdog in recipes; per-cluster bucket; restarts wait for the health service's fabric verdict; hardware-evidenced resubmits use their own budget | 20–40 min per affected run |
| Store blackout | Store down → every adapter fences → compute window → hold window → stop | Fenced compute keeps work that is committed on return; hold keeps allocations; after stop, operator-acknowledged staggered bulk release with queue priority kept (`requeuehold`/release or recovery QOS) | Within compute window: ~0. Within hold: hold time as `idle`. Beyond: blackout + ≤ 1 cut interval + staggered release tail |
| Filesystem brownout | Lustre MDS slow → contract reads fail, saves stall → many phase budgets exceeded at once | Unreadable ≠ frozen; I/O progress; > k stalls on one filesystem suspends recoveries and pages storage | Brownout duration; no mass self-termination |
| Bad adapter release | New version deployed → adapters fault → runs exit or stop renewing | Per-cluster canary; lockfile pins; `adapter_fault` excluded from breakers; rollback by pin (R9) | Canary runs only; without the canary, fleet-wide restarts |
| Bad cut | Latest cut corrupt → restore fails → crash loop | Fallback ladder; S4 breaker | One extra cut interval plus one restart |
| Bad reward/judge release, or outage | Bad scores in all consumers; or all consumers stall | Pinning, canary, masking budget, breaker, lineage impact query | Rollback to last cut before the release; or masked steps then paused GPUs |
| Node loss in payload-heavy families | Storage units and buffers lost → redispatch storms | Persisted buffers; storage units outside the trainer failure domain | Video: 5–15 min regenerated; embodied: one iteration |

## 4. Cascade handling details

### 4.1 Fabric flap and hardware evidence

The framework never excludes nodes on its own evidence; the run reports collective timeouts as evidence and the health service decides verdicts (03 §10.1). Ordinary resubmits use the per-cluster bucket (03 S2). Resubmits whose evidence includes a health-service hardware verdict use a separate per-run budget scaled by node count, 6 + nodes/64 per 24 h by default (03 S3), so a 1,024-node run is not stopped by routine node loss.

### 4.2 Store blackout

Timeline for one run (protocol in 03 §4.3 and 02 §5.5):

1. **Lease lost** (`TTL − ε` without renewal): the adapter stops commits, ledger uploads, verbs other than `stop` and safety pauses, and deletes. Training and refit continue within the attempt.
2. **`fenced_compute_max_s` elapsed:** the adapter sends `pause_and_checkpoint(then=hold)`; the run pauses in place and the allocation is kept so that recovery needs no queue wait.
3. **`fenced_hold_max_s` elapsed** (default 30 min, 03 §4.3): the adapter sends `stop(reason=fenced)`; the run exits with `completion.status=stopped` and `reason=fenced`, which is never a crash-loop failure.

If the store returns with the same epoch during step 1 or 2, uncommitted checkpoints commit and the run continues (from step 2 the adapter releases the hold with `pause_and_checkpoint(then=continue)`). After a long outage the supervisor does **not** recover stopped runs one by one. It lists them as a bulk recovery, an operator acknowledges it (runbook R8), and the supervisor releases them through a dedicated bucket at a staggered rate, by default 20 jobs per cluster per 10 minutes (03 §5.3), ordered by run priority. Recovery prefers `scontrol requeuehold` and release of the same job, or a recovery QOS, so recovered jobs keep their queue priority and age (whether `requeuehold` works on jobs submitted with `--no-requeue` must be verified per Slurm version; if not, the recovery QOS is used). The chaos tests in §9 judge both productive time while fenced and the recovery tail.

### 4.3 Bad cut

Covered by 02. Operations adds: "cut marked bad" notifies the run owner (pages from P3), and a cut marked `bad` or `suspect` is excluded from retention decisions so the previous good cut is never deleted while it is the only one.

### 4.4 Shared reward and judge services

See §5.3. A bad release is handled by version pinning recorded in lineage (05); rollback is a lineage query followed by per-run rollback intents (R4).

### 4.5 Node loss in payload-heavy families

Each family plug-in publishes payload sizing and buffer persistence policy (02, 08); the run view shows bytes at risk per node. Redispatch is rate-limited per run and reported as `wasted` time. Where storage units share the trainer's failure domain, the §3.1 row is re-scored at launch, and runs above the threshold must enable buffer persistence.

### 4.6 Filesystem brownouts (FMEA2-3, OPS2-5)

- An I/O error or timeout reading a contract file is counted as `contract_read_failure`, not as a frozen counter. The adapter keeps renewing while the health service's storage verdict for that filesystem is degraded, and withholds only on a **readable** frozen counter.
- Checkpoint and restore phases advance on bytes written or read, so a slow save is not a hang.
- Phase budgets are the measured p99 per family and phase, with a floor, recalibrated monthly from the run view.
- When more than k runs on one filesystem stall within one window (*(estimate)* k = 3 in 10 minutes), the supervisor declares a filesystem incident: it suspends stall-based recoveries on that filesystem, raises a Sev-2 to storage and the health service, and resumes per-run evaluation when the incident clears.

### 4.7 Adapter faults (FMEA2-5)

The adapter is a correlated dependency: one version runs in every run on a cluster. Containment: the launcher restarts it; versions are pinned per run in the lockfile; a new version goes to one canary cluster and run class first (§11); exits it causes are classified `adapter_fault` and never count toward S4 or mark cuts `suspect`. A rising `adapter_fault` rate on one version blocks that version's rollout and triggers R9.

### 4.8 P2 recovery path (normative for P2; FMEA2-6, OPS2-6, DS2-C, R9-2)

**P2-min (default).** Jobs are untagged and submitted `--no-requeue`. A failed stage stays failed until an operator calls `retry_stage` (runbook R14), which is allowed only when the previous job is absent from the queue with no requeue pending. The health service detects and runs RCA but does not requeue. There is no `issue_epoch` and no janitor.

**P2-full (only on the LLM pilot user's written confirmation).** Jobs keep Slurm requeue (overview D8). Every start, including a Slurm requeue, calls `issue_epoch(run_id, job_id, restart_count)` through the runner (or relay, where used) as the **first line of the batch script, before `srun` and container start** (FMEA3-4):

| Situation | Behavior |
|---|---|
| Job is the stage's current attempt, record `submitted` or `running`, workflow live, cap not reached | Epoch issued; the cut to restore is resolved now, at job start |
| Requeue after cancel, stale requeue of a finished job, superseded attempt, or cap reached | Refused; the script exits with the "do not requeue" code before any step starts; the job's Slurm requeue limit bounds any loop |
| Runner unreachable | Start script requeues its own job held (`scontrol requeuehold`, 03 §3.3) and exits; once the runner returns, the janitor releases it only if `issue_epoch` would succeed, otherwise cancels it (03 janitor rule) |
| Operator `retry_stage` | As in P2-min |

The runner runs on a service node, not a login node, so login reapers do not kill it, and, where the site runs a shared Slurm query cache, it queries Slurm status through it instead of polling slurmctld directly.

### 4.9 Health-service hand-off (R8-11)

- **Diagnostics trigger.** When the framework cancels a tagged job for a run-level failure, it emits an event that triggers node diagnostics, so bad-node verdicts continue while the remediation logic is advisory (needed where diagnostics otherwise run only when the remediation logic acts).
- **Tagging switch.** Tagging and `--no-requeue` switch on per cluster only when that cluster's supervisor leaves shadow mode; the supervisor refuses enforcement unless the cluster's tagging switch is on, and vice versa, so no cluster has tagged jobs without a recoverer.
- **Marker and guard.** The framework-managed marker is an administrator-set account or QOS, not the editable comment; the guard verifies identity against the store nonce. Clusters without a compatible guard get no tagged placement.
- **Break-glass.** Also triggers when the store is unavailable; action IDs are signed, single-use, expiring, and bound to one job (R13).
- **RCA.** If the site's job monitor runs automatic RCA, it stays on for tagged jobs until the health-service watch is live; disabling it is a P3 exit step.
- **Relay.** Unnecessary from P3: the agent reaches `gatedtrain-runtime-svc` only through its authenticated RPC and jobs hold no store credential (R9-9). Reachability of that RPC is checked at admission.

## 5. Degraded-mode alarms

### 5.1 Rollout fleet collapse (FMEA-6)

`FleetHealthPolicy` defaults to `min_healthy_shards=1` and `max_restart_attempts_per_shard=5` (`nemo_rl/models/generation/fleet_health.py:116-117`); `raise_if_exhausted` fires only below that minimum (`fleet_health.py:533-546`). A 32-shard run keeps training on one shard with no error.

- **Dependency.** The serving-fraction signal is exported through the fleet-health reporting callback of **hook 6**, which lands only after G0 (overview §5, R8-3). Until hook 6 is merged, this alarm does not exist; the only control is the framework recipe's `min_healthy_shards` = 50% of desired shards, which turns collapse into a run failure instead of silent degradation.
- **Signal (with hook 6):** serving fraction = `len(serving_shards())` / desired shards (`fleet_health.py:194`, `:246`).
- **Alarms:** warning below 0.75 for 10 minutes; critical below 0.5 for 10 minutes *(initial)*. Critical raises `pause_and_checkpoint` if the run declares `on_fleet_degraded: pause` (default above 8 nodes).
- **SLO:** time with fewer serving shards than desired < 5% of run time (§7), measured only for runs with hook 6.

### 5.2 Refit

`RefitAbortWatchdog` is inert unless a positive timeout is set (`nemo_rl/distributed/refit_watchdog.py:326`); the module notes an observed 1,801 s hang. Recipes set `refit_timeout_s`; an alarm fires when refit p95 exceeds 3× its run median. Refit is not fenced (R8-5), so fenced runs keep fresh rollout weights.

### 5.3 Shared reward and judge services (FMEA-7)

| Control | Rule *(initial values)* |
|---|---|
| Per-client circuit breaker | Opens at ≥ 20% errors or p99 > 3× SLO latency over 1 minute; half-open probes every 30 s |
| Run behavior while open | Rows are masked after their deadline within the per-step masking budget; beyond it, `pause_and_checkpoint` and release |
| Idle-GPU bound | GPUs blocked on a reward service count as `idle`; above 10% of a run's GPU-hours over 30 minutes the run pauses and releases |
| Service-side idle bound | Pools scale down below 30% utilization for 20 minutes; idle ≤ 15% of service GPU-hours per week |
| Admission | Per-tenant concurrency quotas |

### 5.4 Checkpoint age (FMEA-8 operational side, FMEA2-9)

- **Signal:** age of the newest committed cut (store commit time, 02 §6.4).
- **Alarms:** warning at 2× the cut interval; critical at 3×.
- **At 4×:** if a cut is possible (storage writable, store reachable, run not fenced), the run takes `pause_and_checkpoint`. If no cut is possible, pausing would only hold GPUs with nothing to save, so the adapter **stops the run with the declared reason `cut_impossible`**, recorded in `completion.json` and the run record. `cut_impossible` is an infrastructure exit: it never counts toward S4, it notifies the run owner and storage owner, and recovery restores the last committed cut once storage is healthy.
- **Time limit:** NeMo RL saves before the limit only if `checkpointing.checkpoint_must_save_by` is set (`nemo_rl/utils/checkpoint.py:237`, read by `TimeoutChecker`, `nemo_rl/utils/timer.py:439`); example configs set it to `null` (`examples/configs/grpo_math_1B.yaml:143`). The launcher derives it (02 §6.3) and chaos test C11 verifies it.

## 6. Budgets at run time (OPS-5)

Budgets are declared per workflow and per tenant (GPU-hours, labeling cost, external API spend). The P2 hard GPU-hour cap is enforced by the runner (`stop` at the cap, refusal in `issue_epoch`) with a Slurm QOS backstop that holds even when the runner is down (06). The controls below arrive with the P3 budget service. Fenced compute counts as GPU-hours; fenced hold counts as `idle`.

| Control | Rule |
|---|---|
| Burn-rate alerts | Fast (1 h window, ≥ 14× even-spend rate) and slow (6 h, ≥ 6×); thresholds at 50%, 80%, 95% |
| Estimate deviation | Alert when burn exceeds the validator's stage estimate (04) by 25%, or the projected total exceeds the budget |
| At 100% | Default `pause_and_checkpoint`, release, record the cut; alternatives `stop` or `continue_with_approval` |
| Gate waits | Partner pools scale to zero during gate waits; trainer allocations released after a cut when the wait exceeds 30 minutes |
| Accounting | Consumed, fenced, and idle GPU-hours per stage written to lineage (05) |

Budget actions are intents with the precedence defined in 03; safety, operator, and resiliency intents outrank them.

## 7. SLOs and error-budget policy (FMEA-9)

SLOs apply from P3 to framework-managed runs of pilot users. The health-service contract is in 03 §10; the timing targets below are its operational view.

| SLO | Target *(initial)* | Window | Measurement |
|---|---|---|---|
| Metadata store availability | 99.95% | 30 days | Linearizable probes from each cluster |
| Per-cluster store reachability (compute nodes, via the `gatedtrain-runtime-svc` RPC; no relay in P3, R9-9) | 99.9% | 30 days | Probes from a compute-node canary per cluster |
| Duplicate gang launches | 0; any is Sev-2 | Always | Two live jobs or processes with the same step key, or one step key with two epochs computing |
| Infrastructure exits counted as run faults | 0 | Always | Fenced, `adapter_fault`, `cut_impossible`, or filesystem-incident exits that advanced S4 or marked a cut `suspect` |
| Productive time while fenced | ≥ 90% of fenced compute time ends in a committed cut when the store returns within `fenced_compute_max_s` *(estimate)* | Per incident | Fenced span vs commit records |
| Detection to action, in-run | p95 ≤ 30 s event to intent applied | 30 days | Intent log |
| Detection to action, health-service verdict | p95 ≤ 2 min verdict to intent applied | 30 days | Health-service verdict time to intent log |
| Restart to first step | p90 ≤ 15 min (≤ 64 nodes), ≤ 25 min above *(estimate)*, excluding queue wait for new jobs | 30 days | Cycle start to first `step` span |
| Bulk recovery tail | 95% of runs resume ≤ 30 min and all ≤ 90 min after operator acknowledgement *(estimate)* | Per incident | Acknowledgement to first step |
| Checkpoint freshness | Newest cut age ≤ 2× interval for ≥ 99% of run time | 30 days | §5.4 |
| Restore success | ≥ 99.5% from the selected cut; 100% from latest or previous | 90 days | Restore events |
| RL-stage goodput | ≥ 90% (overview §7.3) | Rolling 7 days | `productive` share of nemo-lens buckets |
| Serving shards below desired | < 5% of run time (hook 6 runs) | Per run | §5.1 |
| Rollout efficiency | ≥ 85% of samples contribute gradient, excluding intentional filtering *(estimate)* | Per run | 05 ledger |
| Shared reward/judge latency | p99 within declared SLO (default 2 s) | 30 days | Gateway metrics |
| Shared-service idle GPU | ≤ 15% of service GPU-hours | 7 days | §5.3 |

**Error-budget policy.**

1. Each SLO has an owner (§2). Burning 50% of a 30-day budget in a week triggers a review; 100% freezes non-fix changes to that component until recovered or the owner and the framework operator accept the risk in writing.
2. A duplicate gang, a lost committed cut, a wrong retention delete, or an infrastructure exit counted as a run fault is Sev-2 regardless of budget, with a postmortem within five working days and a chaos test added for the cause.
3. Goodput below 90% for a pilot user's run for a week triggers a goodput review; the largest bucket becomes the operator's next reliability work item.
4. P3 exit requires all SLOs met for four consecutive weeks, the health-service RCA watch live, and responders for the framework services in place (§2).

## 8. Control-plane SLIs and observability (OPS-6, ARCH-9, OPS2-11)

### 8.1 Built on existing telemetry

`nemo_rl/telemetry` integrates with nemo-lens (`nemo_rl/telemetry/README.md:3`, `:21`). It defines span groups in `RLSpanGroup` (`span_groups.py:52`) and goodput buckets `productive`, `overhead`, `idle`, `wasted` (`instrumentation.py:151`), assigned by `_DEFAULT_GROUP_BUCKET` (`:176`) and stamped as `rl.bucket` (`:398`). The framework adds span groups and instruments the same way:

- Span groups: `gatedtrain.restore` (overhead), `gatedtrain.gate_wait`, `gatedtrain.placement_wait`, `gatedtrain.budget_pause`, `gatedtrain.fenced_hold` (idle), `gatedtrain.refit_verify` (overhead). Fenced compute keeps its normal buckets and carries `gatedtrain.fenced=true` so that productive time while fenced is measurable.
- Resource attributes: `gatedtrain.step_key`, `gatedtrain.epoch`, `gatedtrain.workflow_id`, `gatedtrain.attempt`, `gatedtrain.adapter_version`; cardinality grows with concurrent runs, not steps.

### 8.2 Control-plane SLIs

Labelled with `cluster` and `service`, and with `step_key` and `epoch` where a run is involved.

| SLI | Source | Alert *(initial)* |
|---|---|---|
| Store latency (p50/p99) and errors by operation | Store client | p99 > 50% of renewal margin; errors > 1% for 5 min |
| **Per-cluster store reachability** from compute nodes (service RPC, or a P2 relay where used) | Compute-node canary, adapters | Any cluster < 99% over 15 min |
| Lease renewal margin | Adapter, supervisor | < 30% of TTL |
| **Contract-file read-failure rate** by filesystem | Adapter | > 1% of reads for 5 min on one filesystem; any run with 3 consecutive failures |
| Runs fenced, fenced compute and hold seconds | Adapter | Any run in hold; > 10 runs fenced on one cluster |
| Filesystem incidents (> k simultaneous stalls) | Supervisor | Any |
| Adapter restarts and `adapter_fault` exits by version | Launcher, supervisor | > 2 restarts per run per day; canary version rate above the previous version |
| Oldest held job and oldest step in `submitting` | Janitor, workflow layer | > 10 min |
| `issue_epoch` refusals and runner-unreachable holds (P2-full) | Runner | Any refusal for a running stage; any hold |
| Reconciler drift actions | Supervisor, broker | Any on a healthy cluster; > 5/h |
| Resubmits per bucket (ordinary, bulk, hardware-evidenced) | Supervisor | Bucket ≥ 80% used; any run over its budget |
| Fencing rejections and contract signature failures | Store, adapter, run | Any from a live attempt |
| Retention: dry-run delete-set size and delete-time check refusals | Retention job | Any refusal; delete set > 2× previous run |
| Diagnostics triggers vs framework cancels | Health-service adapter | Ratio < 1 over 1 h |
| Gate deadline overruns; engine history fraction | Workflow layer | Any overrun; > 70% |
| Budget burn rate | Budget service | §6 |
| `find_by_key` `unknown`; intent queue age | Supervisor, broker | > 0 for 15 min; > 5 min |

### 8.3 NVRx cycle info for health-service RCA

The run writes NVRx cycle-info files, which a site's health service may consume for per-cycle RCA. It writes them with `CycleInfoWriter` (`src/nvidia_resiliency_ext/fault_tolerance/cycle_info_writer.py:100`): `write_cycle_start` on each (re)start and `update_cycle_end` on exit (`:122`, `:151`); files are `cycle_info.<job_id>.<attempt_index>.<cycle_number>` (`:43-44`) with active and standby nodes (`:138-146`). `attempt_index` comes from `SLURM_RESTART_COUNT` (`:60-61`): it increments for P2 Slurm requeues and for P3 recoveries by `requeuehold`/release of the same job; recovery-QOS resubmits create a new job, which a health service that groups cycles must link by the step-key tag (03). If the site's job monitor runs automatic RCA, it stays on for tagged jobs until the health-service watch is live (§4.9).

## 9. Chaos and game-day tests

Each test runs in deterministic simulation first (03 §12.2) and then on a staging cluster. Pass criteria are the §7 SLOs plus the listed invariant. Every test also asserts zero duplicate gangs and zero infrastructure exits counted as run faults.

| # | Test | Invariant / pass criterion | Phase exit |
|---|---|---|---|
| C1 | Store blackout beyond TTL, within `fenced_compute_max_s` | No stop, no resubmit; refit continues; productive time while fenced ≥ target; runs resume committing on return | P3 |
| C2 | Store blackout beyond `fenced_compute_max_s`, within the hold window | Runs pause in place keeping allocations; on return they resume without resubmission; hold time recorded as `idle` and charged to budgets | P3 |
| C3 | Store blackout beyond compute + hold windows | Runs stop as `fenced`; no breaker advanced; bulk recovery waits for operator acknowledgement, releases at the staggered rate, keeps queue priority; work loss ≤ blackout + one cut interval; bulk recovery tail meets §7 | P3 |
| C4 | slurmdbd down during submit | `find_by_key` returns `unknown`; no resubmit until accounting returns | P3 |
| C5 | IB flap during refit | Watchdog fires within `refit_timeout_s`; no partial weights served; the health service alone excludes nodes | P1b (watchdog), P3 |
| C6 | SIGSTOP the Ray head | Readable frozen counter withholds renewal; recovery within lease TTL + resubmit time; old gang terminal before the new one runs | P3 |
| C7 | Kill a storage-unit node | Rows redispatched or restored; efficiency dip recorded | P1b, P4 |
| C8 | Truncated latest cut | Verification fails; previous cut used; cut marked bad and kept from retention | P1b, P3 |
| C9 | Poisoned reward plug-in | Unknown digest rejected; wrong plug-in detected and paused | P1b, P2 |
| C10 | Shared judge outage | Breaker, masking, then pause-and-release; no deadlock | P4 |
| C11 | Run to the Slurm time limit | Cut committed before the limit; resumed run starts from it | P1b |
| C12 | Node fault seen by the health service and the supervisor | Exactly one exclusion and one restart | P3 |
| C13 | Fleet collapse to 25% shards | Before hook 6: run fails at the recipe minimum; with hook 6: critical alarm and pause intent within 10 min | P1b, P3 |
| C14 | Budget reaches 100% mid-stage | `pause_and_checkpoint`, allocation released, cut recorded; P2: QOS stops the job with the runner down | P2 (cap), P3 |
| C15 | Lustre metadata brownout on one filesystem (MDS throttled, contract reads time out) for 30 min | No mass self-termination: no renewal withheld for unreadable files; filesystem incident declared; no cut marked `suspect`; runs resume after the brownout | P3 (P2: no run killed by the runner) |
| C16 | Kill the adapter repeatedly in a live run | Launcher restarts it with the same epoch; no cut marked `suspect` or `bad`; no S4 advance | P2, P3 |
| C17 | Bad adapter release (fault-injecting build) to the canary cluster | Canary blocks widening; R9 rollback by pin restores runs; no suspect cuts | P3 |
| C18 | P2-full: cancel a workflow, then `scontrol requeue` its job | `issue_epoch` refuses before `srun`; job exits with "do not requeue"; no training step runs | P2-full |
| C19 | P2-full: delayed requeue of a finished attempt after a newer attempt started | `issue_epoch` refuses; the current attempt is unaffected | P2-full |
| C20 | Old job reported `NODE_FAIL` while its processes keep computing | No replacement before the fencing bound elapses or processes are confirmed dead; never two gangs | P3 (P2: `retry_stage` refused while the job or a requeue exists) |
| C21 | P2-full: runner down while Slurm requeues jobs; an administrator hold on another job | Start scripts hold and exit; no requeue loop; no job runs without an epoch; janitor releases only current self-held jobs when the runner returns and leaves the administrator hold | P2-full |
| C22 | Framework cancels a job on a node with an injected fault | Node diagnostics run within their normal latency; the bad-node verdict is applied to the recovery placement | P3 |
| C23 | Relay outage on an SSH-egress-only cluster | Treated as a per-cluster outage; admission refuses new placements; reachability SLI alarms | P2 (only where a relay is used; no relay from P3, R9-9) |
| C24 | Storage outage at 4× cut interval | Run stops with `cut_impossible`; no S4 advance; restore from the last committed cut after storage returns | P3 |

Game days: quarterly from P3, one §3.2 cascade per game day, with the operator's responders executing the runbooks.

## 10. Runbooks

Each runbook lives beside the service code, starts with "how to tell this is the problem", and ends with "how to verify recovery". P2 versions are manual; P3 versions reference the intents and dashboards.

| ID | Runbook | Trigger | Key steps |
|---|---|---|---|
| R1 | Run stuck without progress | Readable frozen counter | Check contract-file read failures first (if unreadable, go to R11); check NVRx cycle file and health-service verdicts; capture stacks; check refit watchdog and fleet state; `pause_and_checkpoint` or let the supervisor recover; open RCA through the health service |
| R2 | Store outage | Store availability SLI | Confirm fail-static and fencing (runs fenced, no resubmits); freeze workflow advances; watch which runs reach the hold and stop bounds; on return verify reconciler drift output, then go to R8 for stopped runs |
| R3 | Restore failure / bad cut | Restore failure, cut marked bad | Check the error class first: a permission or read-only error is `restore_infra` (fix prefix ownership or storage, retry the same cut; never mark it bad); otherwise identify cut; verify manifest hashes; force previous cut; protect it from retention; postmortem if a committed cut was bad |
| R4 | Bad plug-in or reward release | Canary failure, quality alarm, security notice | Revoke digest; lineage query for consumers; roll back each run to the last cut before first consumption; mark downstream artifacts suspect |
| R5 | Shared-service outage | Breaker open | Confirm breaker; scale or fail over; check consumers paused and released GPUs; re-close gradually |
| R6 | Duplicate gang (Sev-2) | Duplicate step key alarm | Cancel the older epoch; verify the newer epoch's paths untouched; check fencing and signature logs; preserve evidence; postmortem |
| R7 | Budget exhausted | 100% budget event | Confirm cut committed and allocation released; owner approves new budget or ends the workflow |
| R8 | Bulk recovery acknowledgement | Supervisor lists > N stopped runs after a store or relay outage | Confirm the store is healthy for 10 minutes and reachability is green on every affected cluster; review the list (runs, clusters, cuts to restore); acknowledge in `ControlAPI` (operator role, recorded); watch the staggered release and the tail SLO; pause the release if new failures appear |
| R9 | Adapter rollback | Canary failure or rising `adapter_fault` rate for a version | Stop the rollout; change the cluster's adapter pin to the previous version; the launcher restarts adapters in place (same epoch); clear any `suspect` marks set by adapter faults; verify renewals and commits resume |
| R10 | Relay outage (P2 only where a relay is used; retired with relays, R9-9) | Per-cluster reachability SLI | Confirm the relay (or the site query-service path) is down, not the store; fail over to the second replica; admission stays closed for that cluster until reachability is green; stopped runs go through R8 |
| R11 | Filesystem brownout | Contract read failures or filesystem incident | Confirm with storage and health-service storage verdicts; keep stall-based recovery suspended on that filesystem; do not cancel runs for unreadable files; after recovery, check for runs with readable frozen counters (R1) |
| R12 | P2-full held or refused jobs | Held-job age, `issue_epoch` refusals, runner down | Restart the runner on its service node; confirm the janitor releases only holds the framework placed (`held_by_broker` and matching hold reason, not before `not_before`) and self-held current attempts, and cancels refused ones; leave administrator holds to their owner; never release a refused job by hand |
| R13 | Health-service break-glass | Supervisor down beyond `T_bg`, or store unavailable, with a health-service hardware verdict | The health service may kill or hold the job; for a `resubmit`, the operator issues a signed, single-use, expiring action ID bound to that job; verify the guard logged it and the ID cannot be reused |
| R14 | P2-min stage retry | Stage failed (jobs are `--no-requeue`) | Read the health-service RCA and the stage's completion and restore classes (`restore_infra` → R3); confirm the previous job is absent from `squeue` with no requeue pending; call `retry_stage` through the `ControlAPI` (operator role, audited, rate-limited); verify the new attempt restores from the last committed cut and reaches its first step |

### 10.1 Alert routing and response

| Severity | Examples | Route | Response target |
|---|---|---|---|
| Sev-1 | Store down for all clusters beyond 15 min; committed cuts lost across runs | Framework and platform responders paged | Acknowledge 15 min |
| Sev-2 | Duplicate gang; wrong retention delete; filesystem incident; adapter canary failure; shared judge down for all consumers; store outage beyond `TTL − ε` | Framework responder paged | Acknowledge 30 min |
| Sev-3 | One run degraded (fleet below 50%, checkpoint age critical, `cut_impossible`, budget at 100%) | Run owner notified; ticket for the framework operator | Next working hours |
| Sev-4 | Warnings, estimate deviation, SLO reviews | Dashboard, weekly review | Weekly |

- Paging starts in P3 (responder arrangement in §2, checked at the P3 entry gate). In P2 all alerts go to the run owner's channel; the framework operator, as P2 store operator, handles store alerts in working hours.
- Hardware alarms stay with the health-service responders; the framework responders do not drain or exclude nodes. Framework alerts carrying a node identity are forwarded to the health service as evidence.
- Run-level alerts go through health-service watches and hooks where the cluster runs them; watches can trigger a health-service RCA for a stalled or paused run.

### 10.2 Dashboards

One dashboard per level, filtered by `gatedtrain.step_key` and `gatedtrain.epoch`: (a) fleet view of §8.2 SLIs per cluster, including reachability, fenced runs, and adapter versions; (b) run view with goodput buckets, fenced spans, serving fraction (hook 6), checkpoint age, refit duration, budget burn, and the NVRx cycle timeline; (c) shared-service view. Each runbook links to the panels that confirm its trigger and its recovery.

## 11. Upgrades and version skew (OPS-7, OPS2-11)

Runs are pinned (NeMo RL version, framework packages, and adapter version in the lockfile, 04). Shared services upgrade on their own schedule, so skew is normal.

1. **Compatibility window.** Shared services support run-contract and API versions N and N−1 for at least 90 days or the longest supported run, whichever is longer *(initial)*. Contract files carry a schema version (02); readers reject unknown major versions and accept unknown minor fields.
2. **Rolling upgrade.** Standby first, lease handover, then the former leader. Supervisor upgrades pause resubmits during handover. No upgrade during a game day or an open store incident.
3. **Adapter releases.** A new adapter version goes to one canary cluster and run class for *(estimate)* 48 hours of runs, then cluster by cluster; widening requires an `adapter_fault` rate no worse than the previous version. Rollback is a pin change (R9).
4. **Reversible schema migrations.** Expand–migrate–contract, tested down-migrations, run scoped to the framework's own tables and version table (never another service's migrations unscoped).
5. **Reward and judge services** are versioned by digest and moved by pin change, never replaced in place.
6. **Nightly skew matrix.** The conformance kit and a short synthetic run for {run N, N−1} × {adapter N, N−1} × {supervisor N, N−1} × {store schema N, N−1} × {**health-service guard** version deployed on each cluster, and the next guard release}, plus the oldest NeMo RL version still pinned by a pilot user. A failing cell blocks the shared-service release; a failing guard cell removes that cluster from tagged placement.

## 12. Changes from revision 6 (revision 7)

- Added a service inventory with owners, HA, backup, RTO/RPO, and phase; P2 services declared non-production (OPS-4).
- Replaced qualitative guardrails with SLOs, an error-budget policy, chaos tests mapped to phase exits, and runbooks (FMEA-9).
- Added degraded-mode alarms for fleet collapse, refit, shared services, and checkpoint age (FMEA-6, FMEA-7).
- Turned "raise a gate at 80%" into burn-rate alerts, estimate-deviation alerts, a defined action at 100%, and scale-to-zero during gate waits (OPS-5).
- Built observability on `nemo_rl/telemetry` and nemo-lens, plus control-plane SLIs and NVRx cycle info (ARCH-9, OPS-6).
- Added a version-skew policy (OPS-7).

## 13. Changes in revision 8

- Status set to "Direction — detailed review after G0"; P2 operations declared normative (R8-20).
- Inventory adds the P2 workflow database, P2 runner on a service node with `issue_epoch`, broker janitor, supervised and canaried run adapter, contract relay, budget service, health-service adapter, health-service guard, store-side retention job with dry-run and delete-time checks, OpenLineage/Parquet owner, audit log, and P3 alert-response guidance (OPS2-7, OPS2-11).
- Fail-static rewritten for the R8-5 fence: compute and refit continue, then pause in place, then stop; fenced time charged to budgets (§1, §4.2, §6).
- FMEA, cascades, and §4 add terminal-but-alive jobs, filesystem brownouts, adapter faults, the P2 requeue path, relay outages, retention errors, guard skew, and the diagnostics trigger (R8-6, R8-7, R8-9, R8-11).
- Bulk store-outage recovery is operator-acknowledged and staggered with its own bucket, keeps queue priority, and hardware-evidenced resubmits have a node-scaled budget (R8-6).
- Checkpoint-age rule stops with `cut_impossible` when no cut is possible; the fleet-collapse alarm is marked dependent on hook 6 (FMEA2-9).
- SLOs add per-cluster reachability, productive time while fenced, bulk recovery tail, and zero infrastructure exits counted as run faults; SLIs add contract-file read failures, reachability, fenced time, adapter faults, held jobs, `issue_epoch` refusals, retention checks, and diagnostics triggers (OPS2-11).
- Chaos tests C1–C3 now judge productive time while fenced and the recovery tail; C15–C24 added. Runbooks R8–R13 added; R1 and R2 updated. Skew matrix adds the adapter and health-service guard versions.

## 14. Round-1 disposition

| ID | Section | Resolution |
|---|---|---|
| ARCH-9 | §8.1 | Observability extends `RLSpanGroup` span groups and nemo-lens buckets instead of a new telemetry contract. |
| OPS-4 | §2 | Every stateful service has an owner, deployment, HA, backup, RTO/RPO, and phase. Updated in rev 8: inventory completed (OPS2-7). |
| OPS-5 | §6 | Burn-rate and estimate-deviation alerts, a defined action at 100%, scale-to-zero during gate waits; rev 8 charges fenced time to budgets. |
| OPS-6 | §8.2 | Control-plane SLIs with thresholds; rev 8 adds reachability and contract-read SLIs (OPS2-11). |
| OPS-7 | §11 | N/N−1 window, rolling upgrades, reversible migrations, nightly skew matrix; rev 8 adds adapter canaries and the health-service guard. |
| FMEA-6 | §5.1 | Superseded in part: the serving-fraction alarm depends on hook 6 (after G0); until then the recipe's 50% `min_healthy_shards` is the only control (FMEA2-9). |
| FMEA-7 | §5.3 | Breakers, masking within budget, idle-GPU bounds, admission quotas. |
| FMEA-9 | §3, §7, §9, §10 | FMEA, SLOs, chaos tests, runbooks. Updated in rev 8: store-outage behavior follows the R8-5 fence and R8-6 bulk recovery rather than "fenced compute then resubmit". |

## 15. Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| OPS2-7 | §2, §3.1 | Inventory adds budget service, health-service adapter, retention job (dry-run; delete-time checks against cut history, gate pins, published artifacts), broker janitor, OpenLineage/Parquet owner, run adapter, relay. |
| OPS2-11 | §2, §7, §8.2, §11 | Contract-file read-failure and per-cluster reachability SLIs and SLO; health-service guard and adapter in the skew matrix; P3 alert-response guidance. |
| FMEA2-9 | §5.1, §5.4, C13, C24 | Checkpoint-age rule stops with `cut_impossible` when no cut is possible; fleet alarm marked dependent on hook 6. |
| FMEA2-1, FMEA2-2 (operational) | §1, §4.2, §7, C1–C3 | Refit not fenced; pause in place then stop; staggered bulk recovery; tests judge productive time while fenced and the tail. |
| FMEA2-3, OPS2-5 | §4.6, C15, R11 | Unreadable vs frozen, I/O progress, p99 budgets, filesystem incidents. |
| FMEA2-4, OPS2-4, OPS2-9 | §4.1, §4.2 | `requeuehold`/release or recovery QOS; operator-acknowledged bulk bucket; node-scaled hardware budget. |
| FMEA2-5 | §4.7, C16, C17, R9 | Supervised adapter, per-cluster canary, `adapter_fault` excluded from breakers. |
| FMEA2-6, OPS2-6 | §4.8, C18, C19, C21, R12 | `issue_epoch` refusals, hold on runner outage, janitor release. |
| FMEA2-7 | §2, §4.8 | Runner on a service node, Slurm queries through a shared Slurm query cache where the site runs one. |
| FMEA2-8, OPS2-8, OPS2-1, OPS2-2, OPS2-3, OPS2-10 | §4.9, C22, C23, R10, R13 | Diagnostics trigger on cancel, per-cluster tagging switch, admin-set marker and guard, break-glass on store outage, automatic RCA kept, relay. Protocol details in 03. |

## Round-3 disposition

Revision 9 applies the operational consequences of R9-1 and R9-2 (header, §1, §2, §4.8, R14), R9-8, and R9-9 (relay rows, C23, R10 limited to P2 where used).

| ID | Section | Resolution |
|---|---|---|
| FMEA3-4 | §3.1 P2 runner row, §4.8, C18 | `issue_epoch` is the first line of the batch script, before `srun` and container start (P2-full); P2-min uses `--no-requeue`. |
| FMEA3-6 | §3.1 restore row, R3, R14 | Permission and read-only restore errors are `restore_infra`, not `cut_corrupt`; R3 checks the class first. |
| DS3-3 (operational) | §2 janitor row, §3.1 janitor rows, §4.8, C21, R12 | The janitor releases only holds it placed (`held_by_broker`, matching hold reason), honours `not_before`, and leaves other holds; protocol in 03. |
| EM3-5 | §2 | P3 is a separate scope decision; the no-P3 end state is P2 plus the health service as the site runs it. |
| FMEA3-1, FMEA3-2, FMEA3-3, FMEA3-5, FMEA3-7 | `open-issues.md` | Deferred to P3 (FMEA3-1 is a P3 entry blocker). |
| OPS3-1 … OPS3-8 | `open-issues.md` | Deferred to P3 (OPS3-6 to be confirmed in G0). |

## Notes for the overview

1. **`requeuehold` on `--no-requeue` jobs (resolved in 03):** 03 §5.2 prefers same-job `requeuehold` only where Slurm allows it for `--no-requeue` jobs and otherwise uses the recovery QOS; the check per Slurm version is listed in 03 §16.
2. **Store owner (revision 9):** the P3 store operator moves from a G0 dependency to a P3 entry-gate prerequisite (R9-1); the P2 store is operated by the framework operator. The store SLO and RTO/RPO in §2 and §7 depend on the P3 prerequisite.
3. **Alert response (resolved; revision 9):** the overview P3 row (§7.1) lists a responder arrangement for the framework services as a deliverable and exit criterion, outside the MVP estimate; it is checked at the P3 entry gate together with the health-service prerequisites.
4. **Estimates to calibrate in P3 shadow mode** (open): bulk release rate, hardware-evidenced budget, and filesystem-incident k (defaults owned by 03 §4.4, §5.1, §5.3), productive-while-fenced target, bulk recovery tail, adapter canary duration.
