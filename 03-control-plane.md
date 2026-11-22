# 03. Control Plane

**Status:** The design set is at revision 21; this document's design content was last changed in revision 9. A prototype (`gatedtrain` 0.3.0) implements revision 19. Frozen until G0 closes; P3+ content is Direction; deferred items in open-issues.md. §1–§3 (packaging, store rules, and the P2 subset) are **normative** for P2. §4–§12 (P3 and later) are **Direction** (R8-20): the protocols are recorded so P2 does not close doors; their parameters and scope are re-reviewed at the P3 entry gate (R9-1).
**Packages:** **R** = `gatedtrain-runtime-agent` + `gatedtrain-runtime-svc` (P2 subset, full in P3; R9-9) and **G `gatedtrain-control`** (P3) (overview §4, §7).
**Decisions applied:** D2 (run contract, two verbs), D7 (linearizable store, adapter-held leases tied to progress, model checking first), D8 (the cluster health service owns hardware verdicts; P2 jobs untagged; tagging per cluster from P3; under R9-2, P2-min jobs are `--no-requeue`), round-2 decisions R8-5, R8-6, R8-9, R8-10, R8-11, R8-18, and the control parts of R8-12, and round-3 decisions R9-1, R9-2, R9-7, R9-8, R9-9 (control parts).
**Code baseline:** NeMo RL `4f39fd8`; NeMo-Run `5f39d66`. The cluster health service (§10) is described by the roles this design relies on; a site's own service must be checked against them before implementation (§16).

## 1. Scope and packaging

### 1.1 What the control plane does

The control plane keeps framework-managed runs alive across controller, node, and service failures without running two copies of one stage, and without fighting the cluster health service. It does not contain workflow semantics (04), checkpoint cut format or restore (02), gate statistics (05), or identity and secret management (06). The broker accepts an opaque fencing token and never reads workflow state (overview §4, dependency rule 5).

### 1.2 Package R and package G (R8-10, ARCH2-3; R9-9)

Revision 7 had package E implement G's broker and store protocols, two run adapters, and a split `ControlAPI`. Revision 8 moved every shared protocol into one package below both; revision 9 splits R along its trust boundary (ARCH3-1).

| Package | Contents | Depends on |
|---|---|---|
| **R `gatedtrain-runtime-agent`** (in the allocation) | The **run adapter** (signing with the per-attempt key, MAC checks, commit gating; P3: lease holder and fencing; §4), using A's contract transport (02). Reaches outside the allocation only through a **narrow authenticated RPC** to `gatedtrain-runtime-svc`; holds no store credential | A |
| **R `gatedtrain-runtime-svc`** (service node) | Store interface and implementations (P2: single-primary SQL; P3: consensus-store client); the broker protocol and its **single** implementation (§6), including the P2 Slurm path and (P2-full) the janitor; `ControlAPI` (§8.1) including `issue_epoch` (§3.3); the **run-record envelope**; the `SchedulerPlugin` and `IntentPolicy` interfaces with P2 defaults and **one** Slurm plug-in (§7.2) | A |
| **E `gatedtrain-workflow`** | Workflow spec, DAG, gates, runner; writes run records only through R | A, R, E-stats, NeMo-Run (script generation only) |
| **G `gatedtrain-control`** | Control-plane HA: run supervisor (§5), health-service adapter (§10); **registers** further `SchedulerPlugin`s (§7) and the P3 `IntentPolicy` (precedence, per-class fencing, §8.2–§8.3) with R; defines no seam of its own (ARCH3-2) | A, R (not E) |

Store credentials never enter jobs. Because the agent talks only to the service, the relay of §10.5 becomes unnecessary from P3 on.

**Run record (ARCH3-3).** An **R-owned envelope**, versioned by R: run and stage keys; attempts (`job_id`, `restart_count`, `controller_epoch`, state, exit class); epochs; committed cut pointers and pinned cuts (05 gate pins). Every other field lives in a **namespaced section per owner, versioned separately**: E (`cancelling`/`cancelled`, open gates, budget counters), the adapter (P3 lease state, fenced seconds), G (`recovery_seq`, bucket accounting). Owners write only their own section through R's API, so E and G never import each other.

**Persisted `WorkloadSpec` (ARCH3-4).** The rendered `WorkloadSpec` is stored as an A-format artifact at first submission and referenced from the envelope; every resubmission (`retry_stage`, P2-full requeue, P3 recovery) uses that artifact, never a re-rendering.

### 1.3 Phase map

**P2-min is the default (R9-2, DA3-1).** **P2-full** is built only if the LLM pilot user confirms in writing at G0 that it needs automatic recovery in P2. In both variants the P2 store's operator is the **operator that runs the workflow runner** (DA3-3).

| Piece | P2-min (default, normative) | P2-full (only on written confirmation at G0) | P3 (R full, G; Direction) |
|---|---|---|---|
| Store | Single primary, no asynchronous replicas, meets §2 | Same | Consensus store with leases (§2) |
| `ControlAPI` | Identity provider, allow-lists, minimal per-principal rate limits (§8.1) | Plus `issue_epoch` | Plus RBAC, intent precedence, per-class tokens |
| Launch | Per-team runner submits its own `sbatch --no-requeue` (§3.2); untagged | Own `sbatch --hold --requeue`, janitor (§3.2) | Broker with tagged, `--no-requeue` placement per cluster (§10.2) |
| Run adapter | Signing, MAC checks, contract through A's transport; forwards commits to the runner; **no lease** | Same | Holds the run lease and fences side effects on lease loss (§4) |
| Dead-run recovery | Operator `retry_stage` only | The health service's Slurm requeue gated by `issue_epoch` (§3.3), plus `retry_stage` | Run supervisor (§5) |
| Tenancy | Per-team Slurm accounts, runners, and `run_root` isolation (§3.4) | Same | Per-tenant identities (§6.5) |
| Health service | Detection and RCA only | Unchanged behavior for untagged, requeueable jobs | Per-cluster hand-off (§10), after the P3 entry gate |

## 2. Metadata-store requirements (normative for P2 and P3)

| Requirement | Reason |
|---|---|
| Linearizable reads and writes for epochs, fencing tokens, broker records, cut commits, gate pins, and (P3) leases | A stale read can grant two holders the same epoch or lease |
| Compare-and-set on a record version, and multi-record transactions | Every protocol step in §3–§6 is one CAS |
| **P2:** a single primary with synchronous commit; **no asynchronous replicas** serve any read, and failover never promotes an asynchronous replica | Promoting an asynchronous replica loses acknowledged epoch and cut writes (DS2-7) |
| **P3:** leases with store-side expiry (clients never send absolute times); on leader change, remaining lease time reset to at least one TTL | Removes client clock skew; prevents mass expiry after failover |
| **Commit timestamps are authoritative.** Every time-based race (gate timeout versus gate decision, lease expiry versus renewal, break-glass expiry) is decided by the store's commit timestamp of the competing records; the first committed wins and the other CAS fails | Removes client-clock disagreement from gate and lease outcomes (DS2-7) |
| Watch or change-feed with resumable revisions (P3) | Supervisor and broker wake-ups; lost notifications are tolerated |
| Availability 99.95% per region, failover under 10 s, backup and point-in-time restore (07) | An outage beyond the fencing window stops runs (§4.3) |

**Not acceptable for fencing state:** asynchronously replicated MySQL or PostgreSQL (including any database a site's health service keeps) and any cache. They may hold dashboards and summaries. In P2, the workflow database qualifies only if configured as above; otherwise it holds no epochs or cuts.

**Owner.** The P2 store is the workflow database, operated by the **operator that runs the workflow runner** in P2-min and P2-full (R9-2, DA3-3). The P3 store product and its operator are a site prerequisite of the **P3 entry gate**, not of G0 (R9-1; DA2-4, overview §3). Without a designated store operator, P3 does not start.

## 3. P2 subset (normative)

P2-min is the default; text marked **P2-full** applies only if P2-full is confirmed at G0 (§1.3, R9-2).

### 3.1 Runner service

The P2 runner (04) runs as a **service on a service node**, not on a login node (FMEA2-7): **one runner per team** (RT3-3), one replica each, kept single-writer by a workflow lease row (04), restarted by the service manager. It builds every `sbatch` command itself; NeMo-Run is used only to generate scripts and container setup, because NeMo-Run's command line hard-codes `sbatch --requeue --parsable` with no hold (`nemo_run/run/torchx_backend/schedulers/slurm.py:148`, `:180`, `:253`) and its job lookup uses a local file (§7.1). If the site runs a shared Slurm query cache, status queries for known job IDs go through it, which spares slurmctld; the design uses it for **status only**: a cached "not found" is treated as `Unknown`, and every absence check that gates a submission or a retry, and every `issue_epoch` authorization (§3.3), queries the controller directly.

### 3.2 P2 broker: record first, submission, release

The P2 broker is R's broker (§6) restricted to one submitter per team:

1. **Record first.** Insert the attempt record as `submitting` with a fresh nonce, keyed by `(workflow_run_id, stage_id, loop_iteration, attempt)`, before any scheduler call.
2. **Submission.** P2-min: `sbatch --no-requeue --parsable` under the team account (§3.4), with the key hash and nonce in the comment; Slurm does not requeue it. Whether `--no-requeue` also blocks an administrator or health-service requeue is unverified (open-issues R9-OI-1, confirmed at G0). **P2-full:** `sbatch --hold --requeue --parsable`; `--requeue` stays because the health service's requeue is then the run-level recovery (D8).
3. **CAS and release.** CAS `submitting → submitted` with the job ID; P2-full then runs `scontrol release`. If the CAS fails because the stage is `cancel_requested`, cancel the job: under P2-full it was held and never ran; under P2-min it may have started briefly, and its commits fail the epoch CAS (accepted P2-min residual).
4. **Ambiguous `sbatch`.** The record stays `submitting`; `find_by_key` (§6.4) resolves it; `Unknown` never submits.
5. **Janitor (P2-full; DS2-6, DS3-3).** Every 60 s the janitor lists held jobs of the team account carrying the runner's marker (direct `squeue`). It **releases only holds the framework placed**: the attempt record carries `held_by_broker` (set before the held submission) **and** the job's Slurm hold reason matches the reason the broker uses; an administrator hold or any other hold is left alone and reported. A held job whose record is `submitted` with the same nonce is released (runner crashed between CAS and release); a record still `submitting` is resolved by `find_by_key`; a held job that matches no record, or whose stage is cancelling or no longer current, is cancelled. Jobs that held themselves (§3.3, runner unreachable; matching hold reason, current attempt) are released only if `issue_epoch` would currently succeed for them, otherwise cancelled. **Recovery holds carry `not_before`** in the record, and nothing releases them earlier.

### 3.3 `issue_epoch` and the requeue path (P2-full; R8-9, R9-8; DS2-C, FMEA2-6, FMEA3-4, OPS2-6, RT2-3, RT3-4)

Under P2-min there is no requeue: each job's `controller_epoch` is fixed in its attempt record at submission, and recovery is an operator `retry_stage` (precondition below). Under **P2-full**, the **first line of every job's batch script**, before `srun` and before any container starts, calls (so a refused requeue releases its allocation at once instead of starting a full gang; FMEA3-4):

```python
def issue_epoch(run_id: str, job_id: str, restart_count: int) -> EpochGrant: ...
# EpochGrant(controller_epoch, attempt_id="<job_id>.<restart_count>", restore_cut_id, contract_root)
```

`issue_epoch` first **authorizes with an authoritative `squeue -j <job_id>`** against the controller (never the query cache): the job must exist, run under the team account bound to the submission token, and carry the key hash and nonce of the stage (RT3-4). One store transaction then grants a new `controller_epoch` only if **all** hold:

- `job_id` is the stage's **current attempt** and `restart_count` is greater than any restart count already granted for it;
- the stage record is `submitted` or `running` (not `succeeded`, `failed`, or superseded by `retry_stage`);
- the workflow is not `cancelling` or `cancelled`, and no human gate is open on the stage;
- the GPU-hour cap and the stage's restart cap (`max_restarts`, default 5) are not reached.

The grant also resolves the **cut to restore at job start** (not at submission), so a requeue restores the newest committed cut. Otherwise the start script exits with the dedicated **"do not requeue" exit code** (79) and starts no GPU work; this job-start handshake (request, response, exit 79) is part of the **run contract in A** (ARCH3-7); the runner cancels any refused job that is not the current attempt. A delayed Slurm requeue of a finished job, or a requeue after cancel, therefore never trains (C'), and the health service's requeue cannot bypass the cap or cancellation. The cluster's Slurm requeue limit (`MaxBatchRequeue`, verified per cluster in P2 onboarding) bounds any remaining loop. Asking the health service to treat exit 79 as terminal for these jobs is a P2 request, not a dependency.

**Runner unreachable (P2-full).** If `issue_epoch` cannot reach the runner within `issue_timeout_s` (default 120 s), the start script requeues its own job held (`scontrol requeuehold $SLURM_JOB_ID`) and exits; no compute runs, so the cap cannot fail open (OPS2-6), and the health service sees no repeated crash. The janitor resolves the held job when the runner returns (§3.2).

**`retry_stage` precondition (both variants).** An operator's `retry_stage` creates a new attempt (with a new `controller_epoch`, from the persisted `WorkloadSpec`, §1.2) only when the previous job is **absent from the controller** (queried directly; purged past `MinJobAge`) or terminal with no requeue pending (`squeue` shows no `REQUEUE*` or pending restart). Otherwise it waits. Under P2-full, a late requeue of the old job is refused by `issue_epoch` anyway.

**P2 commits.** The P2 adapter forwards cut and completion manifests to the runner, which commits them with a CAS on `controller_epoch`. The adapter polls `current_epoch(run_id)` through the runner each heartbeat period and stops the run when a newer epoch exists.

**Residual P2 risk (accepted for non-production P2, closed in P3).** Under P2-min, a `retry_stage` after `NODE_FAIL` can overlap a still-live old incarnation on a partitioned node; under P2-full, a job requeued after `NODE_FAIL` can briefly overlap with a still-live old incarnation on a partitioned node that cannot reach the runner. Durability is protected (per-attempt, per-epoch contract paths; epoch CAS on commits; 02), but GPU time is wasted until the old processes die. P3's lease and resubmit rule (§5.2) close this.

### 3.4 P2 accounts and isolation (RT2-1, RT2-3)

- **Per-team Slurm accounts and runners (RT3-3).** The `ControlAPI` maps the authenticated principal to a team (06) and routes to **that team's runner**, which submits under the team's service user and Slurm account. Each runner's Slurm privileges are limited to `sbatch`, `scontrol release`, and `scancel` on its own team's account, so no runner is a cross-team deputy. A team's jobs cannot `scancel`, `scontrol`, or read another team's jobs or roots.
- **Rate limits (RT3-5).** P2 enforces minimal per-principal rate limits on `ControlAPI` calls and `retry_stage` (§8.1); the per-run paid-judge spend cap is in 05/06.
- **`run_root` isolation.** Each team's `run_root` is owned by its service user and not readable by other teams.
- **No store credential in jobs.** A job holds only a submission-bound token (step key and nonce, readable by the team user) that authorizes contract transport and, in P2-full, `issue_epoch` for that key. All other store effects go through the runner.
- **Tenant identity** finer than a team (per-tenant scheduler identity, §6.5) arrives in P3.

## 4. Leases, epochs, and fenced behavior (P3; Direction)

### 4.1 The run adapter

NeMo RL never talks to the store and never reads leases (D2, R8-4). The **run adapter** (`gatedtrain-runtime-agent`, one implementation for P2 and P3; it reaches the store only through `gatedtrain-runtime-svc`, §1.2) runs in the allocation on the Ray head node, is supervised and restarted by the launcher, and is pinned by version in lockfiles. In P3 it reads contract files through the transport (02), holds `lease/run/<run_id>` and renews it with the latest progress counter and phase, commits cuts and completion with a CAS on `controller_epoch`, signs verbs and pointers (Ed25519, 02), and delivers verbs deduplicated by action ID (§8.3). Every attempt receives a new `controller_epoch` at launch.

**Adapter canary (FMEA2-5).** A new adapter version rolls out first to canary runs on one cluster; exits caused by the adapter (crash loop, internal error, transport bug) are classified `adapter_fault`, never mark a cut `suspect`, and never count toward crash-loop breakers (§5.1). An `adapter_fault` rate above threshold on a cluster rolls the cluster back to the pinned previous version and stops the rollout.

### 4.2 Timing rule

Let `R` be the renewal period, `P` the maximum tolerated process pause, `D` the clock-rate drift bound over one TTL, and `ε` the adapter's fence margin.

1. `TTL > R + P + D`.
2. The adapter records `t0` (`CLOCK_MONOTONIC`) **before sending** each renewal; on success its local deadline is `t0 + TTL − ε`, with `ε ≥ D + P`.
3. If the deadline passes without a newer successful renewal, the adapter **fences** (§4.3) without waiting for the store.
4. The store expires the lease no earlier than `TTL` after granting it, so the local deadline precedes store expiry by at least `ε − D ≥ P`.
5. The supervisor acts only after an extra grace `G ≥ stop_grace_s + P`, on two signals, and under the resubmit rule (§5.2).

Proposed defaults, calibrated in shadow mode: `R = 10 s`, `TTL = 90 s`, `ε = 20 s`, `stop_grace_s = 60 s`, `G = 90 s`.

### 4.3 Fenced behavior (R8-5; FMEA2-1, FMEA2-2, RT2-5, EMB2-4)

| Phase | Bound | What the adapter does | What the run does |
|---|---|---|---|
| **Fenced compute** | `fenced_compute_max_s` (default two cut intervals) | Stops committing cuts, uploading ledger segments, and deleting files; keeps retrying renewal | Keeps training **and refitting** within its attempt (refit is not fenced); checkpoints stay in its epoch-scoped, uncommitted paths |
| **Fenced hold** | `fenced_hold_max_s` (default 30 min, proposed; other documents reference this default) | Delivers `pause_and_checkpoint(then=hold)`: the run pauses in place and **keeps its allocation** | Holds GPU memory and processes, no compute |
| **Stop** | `stop_grace_s` | Delivers `stop(mode=graceful, reason=fenced)`; kills the process group if it has not exited | Writes `completion.json` with `status=stopped`, `reason=fenced`, and exits |

- A renewal with the **same epoch** in either fenced phase resumes the run (from hold, the adapter delivers `pause_and_checkpoint(then=continue)`; there is no third verb); uncommitted cuts from the fenced window may then be committed, so a store outage shorter than `fenced_compute_max_s + fenced_hold_max_s` costs no resubmission and no queue position (FMEA2-2, OPS2-4).
- **`stop` and safety pauses are always honored** while fenced: verbs from the safety principal, the budget service, cancellation, and the launcher's signal handler are delivered immediately; only other verbs are deferred (RT2-5; supersedes revision 7's "rejects verbs").
- **Fenced time counts against budgets.** The adapter records fenced GPU-seconds and the budget service charges them; the Slurm QOS hard cap (06) applies regardless of the store (RT2-5).
- **`safety_critical` runs** use `fenced_compute_max_s = 0`: they enter fenced hold immediately.
- **Bench leases** do not inherit the fenced windows: the bench's local watchdog, independent of the adapter and the store, safe-stops the hardware immediately on heartbeat loss (details in 06; EMB2-4).

**Why fenced compute is safe.** A run whose lease cannot be renewed because the store is down cannot have been superseded, because no one advances the epoch without the store. When the store is up and the epoch has been advanced, nothing the old run writes can commit (epoch CAS, per-attempt paths, 02), and a replacement is released only under the resubmit rule (§5.2), which waits out the entire fenced sequence or requires confirmed process death. Adapter fencing is defense in depth; correctness rests on the epoch CAS and the resubmit rule.

### 4.4 Liveness tied to progress (FMEA2-3, OPS2-5)

A lease proves that the adapter runs, not that training does. The adapter **withholds renewal** only for a readable, frozen run:

| Observation | Classification | Action |
|---|---|---|
| Contract files readable, heartbeat `seq` advancing, progress counter frozen beyond the phase budget | Hang | Stack capture (07), then withhold renewal; supervisor recovery |
| Contract files **unreadable** (I/O error, stale handle, read timeout) | Storage problem, not a hang | Do not withhold; report `contract_io_error`, consult the filesystem's health verdict (the health service or the cluster profile's storage probe); renew with `phase=io_blocked` |
| Checkpoint phase with advancing `io_progress` (bytes or files completed) | Slow save | Not a stall |
| More than `k` runs (default 3) on one filesystem stalling within one window, counted in the store before withholding | Infrastructure | No run withholds on that ground; supervisor **holds** (no recovery, no stop) and pages |

Phase budgets are the measured **p99** of each phase per cluster, family, and scale, times a margin (default 2×), with a floor; uncalibrated phases use conservative defaults in shadow mode. NeMo RL's in-process stall watchdog (`nemo_rl/algorithms/single_controller.py:4185-4280`) runs only on the asynchronous path and inside the process it monitors, so the external check remains.

### 4.5 Restarts and outages

- **Store failover:** remaining lease time is reset (§2); adapters keep local deadlines and resume on a same-epoch renewal.
- **Supervisor start or failover:** a new leader takes no recovery action for one `TTL + G` after election.
- **Adapter restart inside a live run:** the launcher restarts it; it re-acquires the lease only with the same epoch, otherwise it stops the run.
- **Store outage:** shorter than `TTL − ε`, invisible; up to `fenced_compute_max_s + fenced_hold_max_s`, runs resume in place; longer, runs stop and are recovered by a bulk recovery (§5.3). Any outage beyond `TTL − ε` pages (07).

## 5. Run supervisor (package G, P3; Direction)

The supervisor recovers runs whose controller died or hung, on the same cluster, from the last committed cut. One deployment per cluster (two replicas, leader-elected through the store); fencing domain `supervisor/<cluster>`. Cross-cluster recovery is a broker migration (P11).

### 5.1 Safety rules

| # | Rule | Resolves |
|---|---|---|
| S1 | **Two signals.** Act only when the lease has expired plus `G` **and** the scheduler reports the job terminal or not running, or the Ray head is unreachable for a sustained window, or the last renewal carried a frozen counter | OPS-1 |
| S2 | **Buckets.** Store token buckets per cluster and globally for ordinary resubmits (defaults: 5 per cluster, 15 global per 10 min). **Hardware-evidenced** resubmits (health-service verdict or node-diagnostics result naming the nodes) use a separate bucket; **bulk recoveries** after a store outage use their own operator-acknowledged bucket (§5.3). An empty bucket queues recoveries and pages | OPS-1, OPS2-4, OPS2-9 |
| S3 | **Per-run budget.** Ordinary resubmits: at most 6 per run per 24 h. Hardware-evidenced resubmits: separate budget scaled by node count (default `6 + nodes/64` per 24 h). The **first retry has no backoff** when the old processes are confirmed dead, the failure is not `unclassified`, and the restored cut is not `suspect`; later retries back off (5, 10, 20 … min, capped at 60). Exhaustion opens a human gate | OPS-1, OPS2-9, FMEA2-4 |
| S4 | **Crash-loop breaker.** `N` consecutive failures (default 3) that do not advance progress past the restored cut mark that cut `suspect`; the next recovery uses the previous committed cut (02). Exits classified `fenced`, `adapter_fault`, `infrastructure_stall`, or hardware-evidenced **never count** | OPS-1, FMEA2-5 |
| S5 | **Shadow mode first,** per cluster, recording "would recover" decisions next to what the health service and operators did; enforcement is enabled per cluster after review (07) and switches tagging on for that cluster (§10.2) | OPS-1, OPS2-2 |
| S6 | **Same protocol as every launch.** Recovery goes through the broker (§6) with key `(run_id, recovery_seq)` and the supervisor's leader epoch as fence | DS-3 |
| S7 | **Resubmit rule** (§5.2) before any release | DS-3, DS2-B |
| S8 | **Respect cancellation and gates** recorded in the run record (§1.2) | ARCH-8 |
| S9 | **Exclude bad nodes:** the health service's bad-node list, the nodes in the failure evidence, and the node-diagnostics result for this kill (§10.3) | OPS-2, OPS2-1 |

### 5.2 Resubmit rule (R8-6; DS2-B, FMEA2-5)

A replacement is **released** only when the old job is **scheduler-terminal** (not `COMPLETING`) **and** one of:

1. **The fencing bound has elapsed:** `T_fence = t_lease_expiry + max(fenced_compute_max_s + fenced_hold_max_s, adapter_down_max_s) + stop_grace_s + pause_bound_s`, measured from the store's expiry commit timestamp. The first term covers a live adapter walking the fenced phases; `adapter_down_max_s` (02 §2.5, default 120 s) covers a dead adapter, whose run the launcher kills within that time; or
2. **The old processes are confirmed dead:** a terminal state other than `NODE_FAIL` with epilogs complete on every node (slurmd killed the steps), or, for nodes that failed, out-of-band evidence that the node was rebooted or powered off after the job started (health-service verdict, BMC, or slurmd re-registration with a boot time later than the job start).

`NODE_FAIL` alone is **unconfirmed**: slurmd was unreachable and processes may still compute. Such recoveries wait for `T_fence` unless the health service confirms the node reset. If terminal confirmation does not arrive within `terminal_timeout_s`, the supervisor opens a human gate instead of resubmitting.

**Open P3 entry blockers (not redesigned here; see `open-issues.md`).** `T_fence` does not bind orphan workers or suspended VMs (DS3-1), and recovery after `NODE_FAIL` waits about 90 minutes for it (FMEA3-1 / OPS3-1).

**Keeping queue position (FMEA2-4, OPS2-4).** Per cluster profile, recovery prefers (a) `scontrol requeuehold` of the **same job**, recorded `held_by_broker` with `not_before` set to the time the rule can first hold (§3.2), and release after the rule holds (same job ID, new attempt `<job_id>.<restart_count>`, new epoch at start), where Slurm allows it for a job submitted with `--no-requeue` (unverified, §16); otherwise (b) a new submission under a **recovery QOS** with a priority boost agreed with the site's cluster administrators. Either path is a broker operation and obeys the rule above.

### 5.3 Bulk recovery after a store outage

Runs that stopped because an outage outlasted the fenced windows are not resubmitted through the ordinary buckets. When the store returns, the supervisor builds a **bulk recovery plan** (runs, sizes, priorities, clusters), pages the operator's responder, and waits for an operator acknowledgement; it then releases the plan **staggered** from the bulk bucket (default 20 jobs per cluster per 10 min, tuned to slurmctld capacity), first retry without backoff. Runs still in fenced hold when the store returns resume in place and are not part of the plan.

### 5.4 Recovery transaction

1. One store transaction: check the leader epoch, S1, S3, S4, S8; increment `controller_epoch` and `recovery_seq`; record the intent with evidence.
2. Cancel the old job (or `requeuehold` it) through the plug-in; emit the diagnostics-triggering event (§10.3); poll until scheduler-terminal.
3. Wait for the resubmit rule (§5.2) and the node-diagnostics verdict (bounded by `diag_verdict_wait_s`, default 5 min; on timeout proceed with the evidence nodes excluded).
4. Submit or release through the broker with key `(run_id, recovery_seq)`, from the persisted `WorkloadSpec` (§1.2).

Every write of the old run fails from step 1. **Open P3 entry blockers (see `open-issues.md`):** a cancel during the step-3 `T_fence` wait still launches the recovery (DS3-2); the step-3 waits are the subject of FMEA3-1/DS3-1 and FMEA3-7.

## 6. Placement broker protocol (package R; P2 subset in §3.2)

### 6.1 Interface

```python
class PlacementBroker(Protocol):
    def submit(self, key: StepKey, fence: FenceToken, spec: WorkloadSpec) -> BrokerRecord: ...
    def cancel(self, key: StepKey, fence: FenceToken) -> BrokerRecord: ...
    def get(self, key: StepKey) -> BrokerRecord | None: ...

@dataclass(frozen=True)
class FenceToken:          # opaque to the broker
    domain: str            # e.g. "wf/<run>", "supervisor/<cluster>", "op/<ticket>"
    value: int
```

`StepKey` is opaque: E builds `(workflow_run_id, stage_id, loop_iteration, attempt)`, the supervisor `(run_id, recovery_seq)`. The broker checks only that `fence.value` equals the store's current value for `fence.domain`, in the same transaction as its write.

### 6.2 Records, submit, and cancel

States `submitting → submitted → terminal`, with `cancel_requested` (from `submitting`) and `cancelling` (from `submitted`). The record holds key, fence, a random 128-bit nonce, principal, scheduler identity, cluster, job ID, and store timestamps.

- **Submit:** insert `submitting` before any scheduler call; submit **held**; CAS `submitting → submitted` with the job ID, then release. A CAS that fails on `cancel_requested` cancels the held job. Errors and timeouts leave `submitting` for §6.4; the broker never re-submits blindly.
- **Cancel:** `submitting → cancel_requested` (no scheduler call); `submitted → cancelling`, then cancel and poll until terminal; `terminal`: no-op.
- **Janitor:** releases only `held_by_broker` holds with a matching hold reason and a passed `not_before`, whose record is `submitted` (crash between CAS and release), and cancels held jobs that match no `submitted` record (§3.2).

| Step | Revision 6 (DS-2) | Revision 7 and later |
|---|---|---|
| 1 | Broker checks workflow `running`, starts `sbatch` | Inserts `submitting`, starts `sbatch --hold` |
| 2 | Cancel finds no job ID, marks workflow cancelled | Cancel CASes `submitting → cancel_requested` |
| 3 | `sbatch` returns; job runs after cancellation | `submitted` CAS fails; held job cancelled; it never ran |

### 6.3 Placement refusals

The broker refuses: tagged placement on a cluster whose profile lacks the health-service guard at the required version (§10.2); placement on a cluster whose profile fails the store-reachability admission check (§10.5); and `safety_critical` stages outside a validated workflow (06).

### 6.4 Resolving `submitting` records: `find_by_key`

```python
def find_by_key(self, key: StepKey, nonce: str, principal: SchedulerIdentity,
                cluster: ClusterProfile, since: datetime) -> FindResult: ...
# FindResult = Found(list[JobHandle]) | Unknown(reason)
```

Matches key **and** nonce **and** the submitting UID or account in a window starting before record creation; a comment alone is never trusted. Returns `Unknown` when the controller or accounting is unreachable, lagging, or cannot cover the window, and for any cached not-found; `Unknown` never leads to a submission. `Found([])` allows a retry with a new nonce (a late ghost stays held and is cancelled by the janitor). `Found([J])` adopts `J`. More than one job for one nonce is an invariant violation: mark `conflict`, cancel nothing automatically, page.

### 6.5 Scheduler identity (RT-3, RT2-1)

The broker never submits under one shared account. P2 maps the principal to a per-team service user and account (§3.4); P3 maps it to a per-tenant identity (Slurm UID and account, or Kubernetes service account and namespace) server-side. A job created by another user with a forged comment does not match because its UID differs.

## 7. Scheduler plug-ins (`SchedulerPlugin` in R; G registers more; Direction beyond the P2 Slurm plug-in)

### 7.1 Why not NeMo-Run state (OPS-3)

`SlurmTunnelScheduler.schedule()` appends job IDs to a local file `~/.nemo_run/.slurm_jobs` (`slurm.py:71`, `:260-263`, `_save_job_dir` at `:508-514`); `describe()` returns `None` for any job not in that file (`:299-308`); `list()` runs `sacct --json -S1970-01-01-00:00:01` unfiltered (`:380-389`). A replicated component on another host would see "no such job". NeMo-Run stays the script and container generator only.

### 7.2 Slurm plug-in

| Operation | Implementation |
|---|---|
| `submit` | Script from NeMo-Run's executor (comment via `SlurmExecutor.comment`, `nemo_run/core/execution/slurm.py:326`); the plug-in builds `sbatch --parsable` with `--no-requeue` (P2-min, tagged P3 clusters, §10.2) or `--hold --requeue` (P2-full); P3 always adds `--hold`; under the mapped identity |
| `find_by_key` | `squeue --states=all --user=<uid> --name=<key-hash>` against the controller, then `sacct --user=<uid> --name=<key-hash> -S <since> --parsable2`; filter by nonce |
| `status` | The site's Slurm query cache, if any, for known handles (a status lookup by job ID); a direct `squeue -j` for any decision that releases, retries, or authorizes `issue_epoch` |
| Terminal confirmation | Terminal state **and** not `COMPLETING`; process death per §5.2 |
| `cancel`, `requeuehold`, `release` | `scancel` / `scontrol` under the mapped identity; then poll |

**Rule for `Found([])`:** absence is certain only if the controller query succeeded and the record is younger than `MinJobAge` minus a margin, or accounting is confirmed current (`sdiag` database-agent queue drained) for the window. If the query cache can return cached not-found entries, the plug-in maps them to `Unknown`. If the site's query service can only query by job ID, `find_by_key` queries the controller directly; a filtered `sacct` by name and user is an optional capability of that service. Where the site publishes job-state events, the plug-in uses them for wake-ups only.

### 7.3 Kubernetes and elastic operations

The Kubernetes plug-in (after Slurm) uses labels for key, nonce, and `managed_by`, a suspended Job or RayJob for the held state, and the API server for `find_by_key`. Elastic operations (`create_pool`, `grow`, `shrink`, `replace`, `destroy_pool`) and their rules are deferred unchanged to P12.

## 8. `ControlAPI`, intents, and fencing

### 8.1 API and allow-lists (`ControlAPI` in R; GOV2-5)

```python
class ControlAPI(Protocol):
    def submit_intent(self, target: Target, action: Action, evidence: Evidence,
                      expected_token: int | None, idempotency_key: str) -> IntentResult: ...
    def current_token(self, target: Target, target_class: TargetClass) -> int: ...
    def issue_epoch(self, run_id: str, job_id: str, restart_count: int) -> EpochGrant: ...
```

There is no priority argument; the principal comes from the authenticated request (06).

| Principal class (server-side) | Priority | Allowed actions | Default rate per target |
|---|---|---|---|
| Safety and policy services | 1 | `stop`, `pause_and_checkpoint`, `open_gate` | none |
| Safety officer (human) | 1 | Safety resume; co-signs safety break-glass (§10.6) | none |
| Gate approver (human, not the requester) | 2 | `decide_gate` | 10 / h |
| Release manager (human) | 2 | Release of accepted versions | 10 / h |
| Operators (human) | 2 | `cancel_workflow`, `retry_stage`, `resubmit`, `stop`, `pause_and_checkpoint`, bulk-recovery acknowledgement, break-glass grants (non-safety) | 10 / h |
| Resiliency: supervisor, health-service adapter | 3 | `resubmit` (with evidence), `pause_and_checkpoint`, `exclude_nodes`, `alert` | per §5.1 |
| Budget service (07) | 4 | `pause_and_checkpoint`, `stop` | 2 / h |
| Broker reclaim | 5 | `pause_and_checkpoint` | 2 / h |
| Run jobs (submission-bound token) | — | `issue_epoch` for their own key only | per §3.3 |

In P2 the default `IntentPolicy` (R) applies this table without precedence, plus **minimal per-principal rate limits** across all targets on `ControlAPI` calls and `retry_stage` (RT3-5; values calibrated with the other defaults); `issue_epoch` exists only in P2-full. The operator role **excludes** `decide_gate`, release, and safety resume (separation of duties). No role and no break-glass path resumes a run paused by a safety service without a safety officer (no-auto-resume). Out-of-list actions are rejected and audited. Allow-list changes go through reviewed Git merge requests (06).

### 8.2 Precedence and per-class tokens (G's `IntentPolicy`, P3; DS-8)

Precedence applies among pending intents for the same target and class; at most one action per target and class is applied at a time. Tokens are kept per `(target, class)` with classes `lifecycle` (stop, resubmit, rollback, migrate; a rollback is `stop` plus a new attempt from an older cut), `capacity`, `data`, and `gate`; `alert` is unfenced. A `lifecycle` action increments every class.

### 8.3 Executors honor action IDs

Each applied action has an **action ID** = hash(target, class, token). Executors (supervisor, broker, run adapter, the health service under delegation) record it with a CAS `pending → applying → applied` and treat a repeat as a no-op. Verbs carry the action ID and the adapter's signature; the adapter delivers each ID at most once.

### 8.4 Single-writer table

| Decision class | Single writer | Intent sources |
|---|---|---|
| Epoch grant for a P2 job start | P2-min: runner at submission; P2-full: `issue_epoch` in R (§3.3) | Job batch scripts (P2-full) |
| Stage progress, completion, cut commits | Run adapter (P3) or runner on its behalf (P2), CAS on `controller_epoch`; the run never writes the store | — |
| Restart a worker inside a run | The run (NeMo RL fleet health), reported through the fleet-health callback | Only the two verbs from outside |
| Pause, checkpoint, stop | The run, on a verb from the adapter | Supervisor, workflow layer, operators, safety, budget |
| Recover or resubmit a run on the same cluster, incl. rollback | P2-min: operator `retry_stage`; P2-full: also health-service requeue gated by `issue_epoch`; P3: run supervisor | Lease expiry plus a second signal, health-service adapter, operators |
| Record a stage `failed` when recovery is abandoned | Runner (P2), supervisor (P3) | — |
| Migrate, resize | Broker (P11), run (P12) | Supervisor, operators, autoscaler |
| Advance, loop, stop a workflow; gate decisions; publish | Workflow layer (04) | Gate approvers, release managers |
| Node verdicts and the bad-node list | Cluster health service (node diagnostics with fabric checks) | Health-service fault detection |
| Kill or requeue untagged jobs (requeue only under P2-full) | Health-service job controller | Health-service remediation logic |

## 9. Resiliency taxonomy and handling

NeMo RL's `classify_rollout_failure` maps unrecognized exceptions to `DATA` (`nemo_rl/experience/failures.py:315-343`); hook 5 adds `unclassified`.

| Category | Source | Automatic action |
|---|---|---|
| `infrastructure` | GPU, node, network, storage, scheduler; NeMo RL `INFRA` | Yes; hardware-evidenced bucket when nodes are named |
| `infrastructure_stall` | Correlated stalls on one filesystem (§4.4) | Hold; no recovery, no stop; page |
| `runtime` | Hang, OOM, crash, timeout | Yes |
| `external_dependency` | Reward service, simulator, labeling outage | Yes |
| `fenced` | Fenced sequence ended in stop (§4.3) | Recovery under §5.2; never counted by S4 |
| `adapter_fault` | Adapter-caused exit (§4.1) | Recovery; never counted by S4; feeds the canary |
| `do_not_requeue` | P2-full `issue_epoch` refusal (§3.3) | None; recorded |
| `data` | Explicit `RolloutDataFailure` or declared plug-in data failure | Mask and quarantine within a budget |
| `unclassified` | Anything else | **No;** S4 opens a gate at once |
| `numerical` | NaN, divergence, ratio blow-up | Limited |
| `quality` | Reward hacking, entropy or KL collapse | No; pause and gate |

Default handling from revision 7 is kept: node faults go to in-run replacement where supported, else checkpoint and supervisor resubmit excluding the node; hangs to stack capture and withheld renewal; reward or simulator outages to retry, failover, and circuit breakers (07); quality failures to pause and gate. The escalation ladder (retry → restart component → replace units → restart from cut → migrate → human gate), shadow mode for new detectors, and no silent training-affecting changes remain.

## 10. Cluster health service integration (package G, P3; R8-11, R9-1)

The cluster health service is the site's hardware-health system, in the roles this design relies on: fault detection, node diagnostics and fabric checks that produce node verdicts, remediation logic, a job controller that applies kill, hold, and requeue actions, a job monitor with automatic root-cause analysis (RCA), and watches on job state. This design relies on those roles, not on a particular product. The health-service changes this section requires of the site (§10.2–§10.4, §10.6) are **P3 entry gate** items, not G0 items (R9-1). Without P3 the end state is P2 plus the health service as the site runs it. In P2, the health service provides detection and RCA only (P2-min) or requeue as the site runs it (P2-full).

### 10.1 Detect, decide, act

| Concern | Detect | Decide | Act |
|---|---|---|---|
| Hardware and fabric faults | Health-service fault detection, node diagnostics, fabric checks | Health service: node verdict, bad-node list | Health service drains or excludes nodes; for tagged jobs submits an intent |
| Run-level recovery of tagged jobs | Adapter, supervisor, health-service evidence | Framework (§5) | Supervisor through the broker |
| In-run recovery | NeMo RL, plug-in probes | Run | Run |
| Untagged jobs (all P2 jobs) | Health service | Health service (remediation logic) | Job controller as the site runs it; P2-min jobs are `--no-requeue`, so recovery is operator `retry_stage` |
| Root-cause analysis | Stage `failed` record; the job monitor's automatic RCA until the health-service watch is live | Health-service watch on framework stages (from P3 exit) | Health-service RCA linked to the stage |

The framework builds no node health checks. The broker and supervisor read the health service's bad-node list as a placement filter (`--exclude`), cached with a short TTL; when unavailable, placement proceeds and the decision records that fact.

### 10.2 Tagging, per cluster (OPS2-2, OPS2-8, FMEA2-8)

- **Switch-on.** Tagging and `--no-requeue` switch on **per cluster, only when that cluster's supervisor leaves shadow mode** (S5). Until then the cluster's jobs stay untagged and requeueable under the P2 rules, so there is never a period with no recoverer.
- **Marker.** The framework-managed marker is an **administrator-set field**: a dedicated Slurm account or QOS that users cannot select, not the comment that users can edit. The comment still carries key hash and nonce for lookup.
- **Guard.** The health service's job controller refuses every state-changing action (hold, kill, node update, release, requeue-hold, requeue) on a job under the marker unless it carries a valid framework action ID (§10.6); signalling a job for stack dumps stays allowed. The guard **verifies identity against the store**: it looks up the broker record by job ID and requires the job's nonce to match, so a non-framework job in the account, or a framework job with a forged comment, is flagged. The guard fails closed on lookup failure. The guard must run before the job controller dispatches any action, so that no action path bypasses it.
- **Refusal.** The broker refuses tagged placement on any cluster whose profile does not record the guard at the required version; the guard version is in the skew matrix (07).

### 10.3 Node-diagnostics hand-off (OPS2-1)

If a site's node diagnostics run only when its remediation logic acts on a job, then with the remediation logic advisory for tagged jobs, node diagnostics would stop producing bad-node verdicts. Therefore, whenever the framework cancels or requeue-holds a tagged job for a failure (not for a user cancel or a completed stage), the health-service adapter emits an event that triggers node diagnostics with the job ID and attempt, in a schema agreed with the health-service operator. The supervisor waits for the node-diagnostics result before release (§5.4).

### 10.4 Changes on the health-service side

1. The remediation logic is advisory for tagged jobs on enforcing clusters; the health-service adapter converts its plan into an intent with fabric-check, node-diagnostics, and node evidence.
2. The job-controller guard (§10.2).
3. The job monitor keeps tracking tagged jobs for dashboards and goodput, **and, if it runs automatic RCA, that stays on** until the health-service watch for framework stages is live; that watch is a **P3 exit criterion** (OPS2-10).
4. Grouping resubmitted jobs by step key for per-cycle RCA (NVRx `attempt_index` stays 0 without Slurm requeue).

These are health-service capabilities the site provides at the P3 entry gate (R9-1); each must be confirmed against the site's own service before implementation (§16).

### 10.5 Store reachability (OPS2-3)

Store reachability from compute nodes is a **cluster-profile admission check**. Clusters whose compute nodes have only SSH egress use a **relay**: a framework process on a service node, or a site query service able to forward calls, that forwards authenticated adapter calls. The relay is a transport only: it holds no fencing state and caches no reads; its latency counts toward `P` in the timing rule (§4.2). Per-cluster store reachability is an SLI (07). With the agent/service split (§1.2, R9-9) the agent reaches only `gatedtrain-runtime-svc`, so from P3 on the relay is unnecessary; the requirement becomes service reachability from compute nodes.

### 10.6 Break-glass (FMEA2-8, RT2-6, GOV2-5)

The health service may **kill or hold** (never requeue or resubmit) a tagged job with a hardware verdict when the cluster's supervisor has been down longer than `T_bg` (default 15 min) **or the store has been unavailable** longer than `T_bg_store` (default 15 min). A returning supervisor sees the job terminal and recovers it under §5.

An operator may grant a **break-glass action ID** for a specific `resubmit` or `release`. Action IDs are **single-use, expiring** (default 1 h), **bound to one job ID and action**, and **signed** with the `ControlAPI` break-glass key (an offline key held by the operator's responder when the store is down). The guard checks the signature, expiry, and binding, and records used IDs in its own ledger to reject replay. Break-glass for a `safety_critical` run requires a **safety officer's co-signature**, and break-glass never resumes a safety-paused run.

## 11. Placement and coupling (Direction)

| Coupling | Placement | Default elasticity |
|---|---|---|
| `gang_fabric` (trainer workers) | One cluster, one fabric domain, all-or-nothing, homogeneous GPU type | `replace_only` |
| Rollout engines | Same cluster as the trainer | `resizable` (P12) |
| `service` (reward models, judges, simulators) | Any eligible cluster within the latency budget | `resizable` (P12) |
| `stage`, `batch` | Any eligible cluster | per stage |
| `bench` (hardware-in-the-loop) | Named bench pool only; exclusive | `fixed` |

Placement filters by policy tags, tenant, budget, hardware, scheduler capabilities, reachability (§10.5), guard version (§10.2), and the health service's bad-node list, then ranks by capacity and artifact transfer, and records the decision in lineage. Per-role `affinity`/`anti_affinity` (`same_node`, `same_nvlink_domain`, `same_rack`, `same_cluster`) and per-role accelerator types are placement inputs; roles that must start together are one heterogeneous Slurm job (`nemo_run/core/execution/slurm.py:339`), so one allocation, one lease, one epoch. Detail is in 08.

**Benches** are exclusive named resources (Slurm licenses or reservations, or a bench registry in the store) held by a **bench lease**; the bench's local controller rejects commands carrying a stale epoch, and its **independent local watchdog** safe-stops on heartbeat loss without any fenced window (06).

## 12. Formal verification and simulation (DS2-8)

### 12.1 TLA+ models

The P2 model is checked **before P2-full runner code** (P2-min is covered by the stage-lifecycle model without requeue); the others before P3 code.

| Model | Invariants (safety) | Liveness |
|---|---|---|
| **P2 requeue** (P2-full only): held submission, janitor, `issue_epoch`, health-service requeue, `retry_stage`, cancel | `NoTrainingAfterCancel`, `NoTrainingOfFinishedStage`, `OneEpochGrantPerRestart`, `HeldJobsResolved` | Every held job is released or cancelled |
| Stage lifecycle: broker, scheduler, cancel, supervisor | `OneLiveGangPerStage`, `NoLaunchAfterCancel`, `NoSubmitOnUnknown` | Every stage reaches `terminal` or a human gate |
| Lease, epoch, clock, fenced phases | `NoSideEffectFromSupersededEpoch`, `SingleLeaseHolder`, `NoReleaseBeforeFenceBound` | A dead run is recovered within `T_fence` + backoff |
| Cut commit, restore, GC, gate pins | `RestoreOnlyCommittedCurrentLineage`, `NoGCOfPinnedOrReferencedCut` | A run with a restorable cut resumes |
| Gate and publish | `AcceptedVersionAtMostOncePerTurn`, `GateOutcomeByCommitOrder` | Each turn ends accepted, rejected, or gated |

All models include message loss, duplication and reordering, process pause, bounded drift, store failover, and scheduler accounting lag. Revision 8 adds **terminal-but-alive jobs** (a `NODE_FAIL` job whose processes keep computing and writing), **requeue of finished jobs** (a Slurm requeue delivered after the stage succeeded or was cancelled), **delayed health-service actions** (kill, hold, release, or break-glass actions delivered late or replayed), and **the contract layer** (per-attempt contract paths, signed verbs and pointers, an old process writing after supersession).

TLC runs in CI on bounded instances (3 stages, 2 recoveries, 2 supervisor leaders); Apalache checks the timing model.

### 12.2 Deterministic simulation tests

P2 and P3 code run against a simulated store, clock, and scheduler in one seeded process; each fault maps to a conformance test (01); fleet-level chaos tests are in 07.

| Fault | Conformance test |
|---|---|
| `sbatch` timeout after the job was created | `broker/submit-ambiguous` |
| Ghost job after `Found([])` | `broker/ghost-held` |
| Runner crash between `submitted` CAS and release | `janitor/release-held` |
| Accounting lag after controller purge; cached not-found | `find/unknown-on-lag` |
| Cancel during submit | `broker/cancel-race` |
| Delayed requeue of a finished P2 job; requeue after cancel | `p2/requeue-refused` |
| Runner unreachable at job start | `p2/requeuehold-on-unreachable` |
| `retry_stage` while a requeue is pending | `p2/retry-precondition` |
| `NODE_FAIL` with live processes | `supervisor/fence-bound-before-release` |
| Async `scancel`, job `COMPLETING` | `supervisor/terminal-before-resubmit` |
| Store outage shorter and longer than the fenced windows | `lease/fenced-hold-resume`, `supervisor/bulk-recovery-ack` |
| Lossy store failover; supervisor failover mid-recovery | `lease/failover-grace`, `supervisor/epoch-fence` |
| Hung controller still renewing; unreadable contract files; correlated filesystem stalls | `lease/progress-liveness`, `liveness/io-error-not-hang`, `liveness/correlated-stall-hold` |
| `stop` while fenced | `fence/stop-honored` |
| Adapter crash loop | `supervisor/adapter-fault-not-counted` |
| Crash loop on one cut | `supervisor/breaker-fallback` |
| Forged comment from another UID; forged nonce under the marker account | `find/spoof-rejected`, `guard/nonce-mismatch` |
| Replayed or expired break-glass ID | `guard/breakglass-single-use` |
| Gate decision and timeout racing | `gate/commit-order` |

## 13. Invariants after this revision

| Invariant | Guaranteed by |
|---|---|
| At most one released gang per stage key | Record first, held submission released only after `submitted` CAS, janitor, `find_by_key` with nonce and UID, `Unknown` never submits (§3.2, §6) |
| P2-min jobs are never requeued (pending G0 confirmation, open-issues R9-OI-1) | `--no-requeue`; recovery only by `retry_stage` with its precondition (§3.2–§3.3) |
| A P2-full requeue trains only as the current attempt of a live, uncancelled, under-cap stage | Authoritative `squeue -j`, `issue_epoch` conditions as the first batch-script line; "do not requeue" exit; runner cancel; Slurm requeue limit (§3.3) |
| A P2-full requeue never runs while the runner is unreachable | Self-`requeuehold`, janitor resolution (§3.3) |
| The janitor never releases a hold it did not place, or a recovery hold early | `held_by_broker`, matching hold reason, `not_before` (§3.2) |
| At most one computing gang per run across P3 recoveries | Epoch increment first (§5.4); resubmit rule: scheduler-terminal and fence bound elapsed or death confirmed; `NODE_FAIL` unconfirmed (§5.2); per-cluster `--no-requeue` and guard (§10.2) |
| No launch after cancel (P2-min: no commit after cancel) | `cancel_requested` CAS, held job cancelled (§6.2); `issue_epoch` refusal (§3.3); P2-min: immediate cancel and epoch CAS (§3.2) |
| A superseded controller causes no durable side effect | Epoch CAS on commits and per-attempt paths (02); fenced adapter stops commits, uploads, deletes (§4.3); fenced broker calls (§6.1) |
| Cancellation, budget, and safety stops reach a fenced run | `stop` and safety pauses always delivered; fenced time charged; QOS cap (§4.3) |
| A store outage within the fenced windows causes no resubmission | Fenced compute then hold in place; same-epoch resume (§4.3) |
| A store outage beyond them causes no resubmit storm | Operator-acknowledged, staggered bulk bucket (§5.3) |
| A filesystem brownout does not stop runs | Unreadable files are not hangs; I/O progress; p99 budgets; correlated stalls hold (§4.4) |
| A hung but renewing controller is recovered | Withheld renewal on frozen progress (§4.4) |
| Adapter bugs do not mark good cuts suspect or trip breakers | `adapter_fault` class, canary, rollback (§4.1, S4) |
| Hardware recoveries are neither starved nor unbounded | Separate hardware bucket and node-scaled budget (S2, S3) |
| Bad nodes are excluded on recovery while the remediation logic is advisory | Diagnostics-triggering event and bounded wait (§10.3, §5.4) |
| The health service cannot restart a tagged job outside the framework | Admin-set marker, nonce-verified guard, broker refusal on unguarded clusters (§10.2) |
| No recoverer gap during rollout | Tagging per cluster only after shadow mode (S5, §10.2); automatic RCA kept until the health-service watch (§10.4) |
| Break-glass cannot be replayed or used to resume safety-paused runs | Single-use, expiring, job-bound, signed IDs; safety officer co-signature (§10.6) |
| Gate timeout and decision cannot both win | Store commit order (§2) |
| A caller cannot act outside its role; operators cannot decide gates, release, or resume safety pauses | Server-derived principal and allow-lists (§8.1) |
| Actions are applied at most once | Action IDs with CAS at executors (§8.3) |
| Unknown failures are never retried silently | `unclassified` (§9) |

## 14. Changes

### 14.1 From revision 6 (revision 7)

1. The broker takes an opaque fencing token (§6.1). 2. Held submission, `cancel_requested`, and `find_by_key` with nonce, UID, and `Unknown` (§6). 3. Explicit store requirements (§2). 4. Lease timing rule, store-side expiry, monotonic self-fencing, progress liveness, restart grace (§4). 5. The run never writes the store (§4.1, §8.4). 6. Supervisor safety rules and per-cluster sharding (§5). 7. Slurm plug-in without NeMo-Run's local file (§7). 8. Server-derived priority, per-class tokens, action IDs (§8). 9. Health-service contract, tagging, `--no-requeue`, break-glass (§10). 10. Co-location, heterogeneous hardware, benches (§11). 11. TLA+ and deterministic simulation before code (§12). 12. Elastic capacity to P12.

### 14.2 Revision 8

1. **Packaging:** package R holds the store interface, the single broker implementation, `ControlAPI`, the run-record schema, and the single run adapter, which uses the contract transport from A; G depends on A and R, not E (§1.2).
2. **Status:** §1–§3 normative for P2; P3+ marked Direction (header).
3. **P2 subset:** runner as a service on a service node with its own `sbatch --hold`, janitor, status queries through a site query cache where one exists (§3.1–§3.2); `issue_epoch` with four conditions, a "do not requeue" exit, Slurm requeue limit, self-`requeuehold` when the runner is unreachable, `retry_stage` precondition, cut resolved at job start (§3.3); per-team accounts and `run_root` isolation, no store credential in jobs (§3.4).
4. **Store:** P2 store must be single-primary with no asynchronous replicas; commit timestamps decide time races; store operator is a G0 dependency for P3 (§2).
5. **Fenced behavior:** fenced compute with refit, then fenced hold keeping the allocation, then stop; `stop` and safety pauses always honored; fenced time charged; `safety_critical` and bench rules (§4.3).
6. **Liveness:** I/O errors versus frozen counters, I/O progress, p99 budgets, correlated stalls (§4.4).
7. **Supervisor:** resubmit rule with fence bound and confirmed death, `NODE_FAIL` unconfirmed (§5.2); queue-position preservation; bulk recovery (§5.3); separate hardware bucket and node-scaled budget; first retry without backoff; fenced and adapter exits excluded from breakers; adapter canary (§4.1, §5.1).
8. **Cluster health service:** diagnostics-triggering event; per-cluster switch-on; admin-set marker and nonce-verified guard that runs before every job-controller action; broker refusal on unguarded clusters; break-glass on store outage with signed single-use IDs and safety co-signature; automatic RCA kept until the health-service watch; store-reachability admission and relay (§10).
9. **Allow-lists:** operators lose `decide_gate`, release, and safety resume (§8.1).
10. **Verification:** P2 requeue model; terminal-but-alive jobs, finished-job requeue, delayed health-service actions, and the contract layer added to TLA+ scope; new simulation tests (§12).

## 15. Notes for the overview and siblings

Revision 9: overview and siblings should show R as two packages (agent and service), P2-min as the default, the job-start handshake in A's run contract, and the P3 store operator and health-service prerequisites at the P3 entry gate (R9-1). All six revision-8 notes are resolved: overview §4 shows R below E and G; overview §3 and §5.1 list the store operator (now a P3 entry-gate item); 02 §2.3 and §5.5 define `pause_and_checkpoint(then=hold)` and the run no longer rejects verbs; 06 §5.3, §7.1, §8.5, and §14.5 cover the launch token, per-team users, break-glass key custody and co-signature, and the bench watchdog; 07 §2, §8.2, §10, and §11 add the health-service adapter, janitor, bulk-recovery runbook, adapter canary, guard skew, and the new SLIs; overview §5.1 carries the health-service changes (the exit-79 request stays optional, §3.3).

## 16. Unverified facts

- Slurm: `--hold` under each cluster's submit plug-ins; which hold reason `sbatch --hold` and `scontrol requeuehold` set, and that it differs from an administrator hold (janitor, §3.2); whether `scontrol requeuehold` (by the framework, or by a job on itself) is allowed for jobs submitted with `--no-requeue`, and whether requeue preserves priority age; `MaxBatchRequeue` values per cluster; `MinJobAge` retention; `sdiag` agent queue as an accounting-freshness signal; whether a user-unselectable account or QOS can be enforced on every target cluster.
- Whether slurmd re-registration exposes a boot time usable as death evidence (§5.2) on each cluster.
- Store behavior on leader change (lease reset) for the chosen product; synchronous-standby failover for the P2 database.
- Cluster health service: the job-controller owner guard, the action dispatch, the node-diagnostics trigger, the remediation handlers, and the job monitor's automatic RCA are assumed roles; each must be verified against the site's service before implementation.
- Whether a site's Slurm query service can serve as an authenticated relay for adapter calls (§10.5) without changes.
- All numeric defaults (buckets, budgets, `fenced_hold_max_s`, `k`, margins) are proposals for shadow-mode calibration.

## Finding disposition (round 1)

| ID | Section | Resolution |
|---|---|---|
| ARCH-8 | §6.1 | The broker checks only an opaque fencing token per domain and never reads workflow state. |
| DS-2 | §6.2 | Record-first `submitting`, held submission, and `cancel_requested` CAS make cancel and submit linearize. |
| DS-3 | §5.1–§5.4, §6.4, §7.2 | Recovery uses the broker protocol with `(run_id, recovery_seq)`, epoch increment first, the resubmit rule (superseding revision 7's terminal-only check), and nonce-plus-UID lookup returning `Unknown` under lag. |
| DS-4 | §2, §4.2 | Linearizable store with store-side expiry, a stated timing rule, and monotonic-clock fencing in the run adapter (revision 8). |
| DS-8 | §8.2–§8.3 | Per-class tokens; executors deduplicate by action ID. |
| DS-9 | §12 | TLA+ models and deterministic simulation precede code; scope extended in revision 8. |
| OPS-1 | §5.1 | Two signals, separate buckets, per-run budgets, crash-loop breaker, shadow mode. |
| OPS-2 | §10 | The health service owns node verdicts; tagged jobs are guarded per cluster; the remediation logic is advisory with a node-diagnostics hand-off. |
| OPS-3 | §7.1–§7.2 | The Slurm plug-in queries `squeue`/`sacct` filtered by user, name, and window. |
| FMEA-1 | §4.4 | Renewal withheld on frozen progress, now distinguished from I/O errors and correlated stalls. |
| FMEA-2 | §6.4, §7.2 | `find_by_key` returns `Unknown` under lag or cached not-found; `Unknown` never submits. |
| DA-3 | §1.3, §10 | The control plane uses the site's health-service detection, a Slurm query cache where the site runs one, and job-state events where the site publishes them, with single-owner rules. |
| RT-3 (broker) | §6.4, §6.5 | Lookup matches nonce and UID; per-team (P2) and per-tenant (P3) identities. |
| FMEA-9 (health-service contract) | §10 | Detect/decide/act per concern, guard, advisory remediation logic, and break-glass; SLOs and runbooks in 07. |
| EMB-6 (scheduling) | §11 | Affinity, heterogeneous per-role hardware in one job, and leased benches with a local watchdog. |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| ARCH2-3 | §1.2 | Package R holds the store interface, single broker implementation, `ControlAPI`, run-record schema, and single run adapter (using A's contract transport); G depends on A and R, not E. |
| DS2-B | §5.2, §12 | Release requires scheduler-terminal plus the full fencing bound or confirmed process death; `NODE_FAIL` is unconfirmed; terminal-but-alive jobs are modeled. |
| DS2-C | §3.3 | `issue_epoch` refuses non-current attempts, finished stages, cancelled workflows, and capped stages; the job exits "do not requeue". |
| DS2-6 | §3.1–§3.2 | The runner submits its own `sbatch --hold`; a janitor re-releases jobs held after a crash between CAS and release and cancels unmatched held jobs. |
| DS2-7 | §2 | P2 store is single-primary with no asynchronous replicas; gate timeout versus decision is decided by store commit order. |
| DS2-8 | §12.1 | TLA+ scope adds terminal-but-alive jobs, requeue of finished jobs, delayed health-service actions, the contract layer, and a P2 requeue model. |
| FMEA2-2 | §4.3, §5.3 | Fenced hold keeps allocations so outages within the windows need no resubmission; longer outages use an acknowledged, staggered bulk bucket. |
| FMEA2-4 | §5.1–§5.2 | Same-job `requeuehold` or a recovery QOS keeps queue priority; first retry without backoff where safe. |
| FMEA2-5 | §4.1, §5.1 | Adapter-caused exits are `adapter_fault`, never mark cuts suspect or count toward breakers; per-cluster adapter canary with rollback. |
| FMEA2-6 | §3.3 | Runner unreachable → self-`requeuehold` and exit; `retry_stage` needs the old job absent with no requeue pending; restore cut resolved at job start. |
| FMEA2-8 | §10.2, §10.6 | Admin-set marker with nonce-verified guard; broker refuses unguarded clusters; break-glass also on store outage. |
| OPS2-1 | §10.3, §5.4 | Framework cancels emit a diagnostics-triggering event; the supervisor waits, bounded, for the node-diagnostics verdict. |
| OPS2-2 | §5.1 (S5), §10.2 | Tagging and `--no-requeue` switch on per cluster only after its supervisor leaves shadow mode. |
| OPS2-3 | §10.5 | Store reachability is a cluster-profile admission check; relay on a service node or a site query service. |
| OPS2-4 | §4.3, §5.2–§5.3 | Fenced hold keeps allocations; queue-position-preserving recovery; bulk bucket separate from ordinary limits. |
| OPS2-6 | §3.3 | No compute without an epoch grant, and runner outages hold jobs, so the cap cannot fail open. |
| OPS2-8 | §10.2 | The guard keys on an administrator-set account or QOS and verifies the store nonce, not the comment. |
| OPS2-9 | §5.1 (S2, S3) | Hardware-evidenced resubmits use a separate bucket and a node-scaled budget. |
| OPS2-10 | §10.4 | The job monitor's automatic RCA stays on until the health-service watch is live, a P3 exit criterion. |
| RT2-1 | §3.4, §6.5 | Per-team Slurm accounts and `run_root` isolation in P2; per-tenant identity in P3. |
| RT2-3 | §3.3–§3.4 | Jobs hold only a submission-bound token for `issue_epoch`; every health-service requeue passes the cap and cancel checks. |
| RT2-5 | §4.3 | `stop` and safety pauses are always delivered while fenced; fenced time is charged; QOS cap backstop. |
| RT2-6 | §10.6 | Break-glass IDs are single-use, expiring, job-bound, signed, and replay-checked by the guard. |
| GOV2-5 | §8.1, §10.6 | Operators cannot `decide_gate`, release, or resume safety pauses; safety break-glass needs a safety officer; break-glass never resumes safety-paused runs. |
| EMB2-4 (bench lease) | §4.3, §11 | Bench leases have no fenced window; an independent local watchdog safe-stops immediately (details in 06). |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| DS3-3 | §3.2, §6.2 | The janitor releases only `held_by_broker` holds whose Slurm hold reason matches the broker's; other holds are left and reported; recovery holds carry `not_before`. |
| FMEA3-4 | §3.3 | `issue_epoch` is the first line of the batch script, before `srun` and container start (P2-full). |
| RT3-3 | §3.1, §3.4 | One runner per team, with Slurm privileges limited to `sbatch`, `scontrol release`, and `scancel` on that team's account. |
| RT3-4 | §3.1, §3.3, §7.2 | `issue_epoch` authorizes with an authoritative `squeue -j`; the query cache serves status only. |
| RT3-5 | §3.4, §8.1 | Minimal per-principal rate limits on `ControlAPI` and `retry_stage` in P2; paid-judge cap in 05/06. |
| DA3-1 (03 part) | §1.3, §3 | P2-min (`--no-requeue`, operator `retry_stage`, the health service for detection and RCA) is the default; P2-full only on written confirmation at G0. |
| DA3-3 | §1.3, §2 | The operator that runs the workflow runner also operates the P2 store in both variants; the P3 store operator is a P3 entry-gate item. |
| ARCH3-1 | §1.2, §4.1, §10.5 | R splits into `gatedtrain-runtime-agent` (narrow authenticated RPC, no store credential) and `gatedtrain-runtime-svc`; the relay is unnecessary from P3. |
| ARCH3-2 | §1.2, §7, §8.1–§8.2 | R defines `SchedulerPlugin` and `IntentPolicy` with P2 defaults and one Slurm plug-in; G only registers more. |
| ARCH3-3 | §1.2 | Run record = R-owned envelope (keys, attempts, epochs, cut pointers) plus namespaced, separately versioned owner sections. |
| ARCH3-4 | §1.2, §3.3, §5.4 | The rendered `WorkloadSpec` is an A-format artifact persisted at first submission and used for every resubmission. |

**Deferred to `open-issues.md`** (not changed here): DS3-1, DS3-2, FMEA3-1 (P3 entry blockers; pointers in §5.2 and §5.4), FMEA3-2, FMEA3-3, FMEA3-5, FMEA3-7, OPS3-1 … OPS3-8, GOV3-3.
