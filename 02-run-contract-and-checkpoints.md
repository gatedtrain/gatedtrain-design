# 02. Run Contract and Checkpoints

**Status:** The design set is at revision 29; this document was last changed in revision 22. The prototype (`gatedtrain` 0.7.2) implements revision 29 except the training-seed unit of revision 20 (05 §3.3). Frozen until G0 closes; deferred items in `open-issues.md`. **Parent:** `design.md` (overview), decisions D2 and D6; round-2 decisions R8-4, R8-5, R8-7, and the run-side parts of R8-6, R8-9, R8-15, R8-17, R8-19 (round-2 review record, not published); round-3 decisions R9-2, R9-3, R9-6–R9-9 (round-3 review record, not published).

**Baseline:** NeMo RL `4f39fd8`; verl-omni `3580c1a` (pins verl `fefb080`). Every code fact below cites `file:line` in those clones. Numbers marked *(estimate)* are not measured.

This sub-design defines how a training run talks to everything outside it (the run contract), what a sample records about its origin, and how a run checkpoints so that a restart never resumes from an inconsistent, superseded, or deleted state. It builds on what NeMo RL already does and changes as little of it as possible. Revision 8 moved every lease, fencing, signing, and commit decision out of NeMo RL into the run adapter in package R: the run emits files and executes two authenticated verbs, and nothing else. Revision 9 names it `gatedtrain-runtime-agent` and marks P2-full-only parts.

## 1. What NeMo RL already does

| Mechanism | Current behavior | Kept or changed |
|---|---|---|
| Trainer checkpoint naming | `LocalStorage.begin` creates `tmp_step_N`, deleting a stale one first; `publish` renames it to `step_N`, swapping out an existing `step_N` (`nemo_rl/utils/checkpoint_storage.py:65-97`). `MscStorage` writes `step_N` in place, deletes an unpublished prefix, and publishes with `_SUCCESS` (`checkpoint_storage.py:106-127`) | Kept inside an attempt-scoped root (§5) |
| "Latest" and GC | `get_latest_checkpoint_path` returns the highest published step (`nemo_rl/utils/checkpoint.py:682-691`); `remove_old_checkpoints` deletes by `keep_top_k` / `ft_keep_latest_k` and is a no-op when both are unset (`checkpoint.py:562-633`, early return at `:590-591`); deletion runs in the background after the rename (`checkpoint.py:477-485`) | Replaced in contract mode by the store pointer and store-side retention (§5) |
| Full checkpoint | `_save_checkpoint_impl` captures controller state under the exclusive side of `DataPlaneCheckpointBarrier`; "Non-colocated engines keep serving rollouts throughout. Colocated engines are stood down for the save" (`nemo_rl/algorithms/single_controller.py:5927-5940`, barrier at `:5983`); publication is last (`:6137-6140`) | Kept unchanged |
| Rollout snapshots | `_rollout_checkpoint_pump` saves periodically (`single_controller.py:5660-5740`); `_save_rollout_checkpoint` is anchored to a durable `step_N`, refuses while an optimizer commit is in progress, and discards itself if a train step commits meanwhile (`:5363-5473`); Gym is parked only for the snapshot (`:5443-5485`); snapshots are written under the anchor directory (`prepare_snapshot_paths`, `nemo_rl/algorithms/single_controller_utils/rollout_checkpoint.py:548-576`) | Kept; generalized through `CheckpointParticipant` (§4) |
| Snapshot fallback | `commit_snapshot` keeps `keep_latest_k` snapshots (`rollout_checkpoint.py:579-611`); `resolve_latest_snapshot` takes the newest snapshot matching the anchor's step and trainer version (`rollout_checkpoint.py:614-672`) | Kept as the first rung of the fallback ladder (§6) |
| Gym protocol | `GymCheckpointAdapter`: `discover`, `prepare`, `renew`, `commit`, `retire`, `forget`, `restore`, `resume`, each with a deadline; participants declare a `restart_only` mode (`nemo_rl/environments/gym_checkpoint_adapter.py:193-481`, `:218-229`); a lost `resume` is healed by Gym's checkpoint lease (`nemo_rl/environments/gym_checkpoint_coordinator.py:44-48`) | Becomes one implementation of `CheckpointParticipant` |
| Buffer barrier | `DataPlaneCheckpointBarrier` admits concurrent mutations and gives one checkpoint holder exclusivity; it keeps a monotonic `mutation_version` (`nemo_rl/algorithms/async_utils/replay_buffer.py:300-420`) | Kept; hold time measured per cut (§4.4) |
| Data-plane save | Only `simple` and `mooncake_cpu` are checkpointable (`nemo_rl/data_plane/interfaces.py:244-257`); with `mooncake_cpu` and `storage_unit_segment_size > 0`, only CPU storage-unit actors own Mooncake memory and every other process is "off the save path" (`interfaces.py:184-186`, placement at `:289-301`) | Kept; persistence policy and separate storage units (§4.4) |
| Deadline save | `TimeoutChecker` with `checkpoint_must_save_by`, measured from `NRL_JOB_START_EPOCH` (set by `ray.sub:28`), fitting the average iteration time (`nemo_rl/utils/timer.py:439-497`; wired at `single_controller.py:699-703`, used at `:4018`) | Kept; the launcher derives its value (§6.3) |
| Signals | No `USR1` handler exists in `nemo_rl/`; `ray.sub` traps `EXIT TERM HUP INT` in the batch shell only (`ray.sub:36-38`), so a Slurm `--signal=B:USR1` reaches the batch shell, not the controller actor | The launcher traps it and asks the adapter for a verb (§6.3) |
| Status file | `latest_checkpoint_status.json` with time and step of the last save, "for monitoring … no in-repo consumer yet" (`nemo_rl/algorithms/grpo.py:3027-3051`) | Superseded by the heartbeat (§2.4) |
| Weight version | Integer `_trainer_version`, incremented per optimizer step (`single_controller.py:3793`, `:3963`); stamped on rollouts (`nemo_rl/experience/rollout_manager.py:2418-2424`), stored as `int64` per record (`nemo_rl/data_plane/tq_token_sink.py:130`, `:561-563`), and subtracted directly by the staleness sampler (`nemo_rl/algorithms/async_utils/staleness_sampler.py:406`, `:439`); 28 modules under `nemo_rl/` mention `weight_version` | **Unchanged.** Lineage identity adds the controller epoch outside NeMo RL (§7) |
| In-run stall watchdog | `_stall_watchdog_pump` compares (committed groups, train steps) and raises `RolloutStall` (`single_controller.py:4185-4280`) | Kept; the adapter's external check is independent (§2.6) |
| Refit | `WeightSynchronizer` assumes global atomic updates (`nemo_rl/weight_sync/interfaces.py:33-38`); IPC and NCCL transports check `update_success` and raise (`interfaces.py:112-115`); seven synchronizer implementations exist; `RefitAbortWatchdog` aborts a hung collective from a thread inside the worker, and an aborted collective returns without raising and leaves partial weights (`nemo_rl/distributed/refit_watchdog.py:17-33`, `:309-345`) | Unchanged by default; digest is opt-in hook 6b (§8) |

## 2. Run contract

The run contract is the only interface between a run and the outside world (overview §4, dependency rule 1). Package A defines the file formats, the transport interface with both implementations (POSIX directory, and object store behind an optional dependency), and an emitter library (pure Python); package R holds the run adapter `gatedtrain-runtime-agent`, which uses A's transports; NeMo RL emits through hook 8a, wired by package C-ext `nemo-rl-gatedtrain`. Hook 8a in NeMo RL is a sink protocol with a no-op default: NeMo RL calls `emit_heartbeat`, `emit_cut`, `emit_completion`, and `poll_verbs` and contains no transport, cryptography, lease, or store code. NeMo RL never imports A, R, E, G, or F.

**Hook 8a is split (R9-3, NRL3-4).** **8a-1**: the emitters and `stop`. **8a-2**: `pause_and_checkpoint` (`hold`, `continue`, `exit`), a controller state machine proposed only after a prototype with a watchdog test (a held run trips neither NeMo RL's stall watchdog nor collective timeouts). 8a-1 and 8d may ship first as C-ext shims in `nemo-rl-gatedtrain`, so P2 does not wait on upstream merge pace; the upstream PRs replace them without changing the contract. Until 8a-2 exists, every use of `pause_and_checkpoint` below falls back as stated in place (§2.3, §5.5, §6.4).

The run has exactly two channels: launch inputs (§2.1) and the contract transport (§2.2). It never reads a lease, never calls the store, and never decides whether its own outputs count.

### 2.1 Identity and launch inputs

| Input | Assigned by | Meaning |
|---|---|---|
| `run_id` | Workflow layer (04) | One stage run; stable across attempts |
| `attempt_id` | Run supervisor (03); in P2 the job start script, as `<job_id>.<SLURM_RESTART_COUNT>` (always `.0` in P2-min, which submits with `--no-requeue`) | One scheduler submission or one Slurm requeue of it; never reused |
| `controller_epoch` | Store, once per attempt at launch (03). P2-min: `gatedtrain-runtime-svc` when the runner submits the attempt (first run or operator `retry_stage`). P2-full: `issue_epoch` at job start (§5.7) | Fencing token and lineage identity; strictly increasing per run |
| `restore_cut` | Store pointer: resolved at submission in P2-min (the previous attempt is terminal and nothing requeues), at job start in P2-full and P3 | `cut_id`, manifest SHA-256, and an adapter signature over both; absent for a fresh run |
| `adapter_pubkey` | `gatedtrain-runtime-agent`, generated per attempt under the agent's OS user | Ed25519 public key that verifies verbs and the `restore_cut` pointer |

All five are passed as environment variables (the key as a file path) and recorded in every contract file. Nothing in this list changes during an attempt; an adapter restart inside a live attempt reuses the same epoch and key (§2.5). The job-start handshake of P2-full (§5.7) is part of the run contract that package A defines (R9-9, ARCH3-7).

**External runs (revision 22).** A run started by `ext-launch` and collected by `ext-collect` has no scheduler job to name. Its attempts are numbered `A = 1, 2, ...` and the ledger of a new run uses `attempt_id = <run_id>.<A>` and `controller_epoch = A`; the parent of attempt A+1 is `(A, branch_step)` of the cut it restored (05 §12.4).

### 2.2 Layout and transport

```
<run_root>/
  contract/attempt_<A>/epoch_<E>/   # one contract namespace per attempt and epoch
    heartbeat.json                  # run, replaced every 15 s
    completion.json                 # run, create-only, once
    cuts/<cut_id>.json              # run, create-only: manifest path and SHA-256
    ledger/step_<N>.<fmt>           # run, create-only: ConsumptionRecord segments, epoch in header (§3)
    control/index.json              # adapter, replaced: highest verb sequence number
    control/<seq>_<action_id>.json  # adapter, create-only: signed verbs
    acks/<action_id>.json           # run, create-only then replaced: verb outcome
  attempt_<A>/epoch_<E>/step_<N>/   # one trainer checkpoint (§5)
  attempt_<A>/epoch_<E>/step_<N>/rollout_snapshots/snapshot_<k>/
<committed_root>/<run_id>/<cut_id>/ # agent's OS user (or write-once storage): committed cuts (§5.2)
```

Scoping the contract namespace by attempt and epoch means a surviving process of an older attempt can write only into its own namespace; the adapter reads only the current one, so it cannot overwrite a new attempt's heartbeat, completion, or acks (DS2-A).

**Transport interface** (package A, with both implementations; the object-store one is an optional dependency; R8-7). Four operations: `put_create(key, bytes)` fails if the key exists; `put_replace(key, bytes)` replaces atomically; `get(key)`; `list(prefix)`. Readers see either the old or the new object, never a partial one.

| Implementation | `put_create` | `put_replace` | Used on |
|---|---|---|---|
| POSIX directory | Write a temporary file, `fsync`, `link()` to the final name (fails if it exists) | Temporary file, `fsync`, `rename()` | Slurm clusters with a shared filesystem; Kubernetes with a `ReadWriteMany` volume; the research profile |
| Object store | Conditional put with `If-None-Match: *` (or the provider's equivalent generation precondition) | Unconditional put | Kubernetes without shared volumes; clusters whose checkpoints use `MscStorage`; clusters where the contract should not share a filesystem with checkpoints |

A cluster profile (03) names the transport and its root. Where the checkpoint filesystem is Lustre, the profile should place the contract on a different filesystem or the object store, so a checkpoint-filesystem brownout does not also blind the adapter (§2.6).

**Polling load.** The run replaces one heartbeat every 15 s and reads one small `control/index.json` every 5 s and at each optimizer-step boundary; it lists `control/` only when the index sequence number changes. The adapter reads one heartbeat per 15 s and lists `cuts/` and `ledger/` only when the heartbeat's counters say something new was written. That is under 0.5 metadata operations per second per run, about 500 per second for 1,000 concurrent runs *(estimate)*. The adapter backs off its reads to 60 s when the transport reports latency above a threshold, and contract-file read failures are an SLI (07).

### 2.3 Verbs

The run accepts exactly two verbs: `stop` from hook 8a-1, `pause_and_checkpoint` from hook 8a-2.

| Verb | Arguments | Semantics |
|---|---|---|
| `pause_and_checkpoint` (8a-2) | `action_id`, `deadline`, `then ∈ {continue, hold, exit}` | At the next optimizer-step boundary, take a full cut (§4.1); write the ack with the `cut_id`. `continue` resumes training; `hold` keeps the allocation and does no compute until the next verb; `exit` writes `completion.json` with `status=paused` and exits. A cut already taken at that boundary satisfies the verb. While holding, a `pause_and_checkpoint(then=continue)` resumes without a new cut |
| `stop` (8a-1) | `action_id`, `mode ∈ {graceful, immediate}`, `reason` | `graceful`: finish the current optimizer step, write no new cut, write `status=stopped` with the reason, exit. `immediate`: stop at once; the last committed cut stands |

**Authentication (RT2-2).** Each verb file carries `run_id`, `attempt_id`, `controller_epoch`, `seq`, `action_id`, `issued_at`, the arguments, and an Ed25519 signature by the adapter's per-attempt key. The emitter library verifies the signature with `adapter_pubkey` and checks that the identifiers equal the run's launch inputs; anything else is acked `rejected_invalid` and reported. The run never checks leases, so it always executes a valid `stop` or safety pause, including while the adapter considers the run fenced (RT2-5).

**Idempotency.** The run records each accepted `action_id` with its outcome; a repeated file gets the same ack without a second action. Accepted actions appear in the next cut's `applied_actions`. Acks move through `accepted` → `done` or `failed` with a reason.

**Delivery of deadline signals.** On Slurm the launcher sets `--signal=B:USR1@<lead>`. The signal reaches the batch shell (§1), whose trap tells the adapter, which writes a signed `pause_and_checkpoint(then=exit)` with `action_id = sig-<job_id>-<restart_count>`; before 8a-2 the agent only records the signal and the `TimeoutChecker` save (§6.3) is the sole trigger. NeMo RL needs no signal handler. Files avoid a network listener inside the run (06).

Rollback, budget enforcement, and gates are not verbs. An externally requested rollback is `stop`, then a new attempt from an older cut, started by the supervisor (03) or, in P2, the workflow runner. There is no in-run rollback (§7).

### 2.4 Files the run writes

**Heartbeat** (`heartbeat.json`, replaced every 15 s by the controller's event loop):

| Field | Purpose |
|---|---|
| `seq` | Increments on every write; detects a frozen writer |
| `progress.optimizer_steps`, `progress.rollouts_completed`, `progress.rows_committed` | Monotonic counters advanced only by completed work, never by a timer |
| `progress.io_units` | Checkpoint and restore work units completed in the current phase (rank shard saves or loads returned, data-plane partitions saved, participant commits) |
| `phase`, `phase_started_at` | `startup`, `restoring`, `training`, `refit`, `checkpointing`, `held`, `paused_for_verb`. Budgets per phase are the adapter's, not the run's (§2.6) |
| `weight_version` | NeMo RL's integer counter (§7) |
| `last_cut_id`, `last_cut_wall_time` | Newest cut the run wrote (not necessarily committed) |
| `fleet` | Healthy and total rollout shards; applied in-process actions (hook 6) |

**Cut manifest** (`attempt_<A>/epoch_<E>/step_<N>/cut_manifest.json`, written last, after every component is durable). Fields: launch identifiers; `parent_cut_id` (the previous cut this attempt wrote, or the restored cut); `weight_version` and `controller_epoch` (recorded once here, §7); `last_step_included` (the last optimizer step whose update is in the saved weights; 05 §12.4 uses it for ancestry); `model_manifest_hash` (the deployable components only, 05 §11); per-component file lists with sizes and per-file hashes (format in 05; chunked Merkle roots from P3, 05 §13.3; reconciling P2 manifest hashes with P3 roots is LIN3-5 in `open-issues.md`); data-loader cursor and sampler dispatch index (already captured at `single_controller.py:5984-5999`); participant commit summaries (§4.2); buffer policy outcome, `barrier_hold_s`, and `gpu_seconds_lost` (§4.4); covered ledger segments (§3); `applied_actions`. The run then creates `contract/.../cuts/<cut_id>.json` holding the manifest path and its SHA-256. A rollout snapshot gets the same treatment, with `anchor_cut_id` set.

**Completion manifest** (`completion.json`, created once at exit): `status` (`succeeded`, `failed`, `stopped`, `paused`); `reason` for `stopped`; failure class from the taxonomy in 07 (including `unclassified` and `restore_failed`); `final_cut_id`; output artifact manifest hashes; final counters. An attempt that exits without one is classified by the supervisor from scheduler state and adapter records (03). The `fenced` status of revision 7 is gone: the run no longer knows it is fenced.

**Trust in run-written files.** Run-written files are claims, not facts. The adapter accepts them only from the current attempt's namespace, verifies every manifest hash it commits, and records completion only if `final_cut_id` is a committed cut or absent. The store accepts output artifacts only when their manifest hashes match a signed committed cut (06). A forged `completion.json` or cut pointer therefore cannot publish anything that was not committed.

### 2.5 Run adapter: `gatedtrain-runtime-agent` (package R)

One adapter per attempt, `gatedtrain-runtime-agent`, runs inside the allocation under its own OS user and owns every effect that needs authority: it holds the run lease (03), signs verbs, and turns contract files into store writes. It reaches outside the allocation only through a narrow authenticated RPC to `gatedtrain-runtime-svc` (store, broker, `ControlAPI`, `issue_epoch`; 03); store credentials never enter the job (R9-9, ARCH3-1). In P2, before leases exist, `gatedtrain-runtime-svc` runs with the workflow runner on a service node and performs the store effects without a run lease (03, 04). MAC keys and launch tokens are issued to the agent's user, not to the team UID, and each run directory carries per-run ACLs; the team UID remains the trust domain for the MVP as an accepted risk (R9-7, RT3-2; detail in 06).

| Contract input | Adapter action (03) |
|---|---|
| Heartbeat with advancing progress, or a phase within budget | Renew the attempt's lease (P3) |
| `cuts/<cut_id>.json` | Verify the manifest hash; commit by compare-and-set (§5.2) and place the cut in the agent-owned committed prefix. While fenced, queue the commit (§5.5) |
| `ledger/step_<N>` | Add one MAC per segment and upload it to the consumption ledger (05 §12.2); while fenced, queue |
| `completion.json` | Record the stage result, fenced by `controller_epoch` |
| Intent to pause or stop (03, 04), deadline signal, fencing timer | Sign and write `control/<seq>_<action_id>.json`, then replace `control/index.json` |

**Supervision (FMEA2-5).** The launcher starts the adapter as a child process (on Slurm, from the batch script on the Ray head node; on Kubernetes, as a sidecar with the equivalent rule defined by the Kubernetes plug-in in 03) and restarts it with backoff. A restarted adapter rebuilds its state from the store and the contract namespace; commits and verbs are idempotent by `cut_id` and `action_id`. The launcher records each successful renewal time the adapter reports, so a restarted adapter resumes the fencing timers of §5.5 from that time rather than from zero. If the adapter cannot be restarted within `adapter_down_max_s` (default 120 s), the launcher kills the run's process group and the attempt is classified `adapter_fault`.

**Release discipline.** The adapter version is pinned in the run's lockfile (04), so an attempt never changes adapter version mid-run. New adapter releases are canaried per cluster (a small share of runs on one cluster first, then cluster by cluster) and rolled back by re-pinning. Exits classified `adapter_fault` or `fenced` never mark a cut suspect or bad and never count toward crash-loop breakers (03).

In the research profile (overview D15) the adapter runs in local mode: the "store" is a pointer file updated by atomic rename under a file lock, with the same compare-and-set checks.

### 2.6 Liveness signals (FMEA2-3, OPS2-5)

The adapter judges liveness from the heartbeat and distinguishes four cases. 03 §4.2 and §4.4 own lease renewal and the cross-run correlation rule; this section defines the run-side signals they use.

| Observation | Meaning | Adapter response |
|---|---|---|
| Heartbeat readable, `seq` and a progress counter advancing | Alive | Renew |
| Heartbeat readable, `seq` advancing, every progress counter frozen past the phase budget | Run hung while its event loop lives | Request a stack capture, then withhold renewal (03) |
| Heartbeat readable, `seq` frozen past the phase budget | Controller event loop blocked or dead | Same as above, with scheduler state as the second signal |
| Heartbeat unreadable (I/O error or timeout from the transport) | Storage problem, not a run verdict | Do not withhold renewal on this evidence; consult storage verdicts (cluster health service, filesystem health); report the read failure |

**Checkpoint phases count I/O progress.** In `checkpointing` and `restoring`, `progress.io_units` advancing counts as progress, so a slow but moving save on a degraded filesystem is not a hang. *Unverified:* how finely NeMo RL exposes per-rank save completion to the controller; the minimum is one unit per component (policy, optimizer, data plane, each participant).

**Budgets from measurements.** Each phase budget is `max(floor, 2 × p99)` of measured no-progress gaps for that phase, keyed by cluster, filesystem, and model size class, from heartbeat history; until 20 samples exist, conservative defaults apply (07). Budgets live in the adapter's configuration, not in NeMo RL.

**Correlated stalls are infrastructure.** When more than `k` runs on one filesystem stall in the same window (03 defines `k` and the window), 03 declares a storage brownout: adapters on that filesystem stop treating stalls as run failures, extend budgets for the brownout's duration up to a bound, and alert the site's storage operator instead of recovering runs one by one.

### 2.7 Path V: verl-omni through an adapter

verl-omni writes `global_step_N` directories under `trainer.default_local_dir`, prunes them with `max_actor_ckpt_to_keep`, writes `latest_checkpointed_iteration.txt` unless saving asynchronously, and resumes with `resume_mode ∈ {auto, disable, resume_path}` (`verl_omni/trainer/diffusion/ray_diffusion_trainer.py:1126-1200`). A path V emitter, shipped as a verl plug-in that uses package A's emitter library, implements the contract without changing verl-omni:

- `default_local_dir = <run_root>/attempt_<A>/epoch_<E>`; `max_actor_ckpt_to_keep` unset; `resume_mode = resume_path` with the path from the verified `restore_cut` pointer, never `auto`.
- A trainer subclass wraps `_save_checkpoint`: after it returns, it hashes the step folder, writes the cut manifest and pointer, and emits ledger segments.
- A per-step hook writes the heartbeat and polls verbs at the step boundary.
- Cuts are trainer-only; in-flight rollouts are discarded and their prompts recorded for redispatch.
- *Unverified:* how verl-omni signals completion of an asynchronous save; until established, the emitter requires synchronous saves.

## 3. Sample provenance and the hand-off to lineage

Each rollout group carries a provenance record in controller-side replay metadata (`TQReplayBuffer.metadata_state_dict`, `replay_buffer.py:1778`), not in token partitions, so the LLM path adds zero bytes per token (overview §2.2, goal 3).

**Schema.** The record's fields are package A's `ConsumptionRecord` (05 §12.2); this document defines no ledger fields of its own (R9-6, LIN3-2). The controller epoch is not a row field: it is written once in each ledger-segment header and each cut manifest, and rows carry NeMo RL's integer counter (§7, NRL3-5).

**Hand-off.** Hook 8d is a per-step provenance export callback: after each optimizer step NeMo RL passes the rows of that step to the callback, and C-ext writes one ledger segment of package A's `ConsumptionRecord` rows to `contract/.../ledger/step_<N>`. 05 is authoritative for the `ConsumptionRecord` schema, the outcome enum, the segment format, and sizing; this document uses no other ledger schema. **8d ships consumed rows first** (outcome `consumed`); drop reasons follow in a later 8d PR because they need instrumentation at several drop sites (R9-3, NRL3-3). Once they ship, the run side uses these further values of 05's enum: `filtered`, `stale` (reason `version_age:<n>`), `dropped_at_cut`, `redispatched`, `storage_unit_lost`, and `refit_invalid` (only with hook 6b). **Integrity.** The agent adds one MAC per segment; the ledger writer verifies it and accepts a segment only when its hash is listed in a signed committed cut. Rows carry no MAC (R9-6, LIN3-3). Exit reasons such as `fenced`, `adapter_fault`, and `cut_impossible` are completion reasons or attempt classifications, never ledger outcomes. The next cut manifest lists the hashes of all segments since its parent cut, so a committed cut implies its lineage is on disk. The adapter uploads segments idempotently. Segments from an attempt whose cuts never commit are orphaned, and segments after the branch point of an attempt that was superseded by a rollback are marked superseded; consumption is defined by weight-version ancestry over `(controller_epoch, counter)` (05, LIN2-1). Each segment records `precision = row` for the NeMo RL path; path V records `step_range` where it cannot attribute rows (05).

## 4. Checkpoint cuts

### 4.1 Two kinds of cut, as today

- **Full cut**: the existing `_save_checkpoint_impl` path at an optimizer-step boundary, plus the cut manifest.
- **Rollout snapshot**: the existing `_save_rollout_checkpoint` path, anchored to a committed full cut, plus a manifest with `anchor_cut_id`.

Restore always reads a committed full cut and, optionally, a committed snapshot of that anchor. There is no global quiesce.

### 4.2 `CheckpointParticipant`

Hook 9 (P4) extracts the Gym adapter's verbs into a protocol that environments, reward services, simulators, and samplers implement:

```python
class ParticipantMode(Enum):
    CAPTURE = "capture"            # saves in-flight work items
    RESTART_ONLY = "restart_only"  # in-flight items restart from their task
    DRAIN = "drain"                # finishes in-flight items before commit, up to drain_max_s
    STATELESS = "stateless"        # nothing to save; drains or cancels requests

class CheckpointParticipant(Protocol):
    name: str
    mode: ParticipantMode
    version: str  # name@content-digest, recorded in the manifest
    async def prepare(self, cut_id: str, *, deadline_ts: float) -> PrepareInventory: ...
    async def renew(self, cut_id: str, *, deadline_ts: float) -> None: ...
    async def commit(self, cut_id: str, root: Path, items: Iterable[WorkItem],
                     *, deadline_ts: float) -> CommitSummary: ...
    async def resume(self, cut_id: str, *, deadline_ts: float) -> None: ...
    async def restore(self, cut_id: str, root: Path, items: Iterable[WorkItem],
                      *, source_cut_id: str, deadline_ts: float) -> None: ...
    async def forget(self, cut_id: str, item_ids: Iterable[str], *, deadline_ts: float) -> None: ...
```

Rules, all taken from today's Gym behavior: `prepare` stops admitting new work and reports in-flight items; a participant that loses `resume` reopens when its cut lease expires; `restore` must report the source cut it loaded (as `gym_checkpoint_adapter.py:460-473` checks); restore fails if a participant that saved state is missing (`gym_checkpoint_adapter.py:433-441`). `GymCheckpointAdapter` becomes the first implementation with no behavior change. `RESTART_ONLY` items are reported per cut, as `gym_restarted_episodes` is today (`single_controller.py:5555-5563`).

**Drain mode (PERF2-4).** A `DRAIN` participant stops admission at `prepare`, lets in-flight items finish until `drain_max_s`, commits with no in-flight items, and restarts only items still running at the deadline. The costs differ in when they are paid: `restart_only` costs nothing at the snapshot beyond the park, but loses the partial work of in-flight items on every **restore** (after a crash or a planned stop); `drain` costs idle GPU time on every **snapshot** and loses nothing on restore. The rule, with the cost model, is 08 §7.4 (authoritative): periodic snapshots use `restart_only`; snapshots taken for a `pause_and_checkpoint` before a planned stop, preemption, or upgrade (for example `then=exit`) use `drain`, because a restore is then certain. Samplers may declare `inflight_policy: auto`, which applies exactly this rule. The mode used and its cost are recorded in the cut manifest.

### 4.3 What is quiesced, and when

| Component | Full cut | Rollout snapshot |
|---|---|---|
| Trainer | Saves at its own step boundary; nothing else waits on it | Not paused; the snapshot is abandoned if an optimizer commit starts (`single_controller.py:5368-5372`, `:5464-5473`) |
| Non-colocated token engines | Keep serving (`single_controller.py:5938`) | Keep serving; only new dispatch waits while participants are parked |
| Colocated engines | Stood down for the save, as today | Not applicable |
| Data-plane mutations | Wait during the exclusive barrier hold | Same |
| `CAPTURE` / `RESTART_ONLY` / `DRAIN` participants | Not involved | `prepare` → (drain) → `commit` → `resume`; parked for that window only |
| `STATELESS` reward services | Never paused; late results enter as ordinary barrier mutations | Same; requests outstanding at the cut are re-issued after restore |
| Diffusion or simulator samplers (asynchronous mode, P4) | Not involved | `inflight_policy` of `restart_only` (periodic snapshots), `drain` (planned pauses), or `auto` (that rule, 08 §7.4); prompts of restarted items are recorded and redispatched after a restore |

The only window in which rollout work stops is the participant park in a snapshot, measured as `gym_parked_seconds` with a warning when it exceeds the interval (`single_controller.py:5600-5617`); drain time is part of the park and is reported separately. The colocated synchronous diffusion mode of P1b takes full cuts only, when no rollout is running.

### 4.4 Payload-heavy buffers

A stochastic trajectory cannot be regenerated to match its stored rewards and log-probs, so buffered rows are either persisted or dropped, never regenerated. The policy is `buffer_persistence ∈ {auto, persist, drop}` with default `auto`:

- Persist when the buffered payload is at most `persist_max_bytes` (default 64 GiB per run *(estimate, tuned in P4)*) and the expected exclusive-hold time × GPUs blocked is below the expected redispatch cost (formula with safety factor in 08 §4.3).
- Otherwise drop the rows at the cut, record their prompts for redispatch, and mark them `dropped_at_cut` in the ledger (once 8d's drop reasons ship, §3).

**Separate storage units (PERF2-5).** Video runs that persist the buffer use `mooncake_cpu` with `storage_unit_segment_size > 0`, so only CPU storage-unit actors own buffer memory and the trainer and rollout processes are off the save path (`interfaces.py:184-186`); units are placed with `storage_unit_placement` (§9). Every cut reports `barrier_hold_s` (duration of the exclusive barrier hold) and bytes saved. After three cuts, `auto` uses the measured p95 hold time instead of the bandwidth estimate below.

**Sizing for Wan-class video** *(estimate; payload model in 08 §4.3, rechecked in revision 8)*. A Wan 14B row at 81 frames, 480×832, with an SDE window of W = 2 holds three fp32 latent states plus its amortized prompt embedding, about 25.7 MB. With one fresh and one stale iteration buffered (256–512 rows), the buffer is **6.6–13.2 GB**. Writing and reading it at 2 GB/s per node on 4 storage-unit nodes takes about 1.6–3.3 s in total; holding 32 GPUs that long costs about 53–106 GPU-seconds. Dropping the same rows at about 160 sampling GPU-seconds per row costs 41,000–82,000 GPU-seconds (11–23 GPU-hours) per cut. Persistence is the default for this regime; drop applies above the threshold, for example long 720p clips with wide windows.

**Metric.** Every cut reports `checkpoint/gpu_seconds_lost` = exclusive-hold seconds × GPUs blocked + participant park seconds (including drain) × rollout GPUs idled + Σ generation GPU-seconds of dropped and restarted items. It is a manifest field and a telemetry metric, and the per-checkpoint goodput loss in the P4 exit criterion is computed from it at the registered cut interval (08).

## 5. Side effects and storage (DS-1, DS-6)

Epochs fence store writes, but checkpoint files and deletions are side effects on shared storage. Structure, not timing, makes them safe.

### 5.1 Epoch- and attempt-scoped paths

Hook 8b (P3) adds a contract-mode `CheckpointStorage` (protocol at `checkpoint_storage.py:44-62`) rooted at `<run_root>/attempt_<A>/epoch_<E>/`. Inside that root, `tmp_step_N` → `step_N` (`LocalStorage`) and in-place `step_N` plus `_SUCCESS` (`MscStorage`) work as today; `MscStorage.begin` deleting an unpublished prefix (`checkpoint_storage.py:110-120`) then only touches the attempt's own prefix. Snapshot roots are separated from the anchor: today `prepare_snapshot_paths` writes under the anchor (`rollout_checkpoint.py:548-576`); in contract mode it takes a writable root, so a new attempt that restores `attempt_7/.../step_100` writes snapshots under `attempt_8/.../step_100/rollout_snapshots/` and never into the old attempt's directory. The attempt root is created exclusively (`mkdir` on POSIX, a create-only marker object on object stores); if it exists, the run refuses to start. Before hook 8b, P2 uses the mechanism in §5.7.

### 5.2 A cut exists only when committed

**Committed prefix (R9-7, RT3-1).** Before committing, the agent copies the cut into `<committed_root>/<run_id>/<cut_id>/`, a prefix owned by the agent's OS user and read-only to the team UID (or write-once storage with object lock), and verifies the copy against the manifest hashes. A rename is not enough, because the team UID would still own the inodes. The attempt-directory original then counts as an orphan for retention (§5.4). The adapter commits the cut with one compare-and-set in the store (03):

```
if run.controller_epoch == manifest.controller_epoch
   and run.latest_cut == manifest.parent_cut_id
   and sha256(manifest) == pointer.manifest_sha256:
     run.latest_cut = cut_id; append cut_id to run.cut_history
```

A failed compare-and-set leaves the directory as an orphan that is never restored. A process of a superseded attempt can still write files, but only inside its own attempt directory and contract namespace, and none of them commits.

### 5.3 Restore and "latest"

In contract mode `get_latest_checkpoint_path` and directory listings are not used for resume. The launcher passes `restore_cut` (§2.1), resolved from the store pointer at job start, not at submission; the emitter library verifies the adapter signature and restore verifies the manifest hash before reading anything else.

### 5.4 Retention

Contract mode disables every run-side deletion: `remove_old_checkpoints` (both retention keys unset), snapshot pruning in `commit_snapshot`, and `prune_bootstrap_snapshots` (`rollout_checkpoint.py:467`, called at `single_controller.py:5423-5434`). Temporary directories of the run's own in-progress save are still cleaned. A store-side retention job (07, with dry-run and double-checked deletes) deletes only directories that are (a) orphans of attempts confirmed terminal, or (b) committed cuts outside the retention set: the newest K committed cuts (default 3), every cut referenced by a published artifact, every cut pinned by a gate under evaluation (03), and the fallback target of §6. Committed cuts are protected by the agent-owned prefix of §5.2, not by mode bits the team UID could reset.

### 5.5 Fenced behavior (FMEA2-1, RT2-5)

From P3 the adapter holds the attempt's lease (03). When it cannot renew before its local deadline, the adapter is fenced; the run is not told and changes nothing in its own behavior:

1. **Fenced compute, at most `fenced_compute_max_s`.** The adapter stops committing cuts, uploading ledger segments, and permitting deletions; it queues them. The run keeps training and refitting within its attempt, because refit is internal to the attempt and is not a fenced side effect. Cuts written in this window stay in the attempt's directory, uncommitted.
2. **Hold, at most `fenced_hold_max_s`.** The adapter writes `pause_and_checkpoint(then=hold)`; the run takes a cut at the next step boundary and stops computing while keeping its allocation. Without hook 8a-2 this step is skipped and the adapter goes to step 3.
3. **Stop.** The adapter writes `stop(mode=graceful, reason=fenced)`, and the launcher kills the process group if the run has not exited within `stop_grace_s`.

If a renewal with the same epoch succeeds during step 1 or 2, the adapter commits the queued cuts in order (each one's parent is the previous) and uploads the queued segments, and, if holding, writes `pause_and_checkpoint(then=continue)`; nothing is lost. If the epoch has advanced, the adapter discards the queue and stops the run. Signed `stop` and safety pauses from other sources are delivered in every state, because the run never consults the lease.

Fenced compute and hold time count against the run's GPU-hour budget, and the Slurm QOS cap counts them regardless. `safety_critical` runs use `fenced_compute_max_s = 0`, so they go straight to hold; physical benches additionally have a local watchdog that safe-stops on heartbeat loss without waiting for the adapter (03, 06). These bounds, together with `adapter_down_max_s` for a dead adapter, define the fencing bound `T_fence` that 03 §5.2 waits for before releasing a replacement (R8-6). Default values are in 03. `T_fence` does not yet bind orphan workers on other nodes or suspended VMs; that is DS3-1, a P3 entry blocker in `open-issues.md`.

### 5.6 Attempt isolation

A retried attempt gets a new `attempt_id`, a new `controller_epoch`, a new checkpoint directory, and a new contract namespace. The supervisor releases it only under 03's resubmit rule: the old job is scheduler-terminal and the fencing bound has elapsed, or the old processes are confirmed dead (`NODE_FAIL` counts as unconfirmed). If a bug ever ran two attempts at once, their writes would land in disjoint directories and only one could commit. This also removes today's shared-root hazards of `LocalStorage.begin`, `publish`, and `MscStorage.begin` (`checkpoint_storage.py:69-97`, `:118-120`).

### 5.7 P2: restore before hook 8b; requeue only in P2-full

There is no lease in P2. **P2-min** (the default, R9-2) submits jobs with `--no-requeue`; recovery is an operator `retry_stage`, which `gatedtrain-runtime-svc` submits as a new attempt with its epoch and restore cut already resolved (§2.1). **P2-full** adds automatic requeue (overview D8) only when the LLM pilot user confirms it needs automatic recovery in P2; only then does the job-start handshake (step 1) apply.

1. **Job-start handshake (P2-full only; part of A's run contract, R9-9).** The first line of the batch script, before `srun` and container start, calls `issue_epoch(run_id, job_id, restart_count)` on `gatedtrain-runtime-svc` (R9-8, FMEA3-4). It succeeds only under 03's conditions (current attempt of the stage, record `submitted` or `running`, workflow not cancelling, cap not reached, authorized by an authoritative `squeue -j`) and returns `controller_epoch`, the signed `restore_cut`, and `adapter_pubkey`. The job holds no store credential. On refusal the script exits with code 79 ("do not requeue") before any GPU work; if the service is unreachable it holds the job and exits (03, 04).
2. **Both variants.** Set `checkpoint_dir = <run_root>/attempt_<A>/epoch_<E>` with both retention keys unset, so `remove_old_checkpoints` is a no-op (`checkpoint.py:590-591`).
3. **Both variants.** Expose the resolved cut as `step_<N>` inside the new attempt directory as a copy-on-write copy of the committed-prefix cut (reflink where the filesystem supports it, otherwise a full copy), not a hard-link tree (R9-7, RT3-1): the copy is owned by the team and can be written without touching the committed cut, and NeMo RL's own "latest" listing finds exactly that cut. Rollout snapshots are not carried across attempts in P2. `MscStorage` runs wait for hook 8b.

**Temporary dependency (R9-9, ARCH3-7).** Step 3 relies on NeMo RL's checkpoint listing (`get_latest_checkpoint_path`, `checkpoint.py:682-691`) choosing the exposed `step_<N>` and not rewriting it on load. A P2 conformance test pins that behavior at the baseline commit; hook 8b's explicit restore cut (§5.1, §5.3) removes the dependency.

## 6. Restore, fallback, and cut freshness (FMEA-3, FMEA-8)

### 6.1 Restore assertions

Restore fails, rather than silently dropping state, when any of these fails: the pointer signature and manifest hash; component hashes (05); the bootstrap compatibility fingerprint (existing `validate_bootstrap_anchor`, `rollout_checkpoint.py:508-545`); rollout-recovery payload hash and replay inventory (`single_controller.py:1273-1321`); participant set and source cut (§4.2); every buffered row's `weight_version` counter ≤ the restored counter (§7); every row's `reward_versions` equal to the run's pins.

### 6.2 Fallback ladder

| Rung | Action | Cost |
|---|---|---|
| 1 | Newest committed snapshot fails: try older committed snapshots of the same anchor (as `resolve_latest_snapshot` does) | In-flight rollouts since that snapshot |
| 2 | No snapshot restores: restore the anchor's full cut only | Every in-flight rollout |
| 3 | Full cut fails: mark it `bad` in the store; the supervisor restores the previous committed cut in `cut_history` in a new attempt with a new controller epoch (§7) | Training since that cut |
| 4 | Two cuts in a row fail, or no committed cut is left | Escalate to a human gate (07) |

Restore failures are classified: `cut_corrupt` and `assertion_failed` are **non-retryable for that cut**; the cut is never restored again unless an operator clears the mark. `restore_infra` (storage unavailable, OOM while loading, and permission errors such as `EACCES` or `EROFS`, which the read-only committed prefix makes possible; R9-8, FMEA3-6) is retried on the same cut at most twice, then escalates without marking the cut bad. Attempts that ended as `adapter_fault` or `fenced` never mark a cut bad or suspect (§2.5).

### 6.3 Saving before the time limit

The launcher sets `checkpoint_must_save_by` to the scheduler time limit minus 1.5 × the 95th-percentile full-cut duration minus 5 minutes, so `TimeoutChecker` with `fit_last_save_time` saves in time (`timer.py:467-485`). On Slurm the `USR1` path of §2.3 is a second, independent trigger once hook 8a-2 exists. P2-min and P3 submit jobs without Slurm requeue; only P2-full keeps them requeueable (overview D8, R9-2).

### 6.4 Cut age (FMEA2-9)

The adapter computes the age of the last committed cut. It raises `cut_age_exceeded` when the age exceeds twice the registered cut interval, whether because saves fail, commits are rejected, or the run makes no progress (runbook in 07). When the age exceeds `max_cut_age_s` (default 4× the registered interval, 07 §5.4), the adapter writes `pause_and_checkpoint(then=continue)` with a deadline. If no cut can be produced by that deadline (the save fails or times out, typically a storage failure), the adapter writes `stop(mode=graceful, reason=cut_impossible)`: the run ends with a declared reason and recovery restarts from the last committed cut, instead of the alarm silently turning into a stop without a cut. Without hook 8a-2 the adapter raises the alarm only and an operator decides. Today only rollout snapshots expose a comparable signal (`seconds_since_last_success`, `single_controller.py:5644-5653`).

## 7. Weight versions and lineage identity (DS-5, DS2-5)

Weight versions stay NeMo RL integers (`_trainer_version`). No version packing, no `version_age` helper, no admission filter, and no change to the 28 modules that use `weight_version` or to samplers that subtract versions (`staleness_sampler.py:406`, `:439`).

- **Lineage identity** is `(controller_epoch, counter)`. Every attempt receives a new controller epoch at launch, so identities are unique across attempts even for ordinary recovery: an attempt that restores counter 100 and trains to 120 produces `(E+1, 101…120)`, distinct from the superseded attempt's `(E, 101…120)`.
- **Where the epoch lives.** The epoch is a launch input recorded once per cut manifest and once per ledger-segment header, and in the contract namespace path; it is not repeated per row or per group, where it would be a constant (R9-3, NRL3-5). Nothing per token changes.
- **Parent edge.** At attempt start the adapter records `(controller_epoch, parent_epoch, restored_counter)` from the restored cut, giving 05 the ancestry it needs for rollups.
- **No in-run rollback.** A rollback requested from outside is `stop` plus a new attempt from an older cut (§2.3); a rollback after repeated numerical failure is the same, requested by the run's failure class through the supervisor. The run never calls the store and never changes its epoch. Within an attempt versions only increase, so integer staleness arithmetic stays correct.
- **Restore assertion.** Every buffered row's counter is ≤ the restored counter; the rows were captured with the restored cut, so they belong to its ancestry by construction.

## 8. Refit verification and abort (FMEA-5, NRL2-4)

**Default.** Refit relies on today's checks: IPC and NCCL collective transports verify `update_success` and raise (`weight_sync/interfaces.py:112-115`), and the abort watchdog below.

**Opt-in digest (hook 6b, P4, after a prototype).** After each refit, every rollout shard computes a decomposable weighted checksum per parameter, Σ wᵢ · h(seed, global index i) mod 2⁶⁴, over the received tensors before any quantization; tensor- and pipeline-parallel ranks report partial sums that the controller adds per parameter, and the trainer computes the same once per version in the export name space. Because the digest touches every synchronizer and engine, it is a separate upstream ask, prototyped first on one synchronizer to measure cost (one extra read of the shard's weights, a few milliseconds at HBM bandwidth *(estimate)*). On mismatch the shard is marked `refit_invalid`, rows it generated at that version are dropped with that reason, and it receives no dispatch until a later refit verifies; two consecutive mismatches mark it unhealthy. The digest detects partial, stale, or corrupted refits; it is not a security control (06).

**Abort in new samplers.** Samplers in packages D and H that receive weights over collectives must arm `RefitAbortWatchdog` around them and check `fired`, because an aborted collective returns without raising. They raise `RefitAborted` carrying `REFIT_ABORTED_TOKEN`, and a shard caught mid-refit does not serve until a later refit completes (`refit_watchdog.py:27-33`, `:51-86`, `:309-345`). The conformance kit injects a dead peer during refit and checks both behaviors.

## 9. Data-plane failure domain (FMEA-4)

Storage units are Ray actors hard-pinned to the run's own nodes (`nemo_rl/data_plane/adapters/transfer_queue.py:884-917`; `nemo_rl/data_plane/mooncake_storage_unit.py:131-161`), chosen among the run's virtual clusters by `storage_unit_placement` (`interfaces.py:289-301`). Mooncake's SSD offload writes to a node-local SSD directory (`interfaces.py:108-126`), which does not change the failure domain.

**Guarantee.** Buffer contents are durable only as of the last committed cut, and only when that cut persisted the buffer (§4.4). A node loss that ends the run loses every row written since that cut. A node loss the run survives in-process, such as a rollout shard, loses the rows held by storage units on that node. Lost rows are recorded as `storage_unit_lost` (once 8d's drop reasons ship, §3); token rollouts are redispatched through the rollout-recovery ledger, and other prompts through the redispatch list.

**Options.** (a) Persist the buffer at every cut (§4.4), which bounds the loss to one cut interval. (b) Place storage units on nodes whose loss already ends the run, such as the trainer cluster, or on a small dedicated node group, using `storage_unit_placement`, so rollout-shard losses lose no rows. (c) A storage service outside the run's allocation is not supported by NeMo RL today and is not requested.

## 10. Upstream changes, phases, and tests

| Change | Hook, phase | Test |
|---|---|---|
| Contract sink: heartbeat with progress and I/O counters, cut manifest, completion manifest; `stop` read through the sink (verification inside A's library) | 8a-1, by P2 (C-ext shim first allowed) | Contract conformance: invalid signature or foreign identifiers rejected; duplicate `action_id` is a single action; hung trainer shows advancing `seq` with frozen progress; `stop` executed while the adapter is fenced |
| `pause_and_checkpoint` with `continue`, `hold`, `exit` | 8a-2, after a prototype | Watchdog test: a long `hold` trips neither the stall watchdog nor collective timeouts; `continue` resumes without a new cut |
| Per-step provenance export callback | 8d, P2 (C-ext shim first allowed): consumed rows first, drop reasons later | Segment rows equal the rows the step consumed (later also dropped); epoch only in the segment header; segment hashes listed in the next cut |
| Contract-mode `CheckpointStorage` for `LocalStorage` and `MscStorage`; separate snapshot root; run-side deletion off; explicit restore cut | 8b, P3 | Partition test: an old attempt writes `step_120` and tries to garbage-collect, and the new attempt restores `step_100` intact, on POSIX and object storage |
| `CheckpointParticipant` with `DRAIN` | 9, P4 | Existing Gym recovery tests unchanged; a lost `resume` heals through the lease; drain and restart-only costs recorded |
| Fleet-health callback; opt-in refit digest | 6 (P0, after G0), 6b (P4) | Bit-flip and partial-refit injection mark the shard invalid (6b only) |

Framework-side tests (packages A, R, C-ext): both transports pass the same conformance suite (create-only races, atomic replace, unreadable files); adapter restart mid-run loses no commit and resumes fencing timers; fenced compute → hold → stop timings; brownout injection on one filesystem produces one infrastructure incident, not per-run recoveries; P2-full `issue_epoch` refusals for a finished, cancelled, or superseded job exit 79 before `srun`; restore from a read-only committed prefix (`EACCES`/`EROFS` classified `restore_infra`); the P2 restore conformance test of §5.7 at the baseline commit.

## 11. Changes

### From revision 8 to revision 9 (closing)

Only the round-3 MVP findings were applied; the Round-3 disposition below lists each change with its section. P2-min is the default and every P2-full-only step is marked (§2.1, §5.7, §6.3).

### From revision 7 to revision 8

1. NeMo RL no longer reads `lease.json`, self-fences, rejects verbs while fenced, signs, or gates commits; the R adapter does all of it, and `lease.json` and the `fenced` completion status are removed (§2, §5.5).
2. Fenced behavior is adapter-side and staged: compute with refit, then hold in place, then stop; fenced time counts against budgets; `safety_critical` skips compute (§5.5).
3. Verbs are Ed25519-signed by the adapter and verified by the emitter library with a launch-time key; `pause_and_checkpoint` gains `then=hold`; `stop` carries a reason; `USR1` is handled by the launcher (§2.3).
4. Contract paths are scoped per attempt and epoch; a transport interface has POSIX and object-store implementations, covering Kubernetes and `MscStorage`; polling load is bounded (§2.2).
5. Liveness separates unreadable files from frozen counters, counts checkpoint I/O progress, takes budgets from measured p99, and defers correlated stalls to 03's brownout rule (§2.6).
6. The adapter is supervised by the launcher, pinned in the lockfile, and canaried per cluster; adapter and fenced exits never mark cuts suspect (§2.5, §6.2).
7. Version packing, `version_age`, the admission filter, and in-run rollback are removed; lineage identity is `(controller_epoch, counter)` with a new epoch per attempt (§7).
8. The refit digest becomes opt-in hook 6b (§8).
9. Ledger segments use package A's `ConsumptionRecord` and 05's outcome enum via hook 8d; 05 owns the schema; cut manifests record `last_step_included` and per-file hashes, with Merkle roots from P3 (§2.4, §3).
10. Samplers gain a drain mode; video persistence uses separate storage units with measured barrier hold; sizing uses 256–512 rows, 6.6–13.2 GB (§4).
11. When no cut is possible, the cut-age rule stops with a declared reason (§6.4).
12. P2 requeue uses `issue_epoch` and resolves the restore cut at job start; restore before hook 8b uses a hard-link tree (§5.7; superseded in revision 9).

### From revision 6 to revision 7

The run stopped holding leases, writing the store, enforcing budgets, and opening gates; global quiesce was removed; cuts became store-committed with scoped paths, fallback, and a cut-age alarm (§2, §4–§6, §9).

## Finding disposition

| ID | Section | Resolution |
|---|---|---|
| ARCH-1 | §2, §2.5 | NeMo RL only emits contract files and executes two verbs; leases, store writes, budgets, and gates sit in the package R adapter and in 03 (updated in revision 8). |
| ARCH-6 | §4.2 | The Gym checkpoint adapter becomes the first implementation of a generalized `CheckpointParticipant` protocol with unchanged behavior. |
| NRL-2 | §1, §4.1, §4.3 | Full cuts and rollout snapshots reuse `_save_checkpoint_impl` and `_rollout_checkpoint_pump` unchanged; non-colocated engines keep serving and only participants park during snapshots. |
| DS-1 | §5 | Checkpoints live under attempt- and epoch-scoped paths, exist only after a store compare-and-set, are restored from a signed store pointer, and are deleted only by store-side retention. |
| DS-5 | §7 | Superseded in revision 8: versions stay integers, lineage identity is `(controller_epoch, counter)` with a new epoch per attempt, and rollback is always a new attempt. |
| DS-6 | §5.6 | Each attempt has its own directory, contract namespace, and epoch, and is released only under 03's resubmit rule. |
| PERF-1 | §4.4 | Buffers of 6.6–13.2 GB *(estimate, corrected in revision 8)* are persisted by default through separate storage units; drop-and-redispatch applies only above a threshold; `gpu_seconds_lost` is reported per cut. |
| FMEA-3 | §6.1, §6.2 | Failed restores walk a fallback ladder to the previous committed cut in a new attempt, and a failing cut is classified non-retryable. |
| FMEA-4 | §9 | The buffer's durability is stated as "as of the last committed cut", with persistence and storage-unit placement as mitigations. |
| FMEA-5 | §8 | Superseded in revision 8: the per-shard digest is opt-in hook 6b; new samplers implement the in-worker refit abort. |
| FMEA-8 | §6.3, §6.4 | The save deadline is derived from the scheduler limit with a signal backup through the adapter, and an alarm fires when the last committed cut is older than twice the interval. |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| ARCH2-2 | §2, §2.1, §7 | The run has only launch inputs and the contract transport; it reads no lease, never calls the store, and receives its epoch at launch. |
| ARCH2-6 | §2.2, §5.1 | A transport interface with POSIX-directory and conditional-put object-store implementations covers Kubernetes and `MscStorage`, and polling is bounded to under 0.5 metadata operations per second per run. |
| DS2-A | §2.2, §5.6 | Contract files live in a per-attempt, per-epoch namespace that only the current attempt's adapter reads, so an old process cannot overwrite a new attempt's files. |
| DS2-5 | §7 | Every attempt gets a new controller epoch, so `(controller_epoch, counter)` is unique across ordinary recoveries and rollbacks without any rollback epoch protocol. |
| NRL2-2 | §1, §2, §7, §10 | Hook 8 is split into 8a, 8b, and 8d; NeMo RL has no lease, signature, signal-handler, or version-packing code, and `weight_version` is untouched. |
| NRL2-4 | §8 | The refit digest is a separate opt-in hook 6b, prototyped on one synchronizer in P4. |
| FMEA2-1 | §5.5 | Refit is not fenced; fenced runs keep training and refitting, then hold in place, then stop, each with a bound. |
| FMEA2-3 | §2.6 | Unreadable files are not stall evidence, checkpoint phases count I/O progress, budgets come from measured p99, and correlated stalls on one filesystem become one infrastructure incident (03). |
| OPS2-5 | §2.6, §2.2 | Same as FMEA2-3, plus guidance to keep the contract off the checkpoint filesystem and a contract-read-failure SLI. |
| FMEA2-5 | §2.5, §6.2 | The launcher supervises and restarts the adapter, versions are pinned and canaried, and adapter-caused or fenced exits never mark cuts suspect (run-side part; breaker rules in 03). |
| FMEA2-9 (checkpoint age) | §6.4 | When no cut can be produced, the adapter stops the run with `reason=cut_impossible` and recovery uses the last committed cut. |
| PERF2-4 | §4.2, §4.3 | Participants gain a `DRAIN` mode; periodic snapshots use `restart_only` (lost work paid per restore) and planned pauses use `drain` (idle paid per snapshot), per 08 §7.4. |
| PERF2-5 | §4.4 | Persisted video buffers use separate storage units (`storage_unit_segment_size > 0`), `barrier_hold_s` is measured per cut, and sizing uses 256–512 rows. |
| RT2-2 | §2.3, §2.4, §5.3 | Verbs and restore pointers are Ed25519-signed by the adapter and verified by the run; run-written files are treated as claims, and outputs publish only if they match committed cuts. |
| RT2-5 | §2.3, §5.5 | The run never consults the lease, so `stop` and safety pauses are executed in every state, and fenced time counts against the cap. |
| DS2-B, FMEA2-2, R8-6 (run side) | §5.5, §5.6 | The fencing bound (compute, hold, stop grace, adapter-down) is defined here for 03's resubmit rule. |
| DS2-C, FMEA2-6, RT2-3 (run side) | §5.7 | The P2 start script calls `issue_epoch`, holds no store credential, and resolves the restore cut at job start. |
| LIN2-1, LIN2-2 (run side) | §3, §7 | Ledger segments use package A's single `ConsumptionRecord` schema and outcome enum via hook 8d, with epoch parent edges for ancestry rollups. |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| NRL3-3 | §3, §10 | Hook 8d ships consumed rows first; drop reasons, which need several drop sites instrumented, follow in a later PR. |
| NRL3-4 | §2, §2.3, §10 | Hook 8a splits into 8a-1 (emitters, `stop`) and 8a-2 (`pause_and_checkpoint` with `hold`/`continue`, after a prototype with a watchdog test); each use of the verb has a stated fallback until 8a-2. |
| NRL3-5 | §3, §7 | The controller epoch is recorded once per cut manifest and ledger-segment header, not per row. |
| LIN3-2 | §3 | 02's own field table is removed; the record is package A's `ConsumptionRecord` (05 §12.2). |
| LIN3-3 | §2.5, §3 | One MAC per segment, added by the agent and verified against signed committed cuts; no per-record MAC. |
| RT3-1 | §5.2, §5.4, §5.7 | Committed cuts are copied into an agent-owned prefix (or write-once storage); restore uses copy-on-write copies, not hard links. |
| RT3-2 | §2.5 | MAC keys and launch tokens are issued to the agent's user, run directories carry per-run ACLs, and team UID = trust domain is accepted MVP risk (detail in 06). |
| FMEA3-4 | §5.7 | `issue_epoch` is the batch script's first line, before `srun` and container start; refusal exits 79 before any GPU work (P2-full). |
| FMEA3-6 | §6.2 | `EACCES`/`EROFS` on restore are `restore_infra`, retried without marking the cut bad. |
| ARCH3-1 (02 part) | §2.5 | The in-allocation adapter is `gatedtrain-runtime-agent`; it reaches `gatedtrain-runtime-svc` only through a narrow authenticated RPC, and store credentials never enter jobs. |
| ARCH3-7 | §2.1, §5.7 | The job-start handshake (`issue_epoch` request and response, exit 79) is part of A's run contract; the P2 restore's reliance on NeMo RL's checkpoint listing is covered by a conformance test and removed by hook 8b. |

**Deferred** (recorded in `open-issues.md`; this document's text stands until the owning phase updates it): DS3-1 (`T_fence` and orphan workers or suspended VMs; P3 entry blocker, §5.5) and LIN3-5 (P2 manifest hashes vs P3 Merkle roots; P3, §2.4).

## Notes for other documents

Revision-9 notes (all applied in revision 9): 05 §12.2 already keeps the MAC per segment; its key should take `controller_epoch` from the segment header rather than repeat it per row (R9-3, R9-6); 03 §3.3 should mark `issue_epoch` and requeue as P2-full and name `gatedtrain-runtime-svc` (R9-2, R9-9); 06 holds the agent-user, per-run ACL, and accepted-risk detail referenced from §2.5 (R9-7).

All revision-8 notes are resolved (overview §5 and §7.1; 03 §3.3, §4.2–§4.3, §5.2; 05 §12.3; 07 §4.6, §4.7, §5.4).
