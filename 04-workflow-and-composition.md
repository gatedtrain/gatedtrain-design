# 04. Workflow and Composition

**Status:** The design set is at revision 26; this document was last changed in revision 18. A prototype (`gatedtrain` 0.6.0) implements revision 26 except the training-seed unit of revision 20 (05 §3.3). Frozen until G0 closes; deferred items in `open-issues.md`. §1–§8.1: proposed design for P2. §8.2 (engine backends beyond P2) and §9 onwards (composition): **Direction — detailed review after G0** (R8-20).
**Revision:** 9 (2026-11-05); applies the MVP decisions of the round-3 review (record not published) that touch this document (R9-2, R9-5, R9-8, R9-9). Revision 8 implemented the round-2 review (record not published); revision 7 replaced revision 6 §9 and §13.2 (archived revision 6, not published).
**Packages:** E `gatedtrain-workflow` (depends on A, R, E-stats, NeMo-Run; thin in P2, production engine backend in P9); F `gatedtrain-compose` (depends on C-ext, E, G; P10, only after the pilot in §9.2). R (`gatedtrain-runtime-agent`: the run adapter in the allocation; `gatedtrain-runtime-svc`: store interface, broker protocol and its implementation, `ControlAPI`, `issue_epoch`, run-record schema; R9-9) sits below E and G and is specified in 03; the contract transport the adapter uses is in package A (02).
**Related:** overview §3–§8; 01 (decision steps, plug-in contracts); 02 (run contract, cuts); 03 (package R, store, broker, leases, `issue_epoch`); 05 (gate statistics, human feedback, lineage); 06 (identities, accounts, secrets); 07 (operations); 08 (execution modes, cost model)

## 1. Scope

This sub-design covers the workflow layer (spec, semantics, IR, runner, engine backends), gate semantics as seen by workflows, the composition package, alternating training, and the joint-training direction. It does not define statistical procedures (05), the store, broker, `ControlAPI`, or run adapter (package R, 03), or the run contract and its transport (02; package A); it references them.

Terms. The **workflow layer** is framework code in package E: the spec validator, the compiler to the IR, step implementations, and all workflow writes to the store, made through R's store interface. An **engine** is the external or minimal system that provides durable execution through a `WorkflowBackend`. "Store" means the metadata store of 03.

## 2. Two levels of loop and reference patterns

- **Inner loop** (within one run): owned by the trainer and its algorithm plug-in. It talks to the outside only through the run contract (02).
- **Outer workflow** (across runs): stages that consume and produce typed artifacts, separated by gates.

| Pattern | Stages | Iteration | Human touchpoints |
|---|---|---|---|
| LLM reasoning | data → SFT → GRPO with verifiable rewards → eval gate | Optional SFT/RL alternation | Spot checks |
| LLM alignment | data → SFT → preference labeling → RM training → RL or DPO → eval gate | Re-label each cycle | Preference labeling |
| Image/video diffusion | data (latent cache) → SFT/LoRA (optional) → Flow-GRPO/NFT or Diffusion-DPO → pre-distillation gate → step distillation → final gate | RL then distillation | Human rating |
| VLA / driving | logs → behaviour cloning → closed-loop RL in simulation → benchmark gate → real-world evaluation | Long cycles | Safety review |
| CAD / mold | shapes and code → SFT → GRPO with sandboxed CAD rewards → engineering review gate | Add failure cases | Engineer review |
| Flight control | demonstrations → behaviour cloning → PPO in vectorized simulation → robustness gate → SITL → HIL → supervised flight | Return to simulation on failure | Safety pilot |

The P2 exit pipeline is the first row (SFT → GRPO → gate) for the LLM pilot user (overview §7.1).

## 3. Workflow spec

The YAML that users write is a one-way door (overview §8). Gates are typed tests from the gate library (05); every blocking gate declares its margin or bar and the power it must reach at a declared minimum detectable effect (MDE).

**α fields (EVAL2-6).** `alpha` is one-sided and is set in the gate's registration (05 §3.7); 0.025 (05 §3.2) is only the value `plan()` proposes when it drafts a registration. A gate that belongs to a registered family (turn gates of a loop, §10.3; the final gate) takes its α from that family's schedule in 05 (Bonferroni over the known bound, online FDR, or family-wise control for the final gate, one-sided `alpha_fw` = 0.025 by default); a gate inside a family must not set its own `alpha`, and the validator rejects one that does (05 §3.2). The report shows the α actually used.

```yaml
workflow: image-diffusion-alignment
version: 1
budget: {gpu_hours: 2000, labeling_usd: 5000}
artifacts:
  base:    {type: model_checkpoint, uri: "hf://stabilityai/stable-diffusion-3.5-medium@<revision>"}
  prompts: {type: prompt_dataset, uri: "s3://.../prompts@sha256:<hash>"}
  heldout: {type: item_set, uri: "s3://.../heldout@sha256:<hash>", disjoint_from: [prompts]}   # evaluation items (05 §3.3)
  pre_distill_reg: {type: registration, uri: "s3://.../pre_distill_gate.registration@sha256:<hash>"}  # 05 §3.7
  final_reg:       {type: registration, uri: "s3://.../final_gate.registration@sha256:<hash>"}
stages:
  - id: rl
    uses: rl.online@1.2
    algorithm: flow_grpo
    model_family: diffusion.sd3@0.3
    rewards: [{fn: pickscore@1.0, weight: 0.6}, {fn: ocr@1.1, weight: 0.4}]
    inputs: {model: base, prompts: prompts}
    outputs: {model: rl_ckpt}
  - id: pre_distill_gate
    uses: gate.threshold@1.0           # H0: geneval_overall <= bar (05)
    evaluator: geneval@1.0             # T2 for this recipe; independence checked by lineage (05)
    sampler: deployment
    inputs: {model: rl_ckpt, items: heldout, registration: pre_distill_reg}
    test: {metric: geneval_overall, bar: 0.80, alpha: 0.025, power: 0.80, mde: 0.02, cluster_by: item, seeds: 5}
    on_fail: stop
  - id: distill
    uses: distill.dmd@0.2
    inputs: {teacher: rl_ckpt}
    outputs: {model: fast_ckpt}
  - id: final_gate
    uses: gate.human@1.0               # T4 evaluator
    inputs: {model: fast_ckpt, baseline: base, items: heldout, registration: final_reg}
    test: {type: threshold, metric: pref_winrate_vs_baseline, bar: 0.55, mde: 0.05, power: 0.80}
    alpha_family: {name: final, alpha_fw: 0.025}  # one-sided, family-wise; one clause, so alpha = 0.025 (05 §5)
    labels: {sample_size: from_power, deadline: 72h, escalate_after: 48h, deadline_policy: wait}
    on_timeout: pause
    owner: image-quality-team
    escalation: [image-quality-lead]
```

**Final-gate labeling cost (EVAL2-8).** The final gate tests H₀: win rate ≤ 0.55 at one-sided family-wise α = 0.025 with 80% power at 0.60 (bar plus MDE): n = [(1.960·√(0.55·0.45) + 0.842·√(0.60·0.40)) / 0.05]² ≈ 770 independent comparisons, one per held-out item, multiplied by the design effect when several comparisons share an item (05 §3.6). The validator converts this count with the labeling vendor's per-label price into `labeling_usd` and rejects the spec if it does not fit. (05 §3.6's 783 is the 0.55-versus-0.50 case; at α = 0.05 this gate would need about 606.)

## 4. Semantics owned by the workflow layer

### 4.1 Artifacts, loops, pauses, budgets, resume

- **Typed artifacts.** Every input and output has a declared type and a content-addressed manifest (02, 05). A stage consumes artifacts only by manifest hash; the lockfile records every hash, plug-in digest, backend version, compiler version, and the run-adapter release (R). A gate's pre-registration (05 §3.7) is an artifact of type `registration`, bound to the gate's `registration` input role by content hash (`@sha256:`); a blocking gate without one is rejected at validation.
- **Loops** declare `max_iterations` and `max_attempts` (the bound on body executions, including failed, skipped, and no-op ones). Both are validated at configuration time against the budget.
- **Pausing** sends `pause_and_checkpoint` (02) and releases capacity by default; `hold_capacity: true` requires a maximum duration.
- **Budgets.** P2 enforces a hard GPU-hour cap per workflow with a Slurm QOS backstop (overview §7.1); runtime burn monitoring and broker-side enforcement arrive with G (03, 07). The validator rejects a spec whose estimated GPU or labeling cost exceeds its budget.
- **Resume** restarts from the store state; a workflow pins the framework version and stays resumable on version N+1 (§7).

### 4.2 Gates

A gate is a node that runs a typed test from the gate library (05) over evaluator results and then either passes, fails, or times out.

| Gate type | Null hypothesis | Required parameters |
|---|---|---|
| `gate.threshold` | metric ≤ bar | `bar`, `power`, `mde`; `alpha` (in the registration) or a family |
| `gate.superiority` | new − baseline ≤ 0 | `power`, `mde`; `alpha` (in the registration) or a family |
| `gate.noninferiority` | new − baseline ≤ −margin | `margin`, `power`; `alpha` (in the registration) or a family |
| `gate.equivalence` | \|new − baseline\| ≥ margin (TOST) | `margin`, `power`; `alpha` (in the registration) or a family |
| `gate.human` | as one of the above, on human labels | as above, plus `labels` (§4.4) |

Each workflow gate type compiles to one 05 clause (05 §3.1): `gate.superiority` → `superiority`, `gate.noninferiority` → `non_inferiority`, `gate.equivalence` → `equivalence`, and `gate.threshold` → `superiority` against the constant `bar`. Spec fields map to clause fields: `mde` → `planning_effect`, `cluster_by` (`item` or `training_seed`) → `unit`, `seeds` → the number of `eval_seeds`, `constraint_set` → `constraint_set`; `alpha` comes from the family schedule when the gate is in a family (§3). The registration (05 §3.7) is authoritative for every clause field it holds (05 §3.1: type, metric, direction, margin or bar, α, power, planning effect, σ plan, unit, K, evaluation seeds, interval, constraint set) and for the sampler and evaluator: the parameters above are required there, the same fields in the spec are optional, and a spec field that maps to one of them and differs from the registration is rejected at validation. A gate's evaluation items come through its `items` input role, an artifact of type `item_set`; training prompt datasets keep type `prompt_dataset`.

**Constraint gates.** A `gate.threshold` or `gate.noninferiority` with a `constraint_set` compiles to a 05 constraint clause (05 §3.9): the metric is the per-item violation indicator, `direction: lower_is_better`, `bar` is the highest acceptable violation rate, and `margin` the acceptable increase over the baseline's rate. There is no separate gate type, because the hypothesis forms are those of the threshold and non-inferiority types; only the metric and the blocking-evaluator rule (rule 2) differ. A constraint gate blocks beside the quality gate on the same candidate, which is the `and` of 05 §3.9.

**Gates with several clauses.** A gate whose registration lists several clauses (05 §3.1, §3.5) names one evaluator per clause: its `evaluator` is the list of the clauses' evaluators in the registration's clause order, each producing its own paired evaluation records. Its `uses` names the type of the first clause; the types of the others come from the registration, which is authoritative. A `test` field the spec states applies to every clause and must equal each clause's value. The gate passes only when every clause passes (05 §3.5).

Rules owned here; procedures owned by 05:

1. **Statistics.** Clustering (item and seed), confidence intervals, sample-size planning (05 uses an upper confidence bound of σ or an internal pilot), and α allocation across a family of gates are defined in 05. The workflow layer records the reported power and MDE with each result. A blocking gate that does not pass and whose achieved power at the registered planning effect (the `mde`, 05 §3.6) is below the registered power returns `underpowered` (05's INCONCLUSIVE), handled like `on_fail`; an underpowered pass is a PASS flagged in the report (05 §3.1). Only PASS opens a blocking gate; INCONCLUSIVE and MISSING (05 §3.8) follow `on_fail` and are reported as `underpowered` and `missing`.
2. **Independence and contamination.** Every blocking gate must use at least one evaluator that 05 grades as independent of all upstream training rewards and partner policies; contamination checks are a prerequisite of every gate (05). LLM and VLM judges need injection canaries and are never the only blocking evaluator: a non-LLM evaluator (tier T2 or T3) must block, and where none exists for a metric, human evaluation (T4) blocks and the judge is non-blocking (06, EVAL2-6). The one exception is a constraint gate whose judge meets 05 §6.1 (a)–(e): validity, independence, canaries in every judged batch, reproducibility, and egress approval with a pinnable version. The validator rejects violating workflows (P2 exit criterion).
3. **Outcomes.** `on_fail` and `on_timeout` take `stop`, `pause`, or `reject`. `reject` is valid only for a gate inside a loop body: nothing is published, the iteration is recorded as rejected, and the loop dispatcher continues.
4. **Fields.** `owner`, `escalation`, optional `quorum`, `timeout`, `escalate_after`. The owner of a gate decides it through `decide_gate`; the operator role cannot (06).
5. **Sampler.** A gate evaluates each model with its declared deployment sampler unless the gate type says otherwise.

### 4.3 Gate atomicity, pins, and timing (DS-7, DS2-D, DS2-7)

A gate has a deterministic step key `(workflow_run_id, node_id, loop_iteration, attempt)`. Its execution has three phases:

1. **Pin.** The step resolves the candidate's committed cut (02), fixes its manifest hash `h`, and writes a **pin record** `(step key, cut, h)` in the store by compare-and-set on the workflow epoch. Retention (the store-side retention job, 07) never deletes a pinned cut: it first moves a cut from `committed` to `retiring` by compare-and-set, skipping pinned cuts, and deletes only `retiring` cuts; a pin on a `retiring` cut fails as below, so a pin and a deletion cannot both succeed (DS3-5). A pin is released by the decision transaction when the gate fails, rejects, or times out, and by workflow termination; on pass the accepted-version pointer takes over retention of `h`. If the cut is no longer committed when the pin is attempted, the step fails without evaluating.
2. **Evaluate.** The step runs the evaluators on the pinned cut with the registration's `eval_seeds` (common random numbers, 05 §3.3) and writes the raw results as an artifact keyed by the step key. If such an artifact already exists, it is reused; nothing is resampled.
3. **Decide and publish.** The step computes the test (05) and commits **one store transaction**, conditioned by compare-and-set on the workflow epoch, on the loop record's version, and on the pin still being held, that:
   - inserts the gate result row under the step key (decision, statistics, power, evaluator versions, pinned hash `h`, deciding principal for human gates);
   - on pass, publishes the accepted version by setting the node's accepted-version pointer to the pinned manifest hash `h`;
   - updates counters (rejections, low-signal skips, consecutive failures, attempts used, budget consumed) and releases or hands over the pin;
   - advances the loop state (iteration, next body, or terminal status).

**Timeout versus decision.** A gate's deadline is a wall-clock time stored with the gate. Whether a decision or a timeout wins is decided by the **store commit timestamp** of the transaction (single primary, no asynchronous replicas; 03, R8-18), never by an engine, runner, or client clock: both transactions first lock the gate row; the timeout transaction commits only if no result row exists and its commit timestamp is at or after the deadline; a decision commits only if no timeout row exists and its commit timestamp is before the deadline (DS3-4). A human decision submitted before the deadline but committed after it loses and is recorded as late.

On resume, a step whose gate result row exists returns that result and performs no further writes; a re-evaluation would be a second test on the same hypothesis and inflate α. A transaction that fails its compare-and-set means another execution has authority; the step exits without writing. Publication always refers to the pinned hash, never to "the latest checkpoint" of a run.

A new attempt of a failed stage (`retry_stage` intent or an `on_fail` retry rule) increments `attempt` by compare-and-set only after the previous attempt is terminal by the protocol in 03; in P2 the previous Slurm job must also be absent from the queue with no requeue pending (R8-9). The new attempt therefore never shares a run directory or gang with a still-running predecessor.

### 4.4 Human gates and human labels (RES-7)

Revision 6 §13.2 masked unlabeled samples at the deadline, which biases toward samples that are quick to label. Revisions 7 and 8 never mask silently.

- **Human gates** wait. `escalate_after` notifies the escalation chain; at `deadline` the gate applies `on_timeout` (§4.3 timing). It decides on fewer labels than planned only if the missingness check (05 §3.4, `pairing.max_unpaired`) passes and achieved power at the MDE still meets the declared power; otherwise it returns `underpowered`.
- **Human-label rewards** (the connector, a `RewardFn` with a deadline; 05 owns the design) choose one `deadline_policy`:
  - `wait`: the batch waits up to a hard timeout, after which an `external_dependency` failure is raised (07); escalation fires at `escalate_after`.
  - `propensity`: labels received by the deadline are used with inverse-propensity weights from a label-arrival model with declared covariates (for example output length, prompt category, labeler queue); estimator, clipping, and diagnostics are in 05.
- Labels that arrive after a sample exceeds the staleness bound are stored for reward-model training only, with their arrival time.
- Active-sampling selection probabilities are recorded for inverse-propensity weighting (05).

## 5. Execution rules

These rules hold for every backend, including the P2 runner.

1. **The store is the only source of truth.** Lineage, accepted versions, pins, loop state, gate results, and workflow status live in the store. Each workflow run holds a lease and a **workflow epoch**; every write is a compare-and-set on the epoch and on the version of the record it changes. Engine state is never read as a record; if engine and store disagree, the store wins and the run is resumed from the store.
2. **A stage is submit plus await, and the engine never retries GPU work.**
   - `submit` calls the placement broker (package R) with the step key `(workflow_run_id, stage_id, loop_iteration, attempt)` and the workflow epoch as an opaque fencing token. Idempotent upsert by key, resolution of in-doubt submissions by lookup, the "unknown" result, held submission and release, and cancel-versus-submit ordering are defined in 03. The broker knows nothing about workflows (overview §4 rule 5).
   - `await` waits until the stage's run reports completion through the run contract (02), read by R's adapter over the authenticated contract transport (package A) and verified against the adapter's signatures: the completion manifest is recorded in the store by the component 03 assigns (the P2 runner before P3). Heartbeat progress counters drive stall detection; unreadable contract files are an I/O condition, not a stall (03). `await` holds no process for the stage's duration and re-checks the store on a fallback interval and after any engine recovery.
   - Recovery of GPU work belongs to the run supervisor (03). P2-min (the default, §8.1) has no automatic recovery: jobs run with `--no-requeue`, and an operator recovers a failed stage with `retry_stage`. In P2-full, the health service's Slurm requeue of the untagged job is the only automatic recovery (overview D8), admitted only through `issue_epoch` (§8.1). A run that still fails or stalls is handled by the stage's `on_fail` rule.
3. **Decisions enter only through `ControlAPI`** (package R). Human gate decisions and operator actions (`decide_gate`, `retry_stage`, `cancel_workflow`, `resume_workflow`, `extend_loop`) are authenticated, authorized (06), and written to the store with the acting principal before anything else happens. The engine receives only a wake-up hint carrying the record ID; the step re-reads and re-authorizes the record. A signal sent directly to the engine has no effect.
4. **The engine holds no secrets or tenant data.** Step inputs and outputs are store references, never URIs, credentials, or metrics. Only the workflow layer's service identity starts, signals, or terminates executions (06). Jobs hold no store credential (§8.1).

### 5.1 Resume, reconciliation, and cancel

- **Resume:** acquire the workflow lease; terminate the old execution if reachable, otherwise wait for its lease to expire; increment the epoch; start a new execution from the store. In-flight stages are re-adopted by step key; gates are re-adopted by their pin and result rows (§4.3).
- **Reconciliation** (on start and on a fixed interval): store `running` with no live execution → resume; store `cancelled` or `completed` with a live execution → terminate; engine `failed` with store `running` → resume, counted against a per-run resume budget whose exhaustion opens a human gate; store `cancelling` → re-send `stop` to unstopped stages and mark `cancelled` when all are terminal. Every action is recorded in lineage.
- **Cancel:** a `cancel_workflow` intent marks the run `cancelling`; from that commit on, `issue_epoch` refuses every job of the workflow (P2-full, §8.1); every running stage receives `pause_and_checkpoint` or `stop` (02); in-doubt submissions are resolved by the broker (03).

## 6. Workflow IR

The validator compiles a spec into an engine-neutral IR after all checks in §4 and the capability checks in 01. There are five node types; constructs outside them are not expressible.

| IR node | Expresses | Engine capability needed |
|---|---|---|
| `stage` | `submit` and `await` (§5 rule 2) | None |
| `gate` | Typed test, outcomes, `timeout`, `escalate_after`, `owner`, `escalation`, `quorum`; pin, evaluate, decide (§4.3) | `durable_timers`; `wake_on_event` for human gates |
| `pause` | Checkpoint and release; `hold_capacity` with a maximum duration | `wake_on_event` |
| `budget_check` | Budget thresholds against the store | None |
| `loop` | Bounded loop with `max_iterations` and `max_attempts`; its body starts with a **dispatcher** named by plug-in reference and version | `dynamic_loops`, or static unrolling (§6.2) |

There is no `turn` node. Alternating training is a `loop` whose dispatcher comes from package F (§10.5).

### 6.1 Loop dispatchers

A dispatcher is a pure library function `dispatch(loop_state) -> BodyChoice | NoOp | Terminal` with a declared version, allow-listed like other plug-ins (06). It receives the loop record as its `loop_state` input; the runner (or engine step) reads that record from the store and passes it in (ARCH3-8). It returns which body to run next, with the body's resolved inputs (for F, the pinned partner versions, §10.1), a no-op, or a terminal status with its reason. It never reads or writes the store; writes happen only in the gate transaction (§4.3) or in loop bookkeeping steps keyed by step key. E ships `fixed_sequence` and `until_gate_pass`. Composite stages such as `compose.alternating` are macros that F expands into a `loop` with F's dispatcher.

### 6.2 Unrolling, extension, and history growth

Because attempts are bounded, an engine without dynamic loops unrolls a loop into `max_attempts` dispatcher slots. If attempts are exhausted while the loop is active, the last slot marks the loop `attempts_exhausted` and opens a human gate with outcomes `stop` or `extend` by N. Only the gate owner may extend; the amended spec is re-validated, including its cost estimate and its α schedule (an extension that raises the turn bound re-registers the family, 05), the change is recorded in lineage with old and new lockfile hashes, and the loop continues as a new execution through resume.

Waits are event-driven, so a stage adds a fixed number of history events regardless of its duration. The validator estimates events per execution and compares them with the backend's `history_limit`; backends with `continue_as_new` start each loop iteration as a new execution carrying only the store reference.

## 7. Backend contract, versioning, availability

```python
class WorkflowBackend(Protocol):
    kind: str
    version: str
    capabilities: WorkflowBackendCapabilities
    # dynamic_loops, durable_timers, wake_on_event, max_wait_s, history_limit,
    # continue_as_new, namespaces, payload_encryption, production_ready
    def compile(self, ir: WorkflowIR) -> EngineDefinition: ...
    def start(self, definition: EngineDefinition, run: WorkflowRunRef, epoch: int) -> EngineHandle: ...
    def status(self, handle: EngineHandle) -> EngineStatus: ...        # informational only (§5 rule 1)
    def wake(self, handle: EngineHandle, record_id: str) -> None: ...  # hint only (§5 rule 3)
    def terminate(self, handle: EngineHandle) -> None: ...
```

- **Capability matching.** The validator rejects a workflow that needs a capability its backend does not declare instead of degrading it.
- **Versioning.** The lockfile pins the backend plug-in and compiler versions. Running executions keep their compiled definition; an upgrade either keeps old workers until executions finish or drains each execution at a step boundary and resumes it from the store under the new version.
- **Availability.** Fail-static: if the engine is down, running stages continue under their runs, gate deadlines are wall-clock times in the store judged by commit timestamp when the engine returns (§4.3), and gates opened during the outage are picked up on recovery. Engine availability belongs to its operator (07).

## 8. Backends

### 8.1 P2 thin runner (EM-5, R8-8, R8-9, R9-2)

P2 keeps only what the one-way doors need: the workflow spec, stage DAG, metric gates, lockfiles, and manifests. The runner implements the `WorkflowBackend` contract with the smallest machinery that does so, in one of two variants (R9-2, DA3-1):

- **P2-min (default).** Jobs are submitted with `--no-requeue`; a failed or stalled stage is recovered only by an operator `retry_stage` (§4.3, §5 rule 3) from its last committed cut; the health service is used for detection and RCA only. No `issue_epoch`, held submission, or janitor is built.
- **P2-full.** Adds automatic requeue safety (`issue_epoch`, own `sbatch --hold`, janitor; bullets marked P2-full below), built only if the LLM pilot user confirms in writing at G0 that it needs automatic recovery in P2.
- **Both variants:** the P2 store has a designated operator (the operator that runs the runners), and there is **one runner per team**, whose Slurm privileges are limited to `sbatch`, `scontrol release`, and `scancel` on that team's account (06).

- **Placement.** A single-replica Python service on a **service node**, never a login node (login-node reapers kill long-lived processes, and per-run polling loads slurmctld; FMEA2-7). It holds the workflow lease for every run of its team; a second instance fails lease acquisition and exits. Timers, wake-ups, and dispatch are rows in the store, polled on a short interval. No external engine.
- **Non-production label.** It declares `production_ready: false`; every workflow run record, CLI listing, and lineage record carries the backend identity, and no SLO from 07 applies. It is supported only for the P2 pilot users, and models released from P2 are private only (no external release) until P3 adds membership-inference and trigger-probe checks (06).
- **Store.** One transactional database behind R's store interface. Because it holds fencing state (epochs, pins, `issue_epoch` records), it meets 03's store rules: single primary, no asynchronous replicas, commit timestamps from the primary. The linearizable store of 03 replaces it in P3 without spec changes.
- **Lineage and ledger.** Lineage is written as OpenLineage-format tables in the workflow database; the consumption ledger (schema in A, 05) is run-partitioned Parquet queried with DuckDB. Marquez and Iceberg arrive in P3. The audit log is append-only with a WORM anchor.
- **Submit.** Through R's broker (the P2 subset in 03): the runner builds the launch script with NeMo-Run but submits it itself under the owning team's Slurm account (per-team accounts with `run_root` isolation, 06). P2-min submits with `--no-requeue` and records `submitted` by compare-and-set; an in-doubt submission is resolved by step-key lookup (03). P2-full submits with `sbatch --hold`, records `submitted` by compare-and-set, and only then releases the job; its jobs stay untagged and requeueable (overview D8). Slurm state is queried through a shared Slurm query cache where the site runs one, not by polling slurmctld; an "unknown" lookup blocks the stage and never resubmits (03).
- **Janitor (P2-full).** On a fixed interval, re-releases held jobs whose record is `submitted` (a crash between compare-and-set and release), cancels held jobs that match no record, and re-releases jobs that a start script held while the runner was unreachable. It releases only holds it placed (`held_by_broker` with the matching Slurm hold reason; R9-8).
- **Requeue admission (`issue_epoch`, P2-full).** A job's start script holds no store credential. As the first line of the batch script, before `srun` and container start (R9-8), it calls the runner's `issue_epoch(run_id, job_id, restart_count)` endpoint (R's `ControlAPI`), which succeeds only if the job is the stage's current attempt, the run record is `submitted` or `running`, the workflow is neither cancelling nor cancelled, and the GPU-hour cap is not reached. The response carries the controller epoch and the cut to restore, resolved at job start rather than at submission. On refusal the script exits with the dedicated "do not requeue" code; the job's Slurm requeue limit bounds any loop. If the runner is unreachable, the script holds its own job and exits, and the janitor releases it later.
- **Await.** Driven by the run contract (02), read by the R adapter code the runner hosts over A's contract transport: the runner verifies and records the completion manifest, watches the heartbeat progress counter, and sends `stop` on a stall beyond the declared timeout, after which `on_fail` applies.
- **Scope.** Exactly the IR of §6, the `fixed_sequence` and `until_gate_pass` dispatchers, and the gate library. It is the **conformance oracle** for later backends: every backend must produce the same lineage and gate outcomes on the reference workflows. The cheapest alternative (NeMo-Run plus gate scripts plus a manifest library) is recorded in the overview alternatives.

### 8.2 P9 production backend (direction)

Chosen by what the operator's site already operates (overview §11 question 6). Exit: backend conformance against the P2 runner and a composition-sized synthetic loop (6 rounds × 3 nodes with turn gates and toy stages, maximal attempts) with zero duplicate launches under injected engine, worker, and store faults (07).

Temporal is the default candidate because the layer depends on long waits, event-driven wake-ups, and data-dependent loops: timers persist on the server, signals reach running workflows, and waiting consumes no compute. Its history is limited to 51,200 events or 50 MB per execution, with warnings at 10,240 events or 10 MB, which §6.2 handles with `continue_as_new`.

| Engine | Fit | Notes |
|---|---|---|
| Temporal | High | Durable timers, signals, code-defined dynamic loops; per-execution history limit |
| Flyte | Medium-high | Typed, ML-oriented; an official Slurm connector (`flytekitplugins-slurm`, over SSH) exists, although stages here are launched by the broker |
| Argo Workflows, Kubeflow Pipelines | Medium | Kubernetes-native; the workflow service must run on Kubernetes; loops are unrolled |
| Airflow | Medium | Widely operated; mostly static graphs, so loops are unrolled; long human waits need deferrable operators or sensors |
| Prefect, Dagster | Medium | Code-defined control flow; Dagster's asset lineage overlaps with ours |
| ZenML | As a backend only | Abstraction over many orchestrators without gate or human-wait semantics, so it cannot replace this layer |
| Ray Workflows | Excluded | Deprecated in Ray 2.44 and removed; Ray 2.47 is the last release containing it |
| Couler | Excluded | Aims to unify Argo, Tekton, and Airflow but supports only Argo |
| NeMo-Run `Experiment` | Launcher, not an engine | Static task DAG; native scheduler dependencies only on the Slurm executor (`_DEPENDENCY_SUPPORTED_EXECUTORS = (SlurmExecutor,)`); detach only for Slurm and SkyPilot executors |

**Verification level.** Temporal history limits, the Flyte Slurm connector, ZenML's orchestrator list, the Ray Workflows deprecation, and Couler's Argo-only support were verified from web sources in revision 6 (§24 of archived revision 6, not published). The NeMo-Run facts were rechecked in the local clone at `5f39d66` (`nemo_run/run/experiment.py`). Fit ratings for Argo Workflows, Kubeflow Pipelines, Airflow, Prefect, and Dagster are general knowledge and must be confirmed in P9.

## 9. Composition: cases and the pilot

**Status for §9–§13:** Direction — detailed review after G0 (R8-20).

### 9.1 Three cases

Some products are systems of models from different families whose outputs must agree: an LLM writes a storyboard, a video model renders it, an audio model voices it; a VLM plans and a controller executes. Each model can be good alone while the system is poor.

| Case | Example | Policies | Machinery |
|---|---|---|---|
| A. One model, mixed outputs | A unified model emits text then denoises an image | One | Mixed-type rows (01); unscheduled until a user exists |
| B. Several models trained in turns | Planner LLM → video → audio | Several | Package F (§10), only after the pilot |
| C. Frozen partner as reward | A VLM judge or a frozen downstream model scores a generator | One trained | A `RewardFn` (01); no new mechanism |

No gradient flows between nodes; differentiable interfaces between models are out of scope.

### 9.2 Scripted pilot before package F (DA-6)

Case B is built only if a scripted version of it beats the cheaper alternatives at equal cost. The pilot uses only existing mechanisms:

1. **Turn as case C.** Freeze all partners at pinned versions. Wrap the downstream partners and the system reward as one `RewardFn` and the upstream partners as a prompt source, then train one node with an ordinary RL stage on the P2 runner.
2. **Manual alternation.** An operator runs the next turn by hand with the other node trained, for at least two rounds; turn gates use the gate library as in §10.3.
3. **Arms at equal GPU-hours,** counting partner inference: (a) each node trained alone on its own local rewards; (b) one node trained against frozen partners (pure case C); (c) scripted alternation.
4. **Success criterion,** registered before the pilot: arm (c) beats both (a) and (b) on the final test set under a `gate.superiority` test (05) by an evaluator independent of every training reward and partner, with declared α, power, and MDE.

If arm (c) does not win, F is not built; teams keep using case C. If (b) wins, the result is recorded as a recommendation for case C. The pilot's result and registered criterion are recorded in this document before P10 continues.

## 10. Package F: composition from ordinary plug-ins (ARCH-5)

### 10.1 Structure and pinned partners

A composed system is declared as a directed acyclic **rollout graph** of nodes, each naming a model family, rollout engine, and initial checkpoint, and the typed values it consumes and produces. The trained node of a turn runs an ordinary RL stage; F supplies three plug-ins and nothing else:

| F component | General contract it implements | Role |
|---|---|---|
| `ComposedEnvironment` | `Environment` (01) | Calls upstream partners at their pinned versions to produce the trained node's inputs; hosts closed-loop partners (§11) |
| `ComposedReward` | `RewardFn` (01) | Calls downstream partners at their pinned versions and the system and local rewards; implements the credit assigner |
| `compose.alternating` dispatcher | `loop` dispatcher (§6.1) | Chooses the next node, resolves partner versions, skips frozen nodes, returns terminal statuses |

**Pinned partners (ARCH2-8).** Partner versions are stage inputs, not run-time lookups. When the dispatcher selects a turn, it resolves each partner's accepted version from the loop record to a manifest hash; the turn stage is submitted with those hashes as typed inputs, recorded in its lockfile and lineage like any other input. `ComposedEnvironment` and `ComposedReward` receive the hashes as configuration and never read accepted-version pointers. A retried attempt of the same turn reuses the turn's pins.

**Generic parent reference (ARCH2-7).** No general contract or record mentions turns, system samples, or partners. Rows carry the generic `parent_ref` field of A's `ConsumptionRecord`, an opaque reference to the record that caused the row; F sets it to a system-sample ID. The **system-sample index** (for each system sample: every node's partition, sample ID, and version, and therefore the composite version) lives in F and is written to lineage (05). Turn gates are ordinary `gate` nodes with `on_fail: reject`; publication is the gate transaction of §4.3.

### 10.2 Credit across nodes

- `shared`: every node receives the system reward.
- `shared_plus_local`: a declared weighted sum of system and local rewards per node.
- `conditional_groups` (default for GRPO-style algorithms): for the trained node, G variants share upstream inputs; each variant is scored by M complete downstream completions. Downstream cost per upstream input per step is G × M × the sum of downstream per-call costs.

**Common random numbers and group normalization (RES2-7).** With `common_random_numbers` the G variants share one downstream seed vector. With group-mean-centred advantages CRN adds no bias beyond the (G−1)/G scaling that mean centring has anyway (01 §7.5; the advantage is linear in the rewards, and each reward's expectation over the shared seed equals its marginal expectation) and reduces variance. Group-std normalization (GRPO's default) is biased with or without CRN (01 §7.5), and CRN adds a further effect: the shared seeds make the G rewards conditionally dependent, so the group standard deviation estimates a conditional variance and the 1/σ scaling correlates with the seed draw, which reweights the objective toward seeds under which variants are hard to tell apart. The default `common_random_numbers: auto` therefore enables CRN for mean-centred advantages and disables it for std-normalized ones, unless the credit-signal pilot (§10.4) also runs a registered bias check (advantages under shared and independent seeds on the same inputs; the mean absolute difference of normalized advantages and the cosine of the resulting policy-gradient estimates within declared bounds) and records it in lineage.

Convergence is not guaranteed with either assigner; the loop is bounded by §10.6.

### 10.3 Per-turn rules and turn gates

- **KL reference:** the node's initial checkpoint for the whole workflow.
- **Optimizer state:** reset per turn by default.
- **Reward epoch:** a change of pinned partner versions starts a new reward epoch for the trained node and resets reward-normalization statistics (05).
- **Buffers** are flushed at the turn boundary.

Turn gates are evaluated on the **selection set** with deployment samplers; the disjoint **final test set** is used only by the final gate.

**α schedule (EVAL2-2).** Turn gates form one family. The number of gated turns is bounded by `max_iterations` × nodes (failed and low-signal turns never reach a gate twice or at all), so the default is Bonferroni over that known bound: α_turn = α_FW / (`max_iterations` × nodes). A registered online FDR procedure from 05 is the alternative. Revision 7's 1/t² spending is withdrawn because it starved late turns. Within a turn the blocking clauses are AND-combined (intersection-union, each at α_turn), and each clause's sample size is planned for power^(1/m), so the joint power of the turn reaches the declared power (EVAL2-4). **Counting rule (EVAL3-1):** m counts every blocking evaluator in the AND clause, so a clause that blocks on two evaluators contributes two (05 §3.5). The validator computes the required human labels from the actual gate configuration (every T4 blocking evaluator, at its derived α_turn and power^(1/m), times the turn bound) and rejects a spec whose `labeling_usd` does not cover them (05 §8). Relaxing the turn-gate defaults is deferred to P10 (`open-issues.md`, RES3-2 / EVAL3-1).

Blocking clauses (four; m = 4 when each blocks on one evaluator):

1. **System gain:** `gate.superiority` of the new version against the node's accepted version, by an evaluator independent of every training reward and partner. Where the system metric has no non-LLM evaluator (story coherence in §13), it is a human evaluation (T4); VLM evaluators are reported as non-blocking (EVAL2-6).
2. **Standalone regression:** `gate.noninferiority` of the node alone on its own benchmark against its accepted version, margin registered before the loop.
3. **Cross-play, per turn** and 4. **cross-play, cumulative (RES2-5, EVAL-4):** the new version is evaluated with a **partner set** per partner: the anchor (initial version) plus the last k accepted versions (default k = 2), so cost per turn is bounded by k + 1 regardless of the round. Each clause is one **averaged non-inferiority clause** over the partner set: the mean over partner versions of Δ against the node's previous accepted version, with a per-turn margin, and against the node's anchor, with a cumulative margin. The **full cross-play matrix** over every accepted partner version is optional (`cross_play_matrix: descriptive`), stored with the gate result, and carries no α.

### 10.4 Credit-signal pilot before each turn (EVAL-4)

Before training, each turn runs a short pilot with the turn's G, M, and partner samplers on a declared number of upstream inputs (included in the cost estimate). It decomposes the system reward per upstream input into a between-variant component σ²_b (attributable to the trained node) and a within-variant component σ²_w (downstream partner noise) by a random-effects estimate with a confidence interval (05), and computes the trained node's **variance share** s = σ²_b / (σ²_b + σ²_w / M).

- If the lower confidence bound of s is at least `min_variance_share`, the turn trains.
- Otherwise the dispatcher may raise M to the smallest value that reaches the threshold, and G to keep the required number of informative groups, if the re-estimated cost fits the budget; the change is recorded in lineage.
- Otherwise the turn ends as `skipped_low_signal` (RES2-6): it consumes an attempt and trains nothing. It increments the node's **low-signal counter**, not its rejection counter, and spends no α. After `max_consecutive_low_signal` skips the node is frozen with reason `low_signal`, distinct from `rejected`, and the report recommends raising M, changing the credit assigner, or adding a local reward.

### 10.5 Automation, failed and rejected turns

The dispatcher receives the loop record as `loop_state` (§6.1: round, turn index, accepted versions, per-node rejection, low-signal, and failure counters, gate history) and selects the next node in round-robin order, skipping frozen nodes.

- **Rejected turn:** the gates returned `reject` or `underpowered`; nothing is published, so no rollback is needed; the node's rejection counter increments; after `max_consecutive_rejections` the node is frozen with reason `rejected`.
- **Low-signal turn:** §10.4.
- **Failed turn:** the run failed and recovery from its last cut (03) did not succeed. It is classified by the failure taxonomy (07), does not count toward rejections, low-signal skips, or stop rules, and is retried as a new attempt with the same pins once the previous one is terminal (§4.3); after `max_consecutive_failed_turns` the loop pauses and opens a human gate.

### 10.6 Stop rules

The loop stops at `max_iterations` rounds; after a full round in which every gated turn was rejected (failed and low-signal turns excluded); after a full round with no gated turn because every scheduled turn was skipped for low signal (terminal status `no_signal`); when every node is frozen; when the budget is exhausted; or when `max_attempts` is reached, which opens the attempts-exhausted gate (§6.2).

### 10.7 Partner capacity, samplers, and placement (PERF-9)

- **Partner sampler.** Training-time partners default to `partner_sampler: fast`, the distilled or few-step sampler declared in the partner's manifest, when one exists; otherwise `deployment`. Turn gates and the final gate always use deployment samplers, so a gain that does not transfer from the fast partner fails the system-gain gate.
- **No masking.** Partner calls use `deadline_policy: wait` with a hard timeout that raises `external_dependency` (07).
- **Cost estimate.** Each node's manifest declares per-call cost per sampler. The validator estimates a turn as steps × inputs per step × (trained-node rollout + pilot share + G × M × downstream chain + upstream partners) plus training, turn-gate evaluation over the k + 1 partner set, and turn-gate labels, and rejects specs whose estimate over `max_iterations` exceeds the GPU or labeling budget.

Illustrative estimate for training the planner (per-call costs are assumptions to be replaced by measured manifest costs, 08): 32 upstream inputs per step, G = 8, M = 2, so 512 video generations per step; deployment sampler 1,200 GPU-s per clip (08 §3.2: 50 steps with CFG at 480×832 and 150 TFLOPS effective; 400 GPU-s at 450 TFLOPS); distilled sampler 100 GPU-s (about 12× cheaper); target step time 600 s; planner trainer gang 8 GPUs.

| Partner sampler | Video GPU-h per step | Pool GPUs for a 600 s step | Pool ÷ trainer gang | GPU-h per 200-step turn |
|---|---|---|---|---|
| Deployment | 171 | 1,024 | 128× | 34,100 |
| Distilled | 14 | 85 | 11× | 2,840 |

Partner pools dwarf trainer gangs. The trainer gang and the composition driver form one gang, and each partner pool is a separate, independently sized service workload placed by the broker (03) on the same cluster in P10; placing partner services on other clusters is a P11 decision. The driver waits for pools to reach their declared minimum size before the pilot starts.

- **Partner output cache.** Keyed by (input hash, partner version, sampler configuration hash, seed); bounded, namespaced per tenant, dropped when the partner's pinned version changes, can be disabled, and reports its hit rate.

## 11. Hierarchical closed-loop partners (EMB-5, EMB2-5)

A planner at about 1 Hz and a controller at 100 Hz or more form a closed loop in time, not a DAG, so it is **not** modelled as rollout-graph nodes. It is embedded inside `ComposedEnvironment`:

- **Training the planner:** the frozen controller and the simulator run inside the environment's step. One planner decision step issues a subgoal; the environment runs k control steps (k about 100 or more, variable when the interval ends early) and returns the observation and reward.
- **Training the controller:** the frozen planner issues a new subgoal every planner interval; the controller trains at its own rate with the fused vectorized execution mode for the simulator (08). A frozen VLM planner cannot run inside fused mode, so it runs as a **co-located batch inference service** (placement M3, 08), queried once per planner interval with a batch over all environments; the stage declares this **hybrid mode** (fused simulator and controller, M3 planner service), and its latency budget, batching, and failure handling are specified in 08.

**Multi-rate credit.** The planner's decision step is a temporally extended action. Its reward is R = Σ_{i=0}^{k−1} γ_c^i r_i over the control steps it spans, and its discount to the next planner step is γ_c^k with the actual k, recorded in the per-step `[B, K]` discount field (01, 08) that GAE reads, so the two rates optimize the same discounted return. An interval cut by episode end or a time limit sets the planner step's `terminated` or `horizon_cut` flag (01). The controller receives per-step environment and system rewards, plus an optional declared local subgoal-progress reward. Both nodes are trained in alternating turns only; simultaneous training is joint training (§12).

## 12. Joint training (direction only)

All trainable nodes updating in one run needs a multi-trainer controller and a separate design covering at least: atomic all-or-nothing allocation of all trainer gangs on one cluster; deadlock-free data flow (bounded queue depth per edge, a pump order without cyclic waits); a version-skew bound on the composite version using the system-sample index; system-sample granularity for drop-and-redispatch and cuts, so no node's rows are orphaned; grouping under simultaneous updates (conditional groups grow as G^N per input; choose per-node rollout trees, shared groups, or critics); and cadence matching between nodes of very different step costs, with idle time reported as goodput loss (07). Switching a composition from alternating to joint training is a training-affecting change and is never automatic.

## 13. Composition spec

Model and evaluator names are placeholders. `compose.alternating` is F's macro (§6.1). Story coherence has no non-LLM evaluator, so its blocking evaluator is a human panel (T4) and the VLM evaluator is non-blocking; audio-visual sync blocks on a programmatic metric from a different tool than the training reward (T3).

```yaml
workflow: storyboard-video-audio
version: 1
budget: {gpu_hours: 60000, labeling_usd: 40000}   # placeholders; must cover the validator's estimate (§10.7)
artifacts:
  planner_sft: {type: model_checkpoint, uri: "s3://.../planner-sft@sha256:<hash>"}
  video_base:  {type: model_checkpoint, uri: "hf://<org>/<video-model>@<revision>"}
  audio_base:  {type: model_checkpoint, uri: "hf://<org>/<audio-model>@<revision>"}
  briefs:      {type: prompt_dataset, uri: "s3://.../briefs@sha256:<hash>"}
  selection:   {type: item_set, uri: "s3://.../selection@sha256:<hash>", disjoint_from: [briefs]}
  final_test:  {type: item_set, uri: "s3://.../final-test@sha256:<hash>", disjoint_from: [briefs, selection]}
  pilot_record: {type: report, uri: "s3://.../composition-pilot@sha256:<hash>"}   # §9.2 result; required by F
policies:
  planner: {model_family: llm.nemotron@1.0, rollout_engine: vllm@0.25, init: planner_sft}
  video:   {model_family: diffusion.wan@0.2, rollout_engine: diffusion.sde_sampler@0.3, init: video_base,
            partner_sampler: fast}       # distilled sampler from the manifest; gates use deployment
  audio:   {model_family: audio.flow@0.1, rollout_engine: diffusion.sde_sampler@0.3, init: audio_base,
            partner_sampler: fast}
rollout_graph:
  - {node: planner, inputs: [prompts],          outputs: [storyboard]}
  - {node: video,   inputs: [storyboard],       outputs: [clip]}
  - {node: audio,   inputs: [storyboard, clip], outputs: [soundtrack]}
rewards:
  system: [{fn: av_sync@1.0, weight: 0.4}, {fn: vlm_judge.story@1.2, weight: 0.6}]
  local:  {video: [{fn: videoalign@1.0, weight: 1.0}]}
credit: {assigner: conditional_groups, group_size: 8, downstream_samples: 2,
         common_random_numbers: auto}   # off for GRPO std-normalized advantages unless the bias check passes (§10.2)
stages:
  - id: co_train
    uses: compose.alternating@0.1
    algorithm: grpo
    inputs: {prompts: briefs, selection: selection, pilot: pilot_record}
    partner_versions: pinned_at_submission   # resolved by the dispatcher, recorded as stage inputs (§10.1)
    schedule:
      order: round_robin
      nodes: [video, audio, planner]
      steps_per_turn: 200
      max_iterations: 6                  # rounds
      max_attempts: 30                   # every turn, including failed and skipped ones
      max_consecutive_rejections: 2      # per node
      max_consecutive_low_signal: 2      # per node; separate counter (§10.4)
      max_consecutive_failed_turns: 2    # then pause and open a human gate
      kl_reference: initial
      optimizer_state: reset
    credit_signal_pilot: {inputs: 64, min_variance_share: 0.2, confidence: 0.90, adapt: [M, G], crn_bias_check: true}
    turn_gates:
      alpha_family: {name: turn_gates, alpha_fw: 0.05, schedule: bonferroni}   # 0.05 / (6 x 3) per turn (05)
      joint_power: 0.80                  # each of m blocking evaluators planned at 0.80^(1/m)
      system_gain:
        uses: gate.superiority@1.0
        blocking: [{evaluator: human_story_panel@1.0, tier: T4}, {evaluator: av_sync_eval@1.0, tier: T3}]
        non_blocking: [vlm_story_eval@1.0]
        mde: 0.02
      standalone: {uses: gate.noninferiority@1.0, evaluator: node_benchmark@1.0, applies_to: trained_node, margin: 0.01}
      cross_play:
        uses: gate.noninferiority@1.0
        evaluator: human_story_panel@1.0
        non_blocking: [vlm_story_eval@1.0]
        partners: {set: anchor_plus_last_k, k: 2, combine: average}
        margin_per_turn: 0.02
        margin_vs_anchor: 0.03
        full_matrix: descriptive         # optional; no alpha
      sampler: deployment
      on_fail: reject
      timeout: 24h                       # human labels
      on_timeout: reject
      owner: media-quality-team
      escalation: [media-quality-lead]
    outputs: {models: {planner: planner_rl, video: video_rl, audio: audio_rl}}
  - id: final_gate
    uses: gate.human@1.0
    inputs: {models: [planner_rl, video_rl, audio_rl], baseline: [planner_sft, video_base, audio_base], items: final_test}
    test: {type: threshold, metric: pref_winrate_vs_baseline, bar: 0.55, mde: 0.05, power: 0.80}
    alpha_family: {name: final, alpha_fw: 0.025}  # one-sided, family-wise; about 770 comparisons, as in §3
    labels: {sample_size: from_power, deadline: 72h, escalate_after: 48h, deadline_policy: wait}
    on_timeout: pause
    owner: media-quality-team
    escalation: [media-quality-lead]
```

The system-gain clause blocks on both evaluators (AND, each counted in m); with them m = 5 for this spec. A drone mission system uses the same spec with a mission-planner LLM node whose environment embeds the frozen flight controller and simulator (§11), and a mission-completion system reward.

## 14. Changes

### 14.0 Revision 9 (closing)

P2-min runner by default, P2-full on written confirmation, one runner per team (§5 rule 2, §5.1, §8.1); gate decisions locked against the deadline and retention via `retiring` (§4.3); dispatcher receives `loop_state` (§6.1, §10.5); m counting rule and validator labeling check (§10.3, §13). See the round-3 disposition.

### 14.1 Revision 8 (round-2 decisions)

| Revision 7 | Revision 8 |
|---|---|
| E owned the broker, `ControlAPI`, and run-record writes; F depended on C, E, G | Broker, `ControlAPI`, run adapter, and run-record schema are in package R (03); the contract transport is in package A (02); E depends on A, R, E-stats, NeMo-Run; F on C-ext, E, G (header, §1, §5) |
| P2 runner as an in-process service of unspecified placement, submitting through NeMo-Run | Thin P2 layer; runner as a service on a service node, own `sbatch --hold` under per-team accounts, Slurm queries via a shared Slurm query cache where the site runs one, janitor, `issue_epoch` requeue admission, OpenLineage tables and Parquet/DuckDB ledger, private-only releases (§8.1) |
| Gate evaluated a cut that retention could delete; timeout versus decision unspecified | Pin before evaluation, retention never deletes pinned cuts, publish the pinned hash; store commit timestamp decides timeout versus decision (§4.3) |
| Partner versions read by `ComposedEnvironment` at run time; system-sample tag in rows | Partner versions pinned as stage inputs at submission; generic `parent_ref` (§10.1) |
| Cross-play against every accepted partner version, one clause per partner | Anchor plus last k versions; one averaged non-inferiority clause per comparison; full matrix optional and descriptive (§10.3) |
| Turn-gate α by 1/t² spending; independent per-clause α | Bonferroni over the known turn bound or online FDR; per-clause power planned as power^(1/m); gates in a family may not set their own α, and the final gate uses one-sided family-wise α = 0.025 (§3, §10.3) |
| `skipped_low_signal` counted as a rejection | Separate low-signal counter, freeze reason, and stop rule (§10.4–§10.6) |
| CRN always on with conditional groups | `auto`: off for group-std-normalized advantages unless a registered bias check passes (§10.2) |
| VLM story evaluator blocking in the composition example | Human evaluation (T4) blocks story metrics; VLM evaluators non-blocking; final-gate labeling cost stated (§3, §10.3, §13) |
| Frozen VLM planner placement unaddressed | Co-located batch service (M3) in a declared hybrid mode (§11) |
| Single status for the whole file | Composition and post-P2 engine backends marked "Direction — detailed review after G0" (header, §8.2, §9) |

### 14.2 Revision 7 (from revision 6)

P2 minimal runner instead of a phase-1a reference backend (§8); no `turn` IR node, generic `loop` with library dispatchers and a general `reject` outcome (§4.2, §6); typed tests with α, margin or bar, power, MDE, and an `underpowered` result (§3, §4.2); gate, publish, counters, and loop advance in one store transaction (§4.3); broker-to-scheduler protocol and supervisor moved to 03 (§5); completion read from run-contract files (§5); no silent masking of human labels (§4.4); composition from `Environment` and `RewardFn` plug-ins in F (§10); scripted pilot before F (§9.2); cross-play and credit-signal pilot (§10.3, §10.4); distilled partner samplers and separate partner pools (§10.7); hierarchical controllers inside the environment (§11).

## 15. Round-1 disposition

| ID | Section | Resolution |
|---|---|---|
| EM-5 | §8.1, §7 | P2 ships only a single-replica, non-production runner that implements the backend contract; the production engine is chosen in P9. Updated in revision 8: the runner is a service on a service node with its own held submission (§8.1). |
| ARCH-5 | §6, §10.1 | The IR has no `turn` node and composition is built from `Environment`, `RewardFn`, and loop-dispatcher plug-ins in F, with the system-sample index in F. Updated in revision 8: rows carry a generic `parent_ref`. |
| DS-7 | §4.3 | Gate result, publication of the evaluated manifest hash, counters, and loop advance commit in one compare-and-set transaction; results are reused by step key; retries require a terminal previous attempt. Updated in revision 8: a pin precedes evaluation. |
| DA-6 | §9.2 | Package F is built only after a scripted pilot shows a gain over independent and case-C training at equal GPU-hours on an independent evaluator. |
| EVAL-4 | §10.3, §10.4 | Each turn starts with a variance-share credit-signal pilot that adapts G and M or skips the turn. Superseded in part by revision 8: cross-play uses the anchor plus the last k accepted versions in averaged clauses, not every accepted version. |
| RES-7 | §4.4 | Unlabeled samples are never masked at a deadline; connectors wait with escalation or apply propensity correction defined in 05; human gates decide on partial labels only when power and missingness checks pass. |
| EMB-5 | §11 | Hierarchical planner–controller loops are embedded inside the `Environment` with semi-Markov multi-rate credit. Updated in revision 8: discount recorded in the `[B, K]` field; frozen VLM planners as an M3 service in hybrid mode. |
| PERF-9 | §10.7 | Partners default to distilled samplers when available, cost estimates show pools dwarfing trainer gangs, and partner pools are placed as separate units from the trainer gang. |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| DS2-D | §4.3 | The gate pins the evaluated cut in the store before evaluation; retention never deletes pinned cuts; publish references the pinned manifest hash (R8-18). |
| DS2-7 (04 part) | §4.3, §7, §8.1 | Timeout versus decision is decided by the store commit timestamp; the P2 store meets 03's single-primary rules. |
| ARCH2-8 | §10.1, §13 | Partner versions are resolved by the dispatcher and pinned as stage inputs at submission; F's plug-ins never resolve them at run time (R8-10). |
| ARCH2-7 (04 part) | §10.1 | General records carry a generic `parent_ref`; system-sample identifiers stay in F's index. |
| RES2-5 | §10.3, §13 | Cross-play defaults to the anchor plus the last k accepted versions; blocking uses averaged non-inferiority clauses with per-clause power power^(1/m); the full matrix is optional and descriptive (with EVAL2-4, R8-14, R8-16). |
| RES2-6 | §10.4–§10.6 | Low-signal turns have their own counter, freeze reason, and stop rule and do not count as rejections or spend α. |
| RES2-7 | §10.2, §13 | CRN with GRPO group normalization is analyzed; `auto` disables it for std-normalized advantages unless a registered bias check passes. |
| FMEA2-7 | §8.1 | The P2 runner is a service on a service node, queries Slurm through a shared Slurm query cache where the site runs one, and has a janitor (R8-8, R8-9). |
| EVAL2-6 (04 parts) | §3, §4.2, §10.3, §13 | Story metrics without a non-LLM evaluator block on human evaluation (T4) with VLM evaluators non-blocking; gates inside a family may not set `alpha` (05 §3.2), and the default α matches 05; the final gate uses one-sided family-wise α = 0.025 and its labeling cost (about 770 comparisons) is stated. |
| EVAL2-2 (04 part) | §10.3 | Turn gates use Bonferroni over the known turn bound or online FDR; 1/t² spending is withdrawn. |
| EMB2-5 (04 part) | §11 | Frozen VLM planners run as co-located batch services (M3) in a declared hybrid mode; semi-Markov discounts use the `[B, K]` field (details in 08). |
| DS2-6, RT2-1, RT2-3, OPS2-6, FMEA2-6 (04 parts) | §4.3, §5.1, §8.1 | Own held submission with janitor, per-team accounts, `issue_epoch` without store credentials in jobs, cut resolved at job start, `retry_stage` only with the previous job gone (03 owns the protocol). |
| R8-20 | Header, §8.2, §9 | Composition and engine backends beyond P2 carry "Direction — detailed review after G0". |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| DS3-4 | §4.3 | Timeout and decision transactions lock the gate row; a decision commits only with `commit_ts < deadline` and no timeout row (R9-8). |
| DS3-5 | §4.3 | Retention moves a cut to `retiring` by compare-and-set before deletion and skips pinned cuts; pinning a `retiring` cut fails (R9-8). |
| ARCH3-8 (04 part) | §6.1, §10.5 | The dispatcher receives `loop_state` as input and never touches the store; the runner reads the loop record (R9-9). |
| DA3-1 (04 part) | §5 rule 2, §5.1, §8.1 | P2-min (`--no-requeue`, operator `retry_stage`, the health service for detection and RCA only) is the default; P2-full adds `issue_epoch`, `sbatch --hold`, and the janitor only on the pilot user's written confirmation at G0; one runner per team (R9-2, R9-7). |
| EVAL3-1 (counting and validator part) | §10.3, §13 | m counts every blocking evaluator in an AND clause; the validator computes required labels from the actual gate configuration and checks `labeling_usd` (05 §3.5, §8; R9-5). |
| RES3-2; EVAL3-1 (default relaxation) | — | Deferred to P10; recorded in `open-issues.md`. |

## Notes for the overview

1. **Package rows** (resolved): overview §4 shows E depending on A, R, E-stats, NeMo-Run and F on C-ext, E, G.
2. **Labeling budget** (open): the example final gate needs about 770 human comparisons (§3); P2 and P10 cost planning should carry labeling cost next to GPU-hours.
