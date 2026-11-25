# 06. Security, Governance, and Safety

**Status:** The design set is at revision 26; this document was last changed in revision 25. A prototype (`gatedtrain` 0.6.0) implements revision 26 except the training-seed unit of revision 20 (05 §3.3). Frozen until G0 closes; deferred items in `open-issues.md`. MVP controls (phases P0–P2: the §4 items with those phases, §5, §7, §8, §15, and the P0–P2 rows of §16) are normative. Controls that first apply in P3 or later remain **Direction — detailed review after G0** (R8-20) and are marked "(Direction)".

**Revision:** 9 (2026-11-05). Sub-design of `design.md` (the overview). Resolves round-1 (GOV-1–8, RT-1–8, EMB-9 safety parts), round-2 (RT2-*, GOV2-*, EMB2-2, EMB2-4 watchdog and interlocks), and the MVP security findings of round 3 (round-3 review record, not published: R9-7, with R9-2 and R9-9). Constraint-cost, `intervened`, and `proposed_action` step fields are defined in 01.

**P2-min and P2-full (R9-2).** P2-min is the default: jobs are submitted `--no-requeue` and recovered by operator `retry_stage`. P2-full adds automatic requeue safety (`issue_epoch`, the runner's own `sbatch --hold`, janitor) only if the LLM pilot user confirms in writing that it needs automatic recovery in P2. Controls marked "P2-full" apply only then.

**Source:** revision 6 §9.5.1 rules 3–4, §10.2, §11, §12.2, §12.5, §13, §14.2 (archived revision 6, not published); revision 7 of this document.

## 1. Scope and principles

This document defines the threat model, the minimum security baseline, and the governance controls of the framework. It follows overview decision D9: **a minimum baseline lands with the first phase that runs on shared clusters**, and later phases add depth, not the basics.

Package names follow R9-9: package R splits into **`gatedtrain-runtime-agent`** (the in-allocation run adapter under its own OS user: lease, fencing, Ed25519 signing of verbs and pointers, MAC and signature checks at commit, commit gating; it reaches outside the allocation only through a narrow authenticated RPC) and **`gatedtrain-runtime-svc`** (store, broker, `ControlAPI`, `issue_epoch`). "Adapter" below means `gatedtrain-runtime-agent`. Store credentials never enter jobs. The contract transport (interface and both implementations) is in package A with the contract emitter, and the adapter uses it. **C-ext `nemo-rl-gatedtrain`** holds the framework code wired into NeMo RL (emitters, adapters, wrappers). NeMo RL itself never reads leases and only verifies verb and pointer signatures.

Principles:

1. **Fail closed.** A missing tag, identity, signature, policy, or allow-list entry rejects the action; it never defaults to "allowed".
2. **Platform policy beats manifest self-declaration.** A plug-in manifest states what the plug-in needs; the platform decides what it gets (sandbox, egress, identity, priority).
3. **Trust is placed per process and per OS user, not per Ray cluster.** Ray has no authorization between actors of one cluster, so anything that runs inside a run's Ray cluster under the run's OS user is trusted at the level of that run. Untrusted code runs outside it or under a separate OS user (§6).
4. **Decisions and exports go through one door.** Gate decisions, accepted versions, budget extensions, releases, and exports enter only through the `ControlAPI` and the framework exporter, and every one is audited (§8).
5. **Lineage carries policy.** Licence, consent, PII, residency, safety, and `unpromotable` tags are lineage tags, inherited by every derived artifact and checked at stage inputs and at export (§9, §11, §15; lineage model in 05).
6. **No profile is exempt on shared clusters.** The research profile (overview D15) skips the store and the workflow engine, but on a shared cluster it enforces safe formats, signing, the code-execution block, and the plug-in allow-list (§15). Policy tags are enforced by storage on all governed storage regardless of node tenancy (B20). This replaces 01's revision-7 wording "no signing".
7. **A run cannot authorize itself.** Nothing a run writes is accepted by the store unless it matches a cut the run adapter committed and signed; nothing a run receives is acted on unless the adapter signed it (§5.2).

## 2. Assets, actors, and trust boundaries

### 2.1 Assets

| Asset | Why it matters |
|---|---|
| Model weights and checkpoints (cuts, 02) | Theft, substitution, or backdoor insertion compromises every downstream model |
| Training data, labels, preference pairs | Licence, consent, PII; poisoning changes model behavior |
| Rewards, advantages, provenance fields on the data plane | Tampering silently steers training and corrupts lineage |
| Gate decisions, accepted versions, release approvals | Bypass ships an untested or unsafe model |
| Credentials and signing keys (artifact store, scheduler, adapter keys, launch tokens, external judges) | Lateral movement, forged control, and cost abuse |
| GPU-hours and external API spend | Denial of wallet |
| Audit log and lineage ledger | Non-repudiation and impact queries (05) |
| Physical benches and vehicles (HIL, field tests) | Injury and equipment damage |

### 2.2 Actors

Tenant users (submitters, gate approvers, release approvers, budget approvers, safety officers), platform operators, first-party framework services (workflow layer, `ControlAPI`, broker, run adapter, supervisor), components inside a run (run controller, trainer workers, rollout engines), plug-in authors (first-party, vendored third-party, and partner), external services (judges, labeling vendors), human labelers, co-tenant jobs and other teams' framework jobs on shared clusters, the cluster health service, and an external attacker with a compromised plug-in or dependency.

### 2.3 Trust boundaries

| # | Boundary | Crosses from → to |
|---|---|---|
| TB1 | Tenant ↔ `ControlAPI` | Human or CI principal → framework control |
| TB2 | Workflow layer ↔ workflow engine | Framework → engine state, UI, and CLI (04) |
| TB3 | Broker, runner, cluster health service ↔ scheduler and clusters | Framework and health service → Slurm or Kubernetes (03) |
| TB4 | Run controller ↔ Ray workers | Controller → trainer, rollout, and plug-in actors inside one run |
| TB5 | Trainer ↔ data plane | Actors → TransferQueue storage units and transport |
| TB6 | Rewards, judges, simulators ↔ trainer | Scores and observations → training signal |
| TB7 | Artifact store and lineage | Runs → durable artifacts, manifests, ledger (02, 05) |
| TB8 | Plug-in supply chain | Plug-in source, images, dependencies → runtime |
| TB9 | Human labelers | Labeling vendor and labelers → preference store (05) |
| TB10 | Run ↔ run adapter (contract transport) | Run outputs → adapter and store; verbs and pointers → run (02) |
| TB11 | Framework ↔ physical bench | Policy actions → HIL bench or vehicle (03 §11, §14) |

## 3. STRIDE by boundary

Controls refer to baseline items **B1–B30** (§4) or to sections of this document.

### 3.1 TB1 tenant ↔ `ControlAPI`

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | Unauthenticated or impersonated gate approval | OIDC authentication, no anonymous calls (B13, §8.1) |
| Tampering | Self-declared high intent priority overrides safety or operator intents | Priority derived server-side from the source identity (B14) |
| Repudiation | Approver denies approving a release | Append-only audit log with a WORM anchor from P2 (B16) |
| Information disclosure | A tenant reads another team's lineage or metrics | Team-scoped authorization on reads; reads of restricted data audited (§8.2) |
| Denial of service | Intent floods starve other intents | Per-source intent allow-lists (B14) and minimal per-principal rate limits (B15) in P2 |
| Elevation of privilege | One person requests and approves a release; an operator decides a gate or resumes a safety pause | Separation of duties; the operator allow-list excludes `decide_gate`, release, and safety resume (B27, §8.3) |

### 3.2 TB2 workflow layer ↔ engine

Unchanged from revision 7. Spoofed approvals through the engine UI or CLI and edited engine history have no effect: decisions enter only through the `ControlAPI` and store, engine signals are wake-up hints, and engine state is never read as a record (04). Only the workflow layer's identity acts on the engine, and every act is audited (B16). The engine holds store references only, with payload codec encryption and mandatory tenant namespaces; tenants hold no engine credentials (§8.4). An engine outage is fail-static (04 §7).

### 3.3 TB3 broker, runner, cluster health service ↔ scheduler and clusters

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | A co-tenant names a Slurm job with a victim's step key so `find_by_key` adopts it | Adoption bound to the team's submission identity and a per-submission nonce (B12) |
| Spoofing | A non-framework job claims the framework-managed marker, or a forged break-glass ID drives the health service | Marker is an administrator-set account or QOS, verified against the store nonce; break-glass IDs are signed, single-use, expiring, and job-bound (B29, §8.5) |
| Tampering | A job of one team cancels, requeues, or reads another team's jobs or `run_root` | Per-team Slurm accounts and UIDs; `run_root` owned by the team UID with mode 0700; per-run directory ACLs (B21, §7.1) |
| Tampering | A delayed Slurm requeue of a finished or cancelled job relaunches training (C') | P2-min: jobs are `--no-requeue`. P2-full: `issue_epoch` refuses stale, cancelled, and over-cap requeues; the job exits "do not requeue" (B23, §5.3) |
| Repudiation | Jobs launched by a service account cannot be attributed | Each submission records principal, team, workflow, and step key in the audit log (B16) |
| Information disclosure | `sacct` output across accounts leaks job metadata | Queries through a shared Slurm query cache where the site runs one scoped to the team's account (03) |
| Denial of service | Resubmit storms or runaway requeues consume the partition | GPU-hour cap with Slurm QOS backstop from P1 (B14); Slurm requeue limit; supervisor rate limits (03, P3) |
| Elevation of privilege | Runner or broker used as a confused deputy into partitions or teams the tenant may not use | One runner per team whose Slurm privileges are limited to `sbatch`, `scontrol release`, and `scancel` on that team's account; entitlement check for partition and QOS (B12, B21) |

### 3.4 TB4 run controller ↔ Ray workers

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | Co-tenant process joins the Ray cluster or calls actors | Exclusive nodes, host firewall to the allocation, Ray TLS (B5–B7) |
| Tampering | A plug-in actor calls controller methods (masking samples, writing completion) | Controller handle never passed to plug-in actors (B2); non-first-party T1 plug-ins run under a separate OS user (B30, §6) |
| Repudiation | Unattributed in-run actions | Controller logs actions with actor identity to the run's audit stream (B16) |
| Information disclosure | Credentials or the provenance MAC key read by a plug-in sharing the controller's OS user (RT2-7) | Per-process credential files; separate OS user for non-first-party plug-ins; MVP admits first-party T1 only (B10, B30) |
| Denial of service | A plug-in exhausts object-store memory | Per-actor resource limits declared in manifests |
| Elevation of privilege | Dashboard or job-submission API used to run arbitrary code | Dashboard, job-submission API, and Ray client server disabled (B5) |

### 3.5 TB5 trainer ↔ data plane

Unchanged from revision 7. Unauthenticated clients and in-flight reads by co-tenants are stopped by B5–B8; one reward overwriting another's field, advantages, or provenance by field-level write ACLs and the provenance MAC (B1, B3, §5.1); disputed writers by per-`(sample, field)` writer identity in the consumption ledger (05); floods by the allocation firewall and per-client quotas; pickled payloads by authenticated transport and safe formats at rest (B8, B9).

### 3.6 TB6 rewards, judges, simulators ↔ trainer

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | A fake reward service answers in place of the pinned one | Services pinned by content digest (05); mutual TLS with workload identities (§7) |
| Tampering | Judge prompt injection through generated media; backdoored reward | Injection canaries and a non-LLM blocking evaluator from P2 (B26, §12.1); re-scoring audits (§12.2, P3, Direction) |
| Repudiation | Reward version disputed after training | Reward epoch and service digest in lineage (05) |
| Information disclosure | Restricted samples sent to an external judge, or uploaded through a download-only egress rule | Storage-side tag enforcement (B20); downloads only through the site-local mirror, direct Hugging Face and object-store egress denied (§9) |
| Denial of service and wallet | Runaway external judge calls | Per-run paid-judge spend cap and minimal rate limits from P2 (B15, §13) |
| Elevation of privilege | Model-generated code executed by an environment; a vendored simulator escapes | Code-executing environments blocked until P8 (B18); vendored-reviewed tier with no egress and read-only credentials (§6) |

### 3.7 TB7 artifact store and lineage

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | A forged checkpoint registered as a cut | Signed cut and artifact manifests (B9); the store accepts outputs only matching adapter-signed committed cuts (B22) |
| Tampering | Checkpoint shard substituted after commit, including through a hard link from the run's prefix (RT3-1) | Digests verified before load; committed cuts in a prefix owned by the adapter's OS user or write-once storage; restore from copy-on-write copies, never hard links (B9, B11) |
| Tampering | A research checkpoint laundered into production (RT2-4) | `unpromotable` tag; validator requires production lineage or an approver-signed import stage (B24, §15) |
| Repudiation | Unknown writer of an artifact | Write credentials per run; writer recorded in manifest (B11) |
| Information disclosure | Model or data exfiltration through ad hoc copies | Exports only through the framework exporter, private (operator-controlled) destinations only until P3 (B19, B25) |
| Denial of service | Artifact store filled by one run | Per-run and per-team quotas (07) |
| Elevation of privilege | `torch.load` of a pickled checkpoint runs code | Safetensors or `weights_only=True` only, in every profile (B9) |

### 3.8 TB8 plug-in supply chain

Typosquatted or substituted plug-ins are stopped by the digest allow-list with no run-time resolution (B17); compromised dependencies by SBOM, build provenance, and images pulled by digest; unreviewed allow-list edits by the Git merge-request rule, which also records author and approvers (B17, §6 rule 3); unknown in-trainer authors by two-person review and signed commits; exfiltration by the egress allow-list (§9); hangs by the conformance kit and health probes. In-trainer code with trainer identity is first-party or vendored-reviewed only (§6).

### 3.9 TB9 human labelers (P7, Direction)

Unchanged from revision 7. Sybil labelers and coordinated poisoning are met by vendor identity verification, gold items, collusion detection, and a per-labeler share cap (§11.3); disputed consent by a versioned consent tag (§11.1); re-identification by a separately stored identity mapping (§11.2); stalls by deadlines and stage policy (04, 05); exposure of restricted samples by requiring a vendor-specific `labeling_allowed` tag (§9).

### 3.10 TB10 run ↔ run adapter

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | A process forges a verb, a restore pointer, or a committed-cut ack (RT2-2) | Verbs and pointers signed by the adapter with a per-attempt Ed25519 key; the run verifies against a key delivered at launch (B22, §5.2) |
| Tampering | A surviving old attempt overwrites a new attempt's heartbeat, completion, or acks (DS2-A) | Contract paths scoped per attempt and controller epoch; the adapter reads only the current attempt's path (02, B22) |
| Tampering | A forged `completion.json` or cut pointer names outputs that were never committed | Run outputs carry a MAC; the store accepts outputs only when they match a signed committed cut (B22) |
| Repudiation | Disputed verb delivery | Verbs, acks, and commits recorded with action ID and epoch in the audit log (B16) |
| Information disclosure | Another team reads the contract directory | `run_root` isolation (B21) |
| Denial of service | Contract storage unreadable, so healthy runs are judged dead | I/O errors are distinguished from frozen counters (02, 07); not a security control |
| Elevation of privilege | A job uses a store credential to write records or relaunch itself | Store credentials never enter jobs; the adapter reaches `gatedtrain-runtime-svc` only through its authenticated RPC; P2-full: `issue_epoch` with a job-bound launch token (B23) |

### 3.11 TB11 framework ↔ physical bench

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | A superseded run drives the bench | Bench controller rejects stale epochs (03 §11) |
| Tampering | Shield bypassed between policy and actuator | Shield runs as a portable artifact on the bench side (§14.2) |
| Repudiation | Unattributed bench session | Per-session safety-officer approval recorded (§14.5) |
| Denial of service | Framework or store outage leaves actuators running | Local watchdog safe-stops immediately on heartbeat loss; hardware emergency stop independent of the framework (§14.5) |
| Elevation of privilege | A research or standalone fused run reaches a bench | `safety_critical` admission refuses them (B28, §14.4) |

### 3.12 AI-native projects: the user's account and the keeper (05 §18.9, §18.10; revision 25)

On the AI-native path an agent runs with the user's account and a shell, and the keeper, a daemon under its own OS user, holds the sealed items and the signing key (05 §18.9). That separation holds only while the user's account cannot reach a root-equivalent facility: an account that can reach a rootful Docker or containerd daemon, or is in the `docker`, `lxd`, `incus-admin` or `disk` group, can read the keeper's state and key. The keeper therefore checks the user's account for such access (05 §18.10, AN-21), failing closed when it cannot read the host state; when the check is positive, now or at any earlier check on a sealed set, the keeper states a downgraded guarantee that seals nothing from the user's account, keeps it for that set after the access ends, and `verify` exits 1 on it. Root through `sudo` is not detected, and the guarantee says so. The user's jobs and the keeper's sealed evaluation run on two rootless Docker daemons, one per user, so the user's containers map to the user's own uids and the keeper's daemon socket is out of the user's reach. The keeper's inference container runs the candidate as container root, which is the keeper uid on the host, with no network, no capabilities and a read-only root file system: an escape from it through a bug in the inference code or the kernel gives the keeper's account, and with it every sealed set and the key, and the keeper's guarantee states this.

## 4. MVP baseline

Every item is required before the named phase runs on a shared cluster. Items are additive: a later phase inherits all earlier items. **Items marked P1 apply to every P1a, P1b, or P6a run on a shared cluster.** Items marked P3 or later are Direction (R8-20).

| # | Control | Phase |
|---|---|---|
| B1 | Data-plane field-level write ACLs: a reward actor writes only `rewards.<its name>`; only the trainer-side adapter writes advantages; plug-in actors hold no data-plane write capability by default (§5.1) | P1 |
| B2 | The run controller actor is unnamed and its handle is never passed to plug-in actors | P1 |
| B3 | Ledger segments and contract outputs are MACed per segment by the run adapter (`gatedtrain-runtime-agent`) under a per-attempt key held by the agent's OS user, never readable by the team UID; the MAC list is recorded in the signed cut and checked when outputs are accepted; NeMo RL and C-ext carry no MAC keys (§5.1) | P1 |
| B4 | Artifact-store credentials scoped per run: write only under the run's attempt- and epoch-scoped prefix (02), read only declared inputs. Jobs hold no metadata-store credential (B23; R9-9) | P1 |
| B5 | Ray: dashboard, job-submission API, and Ray client server disabled; no ports bound outside the allocation's interfaces | P1 |
| B6 | Exclusive node allocation for any run with a Ray cluster or data plane on shared Slurm; host firewall (Slurm prolog) allowing Ray and data-plane ports only between allocation nodes and the `gatedtrain-runtime-svc` endpoint (or a P2 relay) | P1 |
| B7 | Ray TLS with per-run certificates issued at start; Ray token authentication where the pinned Ray version supports it | P1 |
| B8 | Authenticated, encrypted data-plane transport (CURVE keys per run for the ZMQ TransferQueue backend); an upstream ask if not configurable (R8-3) | P1 |
| B9 | Cut and artifact manifests signed (platform key in a key-management service whose KMS/PKI operator is a G0 site prerequisite, R9-1; a research-scoped key for the research profile); digests verified before load; weights only as safetensors or `torch.load(weights_only=True)`; any other pickle load fails CI. Applies to every profile | P0 |
| B10 | No secrets in plug-in images, Ray runtime env, logged environment variables, or engine history; secret scanning in CI and on image build | P0 |
| B11 | Artifact-store write ACLs: only the run's identity writes its staging prefix; only the exporter writes release locations. From P2, committed cuts live in a prefix owned by the adapter's OS user (or write-once storage), and restore uses copy-on-write copies, never hard links (RT3-1) | P0, P2 |
| B12 | One runner per team, with Slurm privileges limited to `sbatch`, `scontrol release`, and `scancel` on that team's account. The runner and later the broker adopt a job only if it runs under the team's submission identity recorded in the store and carries the per-submission nonce; "unknown" only when the scheduler cannot answer (03 §6.4) | P2 |
| B13 | `ControlAPI` authenticates every call through the identity provider; fixed role set and separation of duties (§8) | P2 |
| B14 | Hard GPU-hour cap: a Slurm QOS limit per team and workflow from P1; from P2 also enforced by the runner at submission, at `issue_epoch` (P2-full), and at run time, with the QOS as backstop. Fenced and paused time counts. Intent priority derived server-side; per-source intent allow-lists | P1 (QOS), P2 |
| B15 | P2: minimal per-principal rate limits on `ControlAPI` calls (including `retry_stage`) and submissions, and a per-run spend cap on paid judges. P3: full per-principal and per-source limits (Direction) | P2, P3 |
| B16 | Append-only audit log, hash-chained, with the chain head streamed to (or anchored at most every 5 minutes in) a WORM store with a retention lock under production operations, covering decisions, reads of restricted data, verbs, commits, and exports (§8.2) | P2 |
| B17 | Digest allow-list of plug-ins per environment; lockfiles resolve exact digests at validation; no resolution at run time; allow-list changes only through a reviewed Git merge request | P1 |
| B18 | Code-executing environments blocked by platform policy until P8, in every profile, regardless of the manifest's `sandbox` field (§10) | P0 |
| B19 | Telemetry artifact capture off by default; models and data leave only through the framework exporter | P0 (capture), P2 (exporter) |
| B20 | Policy tags (licence, consent, PII, residency, safety, `unpromotable`) enforced on the storage side: the launcher issues tag-scoped read credentials, so a run can read only inputs whose tags its stage admits, on all governed storage regardless of node tenancy. Loader checks remain as a second layer. From P2 also at export and before any external reward, judge, or telemetry sink (GOV3-1) | P1 (inputs), P2 |
| B21 | Per-team Slurm accounts and submission UIDs; `run_root` per team owned by the team UID, mode 0700, with per-run directory ACLs; MAC keys and launch tokens issued through the adapter's OS user; entitlement check for partition and QOS; no shared framework account. The team UID is the trust domain (accepted MVP risk, §7.1) | P2 |
| B22 | Authenticated contract transport: per-attempt, per-epoch contract paths; adapter-signed (Ed25519) verbs and pointers verified by the run; MACed run outputs; the store accepts outputs only matching signed committed cuts (§5.2) | P2 |
| B23 | `issue_epoch(run_id, job_id, restart_count)` with a job-bound launch token, run as the first line of the batch script; authorizes with an authoritative `squeue -j`; refuses stale, cancelled, and over-cap requeues (§5.3). Not needed in P2-min (`--no-requeue`) | P2-full |
| B24 | Research profile on shared clusters: safe formats, signing, code-execution block, plug-in allow-list, research-use tag checks on data loaders; outputs tagged `unpromotable`; promotion only through an approver-signed import stage (§15) | P1 |
| B25 | Releases from P2 go to private (operator-controlled) destinations only; models trained on PII or consent-restricted data are not released, even privately, before P3's checks; external release requires the P3 membership-inference and trigger-probe checks (§12.4) | P2 |
| B26 | Any LLM or VLM judge requires injection canaries in conformance and a non-LLM blocking evaluator in each blocking gate, or human evaluation where no non-LLM evaluator exists (§12.1); the one exception is a constraint clause whose judge meets 05 §6.1 (a)–(e), including canaries in every judged batch | P2 |
| B27 | Operator allow-list excludes `decide_gate`, release, and safety resume; safety break-glass needs a safety officer (§8.3, §8.5) | P2 |
| B28 | `safety_critical` admission: only validated workflows; research-profile and standalone fused runs refuse such data; `fenced_compute_max_s = 0`; until P2, every profile refuses such data. The tag is set at dataset registration or inferred by the validator (§14.1, §14.4) | P0 (refuse), P2 |
| B29 | Health-service break-glass action IDs are signed, single-use, expiring, and bound to one job and action; the framework-managed marker is an administrator-set account or QOS (§8.5) | P3 (Direction) |
| B30 | Plug-in actors run under a separate OS user wherever non-first-party T1 plug-ins are admitted; the MVP admits first-party T1 only (§6) | P3 (Direction) |

## 5. Intra-run authorization and the contract channel

### 5.1 Inside the run (P1)

Revision 6 let any actor in a run write any data-plane field and reach store credentials (RT-1). The baseline closes this without adding MAC or signing code to NeMo RL upstream:

1. **Plug-ins return, the adapter writes.** Reward, environment, and judge actors return results to the trainer-side adapter (C-ext or D, first-party), which writes `rewards.<name>` for the calling plug-in only. The write ACL is a table `{field prefix → writer component}` validated at configuration time from plug-in manifests.
2. **Direct data-plane access by plug-ins** (later, for large payloads) requires a per-actor capability token naming readable and writable fields, verified by the storage unit. Until NeMo RL's data plane supports this, plug-ins needing direct access must be first-party (§18 note 1).
3. **Provenance MAC.** The run writes ledger segments, cut files, and the completion manifest through C-ext emitter wiring (hooks 8a-1 and 8d) without any key. The run adapter (`gatedtrain-runtime-agent`), under its own OS user, computes one MAC per segment and per contract output when it accepts them, and records the MACs in the signed cut (02 §2.5, §3; R9-6). A later mismatch fails acceptance with an `unclassified` result and an alert, never a silent drop. Because the key never enters the team UID, a sibling job cannot forge segments after acceptance; content written by a compromised job before acceptance is bounded by the accepted MVP risk (team UID = trust domain, §7.1).
4. **Writes outside the run.** Accepted versions, gate decisions, and budget changes are never written by a run. Only the `ControlAPI` and the workflow layer write those records (04).
5. **Credentials per component.** The trainer gets the run's artifact-store write credential; rollout engines and plug-ins get read-only credentials for declared inputs; nobody inside the run holds a metadata-store credential.

### 5.2 Authenticated contract transport (RT2-2, P2)

The contract transport (package A: the interface and both its POSIX-directory and object-store implementations, the latter an optional dependency; R8-7) carries two directions.

| Direction | Object | Protection |
|---|---|---|
| Adapter → run | Verbs (`pause_and_checkpoint`, `stop`), the restore pointer resolved at job start, committed-cut acks | Signed with a per-attempt Ed25519 key generated by the adapter at attempt start. The run verifies with the public key delivered by the launcher in its launch configuration, never read from the contract directory. Each object names `run_id`, `attempt_id`, `controller_epoch`, `action_id`, and an expiry |
| Run → adapter | Heartbeat, cut pointers, ledger segments, completion manifest | MACed by the adapter on acceptance and recorded in the signed cut (§5.1 item 3); written only under `contract/attempt_<A>/epoch_<E>/` |

Rules:

1. The adapter (`gatedtrain-runtime-agent`) runs under its own OS user, which differs from the run's; its signing key, the committed-cut prefix (B11), and the issuance of MAC keys and launch tokens belong to that user. It holds no store credential and reaches `gatedtrain-runtime-svc` only through the authenticated RPC, so no relay is needed from P3 onward (R9-9). Where a cluster cannot run a second OS user inside the allocation, the adapter runs on a service node (R8-11). Both are G0 site prerequisites.
2. The run acts on a verb only if the signature verifies, the epoch is its own, and the action ID is new (a repeated ID gets its earlier ack); otherwise it acks `rejected_invalid` and reports it (02 §2.3). It restores only from a signed pointer.
3. The adapter commits a cut only after verifying the pointer MAC, the manifest hash, and the manifest signature (B9), then by compare-and-set in the store; it then writes a signed committed-cut ack.
4. **The store accepts outputs only when they match signed committed cuts.** A completion manifest, output artifact manifest, or ledger segment is recorded only if it references a cut committed for the same attempt and epoch. A forged `completion.json` or cut pointer therefore cannot name outputs that were never committed.
5. NeMo RL never reads a lease file or any lease state; lease handling and fencing stay inside the adapter (R8-4).

Residual risk: code running as the run's OS user can forge its own run's outputs within its own attempt. It cannot forge verbs, cannot commit a cut whose files do not match, cannot modify a committed cut, and cannot touch another team's paths. Within a team, the team UID is the trust domain (§7.1).

### 5.3 `issue_epoch` and requeue safety (RT2-3; P2-full)

**P2-min** submits jobs `--no-requeue`; a failed stage is retried only by an operator `retry_stage` through the `ControlAPI` (B13, B15, B27), so no requeue can relaunch training and this section does not apply. **P2-full** keeps Slurm requeue. Start scripts hold **no store credential**. At submission, the runner mints a **launch token** bound to `(run_id, job_id)`, usable only for `issue_epoch` and the contract transport, and issues it through the adapter's OS user rather than as a team-readable file (B21). `issue_epoch(run_id, job_id, restart_count)` runs as the first line of the batch script, before `srun` and container start. The runner authorizes it with an authoritative `squeue -j` for that job (account, user, state), never from a shared Slurm query cache and its cached entries (RT3-4). The call succeeds only if:

- the job is the stage's current attempt;
- the stage record is `submitted` or `running`;
- the workflow is not cancelling or cancelled;
- the GPU-hour cap (B14) is not reached, counting fenced and paused time.

Otherwise the start script exits with the dedicated "do not requeue" exit code, and the job's Slurm requeue limit bounds any loop. The cut to restore is resolved at job start and delivered as a signed pointer. If the runner is unreachable, the start script holds the job and exits; the janitor releases only holds the framework placed, and a self-held job only if `issue_epoch` would currently succeed (03 janitor rule, R9-8). A health-service requeue therefore cannot bypass `ControlAPI` cancel, the cap, or a newer attempt. A stolen launch token can only request an epoch for its own job, which the checks above refuse unless that job is current.

### 5.4 Fenced behavior and control verbs (RT2-5)

Per R8-5, on lease loss the adapter stops committing, uploading, and deleting; the run keeps training and refitting within its attempt for at most `fenced_compute_max_s`, then is paused in place for at most `fenced_hold_max_s` (the adapter sends `pause_and_checkpoint(then=hold)` and, after a same-epoch renewal, releases it with `pause_and_checkpoint(then=continue)`; defaults in 03 §4.3), then stopped. Security rules:

1. **`stop` and safety pauses are always honored**, fenced or not. The adapter delivers them signed with the attempt key; they need no store write to execute.
2. **Fenced time counts against caps.** GPU-hours come from scheduler accounting of the allocation, so fenced compute and paused holds count; the QOS backstop (B14) applies independently of the store.
3. **`safety_critical` runs use `fenced_compute_max_s = 0`**: on lease loss they pause at once. Physical benches additionally safe-stop under the local watchdog (§14.5).

## 6. Plug-in trust tiers and supply chain

| Tier | Examples | Who may author | Isolation | Required before use |
|---|---|---|---|---|
| T0 in-trainer | `StepForward`, algorithm loss, worker part of an algorithm, first-party in-process simulators in fused mode (08) | First-party only | None; runs with trainer identity | Two-person review (the owning team plus a reviewer from the site's platform security); signed commits; SBOM; build provenance; digest allow-list |
| T0-V vendored-reviewed | Pinned third-party simulators run in-process in fused mode (for example a physics engine release) | Third party, vendored into a first-party repository | Runs with trainer identity but holds no egress and only read-only credentials; in fused mode, checkpoints are written by a separate writer process that holds the artifact write credential | Pinned source digest; two-person diff review of every upgrade against the previous pin; SBOM; build provenance; digest allow-list |
| T1 in-run actor | Rewards, environments, rollout engines | First-party in the MVP; reviewed partners from P3 | Own Ray actor; no controller handle; no data-plane write capability (§5.1); separate OS user when not first-party (B30) | SBOM; digest allow-list; conformance kit; declared egress |
| T2 external service | Third-party judges, reward services, simulators, labeling connectors | Any approved vendor | Outside the run's Ray cluster; mutual TLS with a least-privilege workload identity | Service-version pin; egress and data-tag approval (§9); spend limits (§13) |

Rules:

1. Third-party code never runs as T0. It runs in-process only as T0-V, and inside a run's Ray cluster only as allow-listed T1 (principle 3). This resolves the revision-7 contradiction between "T0 first-party only" and fused-mode third-party simulators (GOV2-4).
2. **Separate OS user (B30).** Wherever a non-first-party T1 plug-in is admitted, its actors run under a dedicated OS user, so per-process credential files and the MAC key are unreadable to it. **The MVP admits first-party T1 plug-ins only.** Residual risk in the MVP: a compromised first-party plug-in or dependency shares the controller's OS user and can read the MAC key and trainer credentials; this is bounded by B9, B17, B22, and run-scoped short-lived credentials.
3. **Allow-list changes** (adding, upgrading, or revoking a digest) go through a reviewed Git merge request in the allow-list repository: two approvers, one from platform security, CI verifying SBOM and provenance; the merge commit is recorded in lockfiles and the audit log (GOV2-7). This applies from P1.
4. **Revocation.** A revoked digest is removed by merge request; the lineage tables find derived artifacts, which are tagged `quarantined` (blocked at export and as stage inputs until a `release_approver` clears or deprecates each one). P2: a manual lineage query; P3: automated quarantine (Direction).
5. **Signing.** P0–P2 rely on the digest allow-list plus build provenance. P3 adds signature verification of plug-in images and manifests at admission and the plug-in index (Direction).

## 7. Identities and secrets

| Workload | Identity | Credentials it holds |
|---|---|---|
| Trainer (run controller and workers) | Team UID; per-run workload identity | Run staging-prefix artifact write, tag-scoped input reads (B20), data-plane CURVE key, per-attempt MAC key issued by the adapter (controller process only), adapter public key |
| Job start script (P2-full) | Team UID | Job-bound launch token, issued through the adapter's OS user, for `issue_epoch` and the contract transport only (§5.3) |
| Run adapter (`gatedtrain-runtime-agent`) | Adapter OS user, or a service node (§5.2 rule 1) | Per-attempt Ed25519 signing key; committed-cut prefix write; an RPC credential to `gatedtrain-runtime-svc` scoped to its run (no store credential) |
| Partner / rollout engine | Per-run, per-role identity | Read-only artifact access for its pinned weights |
| Reward or judge service | Per-service identity | Its own model weights; nothing of the caller's |
| P2 runner (one per team) and `ControlAPI` (`gatedtrain-runtime-svc`) | Service identity on a service node | Store write for workflow records; Slurm `sbatch`, `scontrol release`, and `scancel` on its own team's account only; no artifact read |
| Run supervisor, broker (P3, Direction) | Service identities | Store write for run records and placements; scheduler submit scoped by team entitlement |
| Exporter | Service identity (P2) | Read of release-approved artifacts; write to private release locations (external from P3) |

- Credentials are short-lived (hours, renewed by the holder) and scoped to one run or one service. Revoking a run's identity at run end invalidates everything it held.
- Delivery: a file readable only by the consuming process, written at start by the launcher from the secret manager. Never in images, Ray runtime env, command lines, scheduler job scripts, or engine payloads (B10).
- Rotation: platform keys (manifest signing, break-glass signing) rotate at least quarterly with overlap; per-attempt keys die with the attempt. Runbooks in 07.

### 7.1 Per-team accounts and `run_root` isolation (RT2-1, P2)

- Each team has its own Slurm account and submission UID and its own runner (RT3-3), whose Slurm privileges are limited to `sbatch`, `scontrol release`, and `scancel` on that account. It submits the team's jobs (`--no-requeue` in P2-min; `sbatch --hold` in P2-full, R8-8); there is no shared framework account or shared runner, so one team's job or runner cannot cancel, requeue, or read another team's jobs.
- Each team's `run_root` tree is owned by the team UID with mode 0700; each run directory carries a per-run ACL, and contract paths below it are per attempt and epoch (02). Committed cuts are moved to a prefix owned by the adapter's OS user, or to write-once storage, and restores read copy-on-write copies, never hard links (B11). Cross-team reads go only through the exporter or a lineage query with authorization.
- **Accepted MVP risk (RT3-2, GOV3-2).** The team UID is the trust domain: a compromised job acts with the team's privileges. Per-run ACLs, adapter-issued MAC keys and launch tokens, and adapter-owned committed cuts narrow what it can read, forge, or destroy in sibling runs of the same team, but do not remove that reach. Isolation of the fused-mode checkpoint writer is deferred to P6a (`open-issues.md`).
- In P1, runs on shared clusters already use the submitting team's own account and QOS; the cap is the QOS limit (B14).

## 8. `ControlAPI` authentication, authorization, and audit

### 8.1 Authentication and principals

- **Identity provider:** an OIDC provider operated by the platform; recommended is the organization's existing OIDC provider (overview open question 7).
- CI and automation use workload identities from the same provider, never shared user tokens.
- **Team-to-person mapping** comes from identity-provider groups. A workflow belongs to one team; budgets, quotas, accounts, and authorization are evaluated against that team.

### 8.2 Authorization and audit

- P2 ships a fixed role set: `submitter`, `gate_approver`, `release_approver`, `budget_approver`, `safety_officer`, `operator`, `platform_admin`. P3 replaces it with full RBAC (Direction).
- **Intent priority is derived server-side** from the source identity (safety and policy > operator > resiliency > budget > broker reclaim > autoscaler, 03). Each source identity has an allow-list of actions and targets (B14).
- **Evidence is untrusted input.** It is stored and shown to deciders, never executed or used to authorize, and size-limited.
- **Audit log (B16, P2).** Every call, accepted or rejected, every verb and commit, every read of a restricted artifact or lineage record, and every export is appended with principal, action, target, and result. Records are hash-chained; the chain head is streamed to, or anchored at most every 5 minutes in, a write-once (WORM) store with a retention lock, operated separately from the workflow database, under production operations from P2 even though other P2 services are non-production. Retention is at least the life of any model released from the recorded runs (GOV2-6).

### 8.3 Separation of duties

| Action | Requires |
|---|---|
| Release or export a model | Request by a `submitter`; approval by a different person holding `release_approver` |
| Decide a gate | `gate_approver` who is not the workflow submitter |
| Extend a GPU-hour cap or budget | Request by the workflow's team; approval by a different person holding `budget_approver` |
| Declare evaluator independence (05) | Signed by an evaluation owner's key, not the workflow owner's |
| Release of a `safety_critical` model | §14 quorum, in addition to release approval |
| Resume after a safety pause; safety break-glass | `safety_officer` (§14.1, §8.5) |
| Import a research artifact (§15) | `release_approver` who is not the artifact's author |
| Clear a quarantined artifact | `release_approver` who is not the plug-in's author |

The `ControlAPI` rejects an approval whose approver equals the requester or the workflow submitter. **Operator allow-list (GOV2-5, B27):** the `operator` source may cancel workflows, retry stages, pause, stop, and request resubmits; it may not call `decide_gate`, release or export, or resume a safety pause. 03 §8.1's operator row matches.

### 8.4 Workflow engine tenancy

Tenant isolation in the engine is mandatory: per-tenant engine namespaces or one deployment per tenant. Tenants hold no engine credentials. The P2 runner runs one deployment per team. This is a backend conformance requirement in 04.

### 8.5 Health-service break-glass (RT2-6, FMEA2-8; P3, Direction)

- **Signed action IDs.** A framework action ID granted to the health service is an Ed25519-signed token naming the job ID, the action (`kill`, `hold`, `resubmit`), an issue time, an expiry (default 1 h, 03 §10.6), and a random nonce. The health-service guard verifies the signature, the job binding, and the expiry, and records consumed nonces so each ID works once. IDs cannot be guessed or replayed.
- **Marker.** The framework-managed marker is an administrator-set Slurm account or QOS, not a job comment; the guard verifies the job's identity against the store nonce (B29). The broker refuses tagged placement on clusters without the guard (R8-11).
- **Triggers.** Break-glass also triggers on store unavailability. Kill and hold need no ID after `T_bg`; resubmit always needs an ID.
- **Safety.** For `safety_critical` jobs, a break-glass resubmit or resume needs an ID co-signed by a `safety_officer`; kill and hold remain available because they move toward the safe state. Break-glass never bypasses the no-automatic-resume rule (§14.1).

## 9. Egress, residency, and policy tags

- **Mirror, not direct egress (RT3-6).** Models and datasets are downloaded only through a site-local model and data mirror; direct egress to Hugging Face and to external object stores is denied, so no "download" rule can carry an upload.
- **Declared egress, download versus upload.** Every plug-in manifest lists destinations with purpose `download` or `upload`. Default is deny. On Slurm, the allocation firewall (B6) routes egress through an allow-listing proxy from P1; on Kubernetes, network policy plus the same proxy. A `download` destination permits only `GET` and `HEAD` with bounded request size, to pinned paths or digests; request bodies and other methods are refused. An `upload` destination needs a data-tag approval for the tags of the data it may receive (B20) and is logged per request; an external judge that receives candidate outputs is one (05 §6.1 (e)). Undeclared destinations fail closed (RT2-8).
- **Storage-side tag enforcement (GOV3-1).** Tag checks do not rely on user-controlled loader code; launcher-issued, tag-scoped read credentials enforce them (B20).
- **Residency tags** are lineage tags. Placement (03), including scale-out and replacement, checks them and fails closed.
- **Inheritance.** Latent and embedding caches, reward caches, telemetry captures, preference connectors, and exporter outputs inherit the union of their inputs' tags (05).
- **External services.** A sample with a tag that forbids external processing is never sent to a T2 service outside its permitted locations; the adapter checks the tag before the call.
- **Exit test (P2):** a workflow whose inputs carry a `restricted` tag, configured with an external reward and telemetry capture, must show zero restricted samples reaching either; the attempt is rejected at validation and, when injected at run time, blocked at the adapter and audited. A plug-in declaring only `download` to a host must fail to `POST` to it.

## 10. Sandbox

- Whether model output is executed is decided by the **platform**, from the environment's type and code review, not by the manifest's `sandbox` field. The default for any environment that compiles, interprets, or loads model output is `required`.
- Until P8, such environments are blocked on every cluster and in every profile (B18). P8 delivers an approved sandbox per cluster type with no network, no credentials, read-only filesystem except scratch, resource limits, and a passing escape suite (Direction).
- Simulators that only consume actions do not execute model output; they are T0, T0-V, or T1 plug-ins (§6).

## 11. Consent, licensing, and human labelers

### 11.1 Consent and licence as lineage tags

- Consent scope is a lineage tag, inherited like licences. It is checked at stage inputs (from P1 on shared clusters, B20) and at export (P2).
- **Withdrawal despite immutability** (P3, Direction). Artifacts stay immutable. Withdrawal writes a tombstone; the lineage ledger finds derived datasets, reward models, and checkpoints; those are marked `deprecated` and blocked as inputs and at export. Retraining is a governance decision through the `ControlAPI`. Raw bytes are deleted where consent terms require. Until P3, a withdrawal is handled by blocking the tag at inputs and a manual lineage query.

### 11.2 Labeler identity (P7, Direction)

The preference store holds only pseudonymous labeler IDs; the mapping to persons is in a separate, audited system readable only by the labeling-program owner. Labeler metadata is minimized: no demographic fields unless the study design (05) requires them under consent.

### 11.3 Sybil and collusion detection (P7, Direction)

Gold and honeypot items per batch exclude labelers below the gold threshold and quarantine their labels. The collusion signal is pairwise agreement on non-gold items far above the population, or near-identical timing. A per-labeler share cap applies per batch; vendor-side identity verification is a connector requirement.

## 12. Model-integrity threats

### 12.1 Judge prompt injection (P2, B26)

- Judges receive only the final media and a framework-built rubric. Planner context, chain-of-thought, and other model-written text never reach a judge prompt. Required context (the user prompt for a faithfulness score) is delimited, placed in a separate field, and labeled as data.
- **From P2, any LLM or VLM judge** used as a reward or gate evaluator requires (GOV2-3): injection-canary tests in reward conformance (generated media with rendered instructions must not move the score beyond a registered tolerance), and at least one non-LLM evaluator in every blocking gate. Exception: a judge may be the sole blocking evaluator of a constraint clause under 05 §6.1 (a)–(e), with registered canaries mixed into every judged batch; a missed canary makes the clause MISSING.
- **Where no non-LLM evaluator exists** (for example story-level system metrics such as 04's `av_story_eval`), the blocking clause uses human evaluation under 05's labeling procedure, and the LLM judge result is descriptive only (EVAL2-6; coordinated with 04 and 05).
- OCR rewards train text rendering, which is the injection channel into VLM judges; workflows combining both are covered by the rule above.

### 12.2 Backdoors through plug-ins and partners (P3, Direction)

- **Randomized re-scoring audits:** a random fraction of reward outputs is re-scored by a first-party reference implementation; divergence above tolerance raises an intent and blocks the next gate.
- **Trigger and canary probes** in release gates: a fixed, secret probe set of trigger patterns plus held-out canaries; behavior shifts block external release.
- Two-person review of T0 and T0-V code (§6); partners are pinned by digest and pass the same audits.

### 12.3 Exfiltration and membership inference

- Exports only through the framework exporter (B19), which checks policy tags, records lineage, and requires release approval (§8.3). PII flags are inherited by model artifacts; a PII-flagged model cannot leave its permitted locations.
- Before external release (P3, Direction), a membership-inference and memorization test runs on canaries and held-out members of sensitive datasets; failure blocks release.

### 12.4 Release gating by phase (RT2-8, GOV2-6)

P0–P1 produce no releases (research and parity artifacts only). P2 releases go to private registries and private evaluation only (B25), after B9, B19, B20, B26, and release approval; a model trained on PII or consent-restricted data is not released at all, even privately, until P3's membership-inference checks pass (GOV3-6). External release starts in P3 and additionally requires §12.2 trigger probes and re-scoring audits and §12.3 membership inference.

## 13. Denial of wallet

- **GPU-hour cap from P1:** a Slurm QOS limit per team and workflow (`GrpTRESMins`) applies from the first shared-cluster run (B14). From P2 the runner also enforces a per-workflow cap at submission (projected cost), at `issue_epoch` (P2-full), and at run time (the `stop` verb at the cap); the QOS stays as an independent backstop if the runner is down (OPS2-6). Fenced and paused time counts (§5.4). Behavior at 80% and 100% is in 07.
- **External spend caps** per run for paid APIs and paid judges from P2; minimal per-principal rate limits on `ControlAPI` calls, `retry_stage`, and submissions from P2; full per-source limits in P3 (B15, Direction).
- **Measured, not declared, costs:** after the first stage attempt, projections use measured GPU-hours and spend from accounting.
- Cap extension needs two distinct roles (§8.3).

## 14. Safety-critical workloads

### 14.1 The `safety_critical` tag

A workflow or artifact tagged `safety_critical` (driving, flight, physical robots) is enforced by the workflow validator, which rejects the workflow unless:

1. **Independent safety role with quorum:** gates marked `safety` require approval by a quorum (at least two) of `safety_officer` principals who are not the submitter or the workflow's team lead.
2. **Signed safety case before export:** the export stage consumes a `safety_case` artifact signed by the quorum.
3. **No automatic resume after a pause:** a run or workflow paused for a safety reason resumes only by a `safety_officer` decision through the `ControlAPI`; neither supervisor recovery, operators, nor break-glass resumes it (§8.3, §8.5).
4. **Shield present:** every policy export bundles a `safety_shield` artifact (§14.2) and the gates evaluate the policy with that shield active.

The tag is inherited through lineage: a model distilled from a safety-critical policy is safety-critical.

**Who sets it (GOV3-4).** Dataset owners set `safety_critical` at dataset registration. The validator also infers it for any workflow with a HIL, SITL, shield, or vehicle-export stage. Removing the tag from a dataset or workflow requires a `safety_officer` signature, recorded in the audit log.

### 14.2 Safety-shield runtime contract (EMB2-2)

```python
class SafetyShield(Protocol):
    version: str                          # content digest, pinned in lineage
    max_latency_us: int                   # per batch of max_batch, verified by conformance
    max_batch: int
    def reset(self, odd: OperatingDomain, env_ids: Tensor | None = None) -> None: ...
    def filter_batch(self, obs: TensorDict, proposed: Tensor) -> ShieldBatchResult: ...
    # ShieldBatchResult: action [B, A], intervened [B] bool, reason_code [B] int32,
    #                    constraint_costs dict[str, Tensor[B]]
    def filter(self, obs: Obs, proposed: Action) -> ShieldResult: ...   # B = 1 convenience
    def export(self, form: Literal["c", "onnx"]) -> ShieldArtifact: ...
```

- The shield is a T0 plug-in, deterministic, with no network or file access, and bounded latency. `filter_batch` operates on tensors on the simulator's device so fused vectorized mode (08) needs no per-sample Python.
- **Portable artifact.** `export` produces generated C (no dynamic allocation, declared numeric type) or ONNX with a fixed opset. The artifact, not the Python object, runs in SITL, HIL, and deployment; its digest is pinned in lineage and in the `safety_shield` artifact.
- **Cross-runtime equivalence tests:** the Python, host-artifact, and target-artifact forms run the known-unsafe suite, the known-safe suite, and a randomized fuzz set; `intervened` and `reason_code` must match exactly, actions within a declared tolerance. Failure blocks export.
- Training records the policy's `proposed_action`, the executed action, `intervened`, and constraint costs in the step fields defined in 01; the training rule for intervened steps (`train_on_proposed_plus_cost`, `mask`, or `executed_with_is`) is declared per algorithm in 01.
- Conformance: latency bound met on target hardware; unsafe suites always intervene; safe suites intervene at most a registered rate. Gates report intervention rate and constraint violations per ODD slice (05).

### 14.3 Artifact types

| Type | Contents | Notes |
|---|---|---|
| `flight_log` / `drive_log` | Time-stamped observations, proposed and executed actions, shield interventions, vehicle state | Location tags inherited (sensitive); signed manifest (B9) |
| `hil_log` | HIL bench run with bench ID, firmware digest, shield artifact digest, session approval ID | Required input to the HIL gate |
| `safety_case` | ODD, geofence, kill-switch design and test evidence, hazard analysis, references to logs and gate results | Signed by the safety quorum; immutable; consumed by export |
| `safety_shield` | Shield source digest, portable artifact digests, equivalence-test results, parameters, ODD | T0 plug-in artifact |

### 14.4 Admission of `safety_critical` data (GOV2-1, B28)

- `safety_critical` inputs or outputs are admitted **only in a validated workflow** (P2 validator, §14.1). Hence P6a with such data depends on P2, and P6b inherits it.
- **Refused:** research-profile runs and standalone fused runs (outside a validated workflow) refuse any input carrying the tag and cannot be configured to produce it. A fused stage inside a validated workflow is admitted.
- Before P2 exists, every profile refuses the tag (P0 rule).
- `safety_critical` runs use `fenced_compute_max_s = 0` (§5.4).

### 14.5 HIL interlocks (GOV2-1, EMB2-4)

1. **Per-session approval.** Every bench session needs a `hil_session` record approved through the `ControlAPI` by a `safety_officer`, binding bench ID, policy and shield artifact digests, ODD, operator on site, and a time window. The bench controller refuses commands without a valid session.
2. **Hardware emergency stop independent of the framework.** Each bench has a physical, operator-reachable emergency stop wired to cut actuator power without software; it is tested at the start of each session and the result recorded in the `hil_log`.
3. **Local watchdog.** The bench's local controller runs a watchdog on the run's heartbeat; on loss it **safe-stops immediately** (no fenced-compute window, no hold). It does not depend on the store, the adapter, or the network.
4. **Stale epochs.** The bench controller rejects commands with a stale epoch (03 §11).
5. **Shield on the bench.** The shield's portable artifact runs on the bench side between the policy and the actuators.

## 15. Research profile on shared clusters (RT2-4, GOV2-2, B24)

On a dedicated workstation the research profile is as in 01. On any shared cluster it enforces safe formats and signing with the research-scoped key (B9); the code-execution block (B18); the digest allow-list (B17), with local paths only for the submitter's own first-party code, recorded by digest; tag-scoped read credentials that admit only data tagged for research use (B20, on all governed storage) and refuse `safety_critical` data (B28); lineage inputs recorded by the launcher from the credentials it issued, not self-reported by the job (GOV3-5); `unpromotable` tags on all outputs, inherited through lineage; B5–B8 where a Ray cluster spans nodes; and the QOS cap (B14).

**Promotion.** The validator rejects any checkpoint as a production stage input unless it has production lineage, or a release approver signs an **import stage**. The import stage records the research artifact's digest and lineage, verifies safe formats and the research-key signature, re-checks policy tags of all inputs, and emits a production artifact; `unpromotable` is cleared only on that output. Promotion reruns may reuse research artifacts through this path (R8-14).

## 16. Controls by phase

| Phase | Controls added |
|---|---|
| G0 | Site prerequisites cover per-team accounts and QOS caps, exclusive nodes and the prolog firewall, a second OS user (or service node) for the adapter, and the site-local model and data mirror; a designated KMS/PKI operator for signing keys (R9-1) |
| P0 | B9 signed manifests and safe formats; B10; B11; B18; B19 capture off; B28 refusal of `safety_critical` data; two-person review, SBOM, and build provenance for T0 code |
| P1a, P1b, P6a on shared clusters | B1–B8; B14 QOS cap; B17 with Git merge-request changes; B20 tag-scoped input credentials; B24 research profile enforcement; egress via proxy and the site-local mirror, download and upload distinguished |
| P2-min | B11 adapter-owned cuts; B12 per-team runner; B13; B14 runner cap; B15 minimal limits; B16 WORM anchor; B19 exporter; B20 export checks and egress exit test; B21; B22; B25; B26; B27; B28 validator and admission; manual revocation query |
| P2-full | P2-min plus B23 `issue_epoch` and the janitor rules (03) |
| P3 (Direction) | Full RBAC; B15 full rate limits; plug-in image signing and index; automated quarantine; consent tombstones; B29 break-glass; B30 separate OS user for partner T1 plug-ins; re-scoring audits, trigger probes, and membership inference before external release; service identities for supervisor and broker |
| P4 (Direction) | VLM-judge reward service inherits B26 |
| P6a | B28 applies: `safety_critical` data only inside P2 workflows; shield batched API and portable artifact; flight/drive/HIL log and safety-case types; HIL interlocks (§14.5) |
| P6b (Direction) | Same as P6a for VLA flow heads |
| P7 (Direction) | Labeler identity separation, sybil and collusion detection, vendor identity verification |
| P8 (Direction) | Approved sandbox per cluster type; code-executing environments unblocked only through it |
| P9–P12 | No new baseline items |

## 17. Changes

### 17.1 Revision 7 (from revision 6)

Revision 7 replaced per-environment signing with a phased baseline (manifests signed in P0, digest allow-list in P1, plug-in signing in P3); made in-trainer code first-party only with two-person review, SBOM, and provenance; added authorization inside a run (field ACLs, provenance MAC, controller isolation, per-run credentials); replaced "Ray bound to private networks" with exclusive nodes, firewall, TLS, disabled dashboard and job API, and an authenticated data plane; let the platform, not the manifest, decide sandboxing and blocked code execution until P8; derived intent priority server-side; added OIDC, team mapping, fixed roles, and two-person actions; made engine tenant isolation mandatory; replaced the 80% budget gate with hard caps and measured costs; enforced consent and PII as lineage tags; made `safety_critical` validator-enforced with a shield contract and safety case; and added §12's judge-injection, backdoor, and membership-inference controls.

### 17.2 Revision 8 (from revision 7)

| Revision 7 | Revision 8 |
|---|---|
| One Slurm service account runs all P2 jobs | Per-team accounts and UIDs; `run_root` isolation (B21, §7.1) |
| Unauthenticated contract channel; run reads `lease.json` | Adapter-signed verbs and pointers; MACed outputs; store accepts only outputs matching signed committed cuts; NeMo RL reads no lease (B22, §5.2) |
| P2 start scripts hold a store credential | `issue_epoch` with a job-bound launch token; stale, cancelled, and over-cap requeues refused (B23, §5.3) |
| Research profile exempt from allow-list and code block; 01 says "no signing" | Enforcement on shared clusters, `unpromotable`, approver-signed import (B24, §15) |
| Run rejects `stop` while fenced; fenced compute outside caps | `stop` and safety pauses always honored; fenced time counts; `fenced_compute_max_s = 0` for `safety_critical` (§5.4) |
| Break-glass action IDs unsigned; marker in job comment | Signed, single-use, expiring, job-bound IDs; administrator-set marker (B29, §8.5) |
| Plug-in actors share the controller's OS user | Separate OS user for non-first-party T1; MVP first-party only, residual risk stated (B30, §6) |
| Re-scoring, trigger probes, membership inference without a phase; P2 releases unrestricted | P2 releases private only; checks in P3 before external release (B25, §12.4) |
| `safety_critical` bypassable via fused mode and research profile; HIL interlocks unspecified | Validated workflows only; research and standalone fused refuse; HIL session approval, hardware emergency stop, local watchdog (B28, §14.4, §14.5) |
| Judge injection controls in P4 | Canaries and non-LLM blocking evaluator from P2; human evaluation where none exists (B26, §12.1) |
| T0 "first-party only" contradicted by fused third-party simulators | T0-V vendored-reviewed tier (§6) |
| Operator allow-list includes `decide_gate`, release, safety resume | Excluded; safety break-glass needs a safety officer (B27, §8.3) |
| Audit log best-effort in P2; allow-list changes unaudited; tag checks from P2; GPU cap from P2; egress without direction; HMAC stamping upstream in the controller | WORM-anchored audit log (B16); Git merge requests for allow-lists from P1 (B17); input tag checks from P1 (B20); QOS cap from P1 (B14); download and upload distinguished (§9); MAC in C-ext wiring, verified by the adapter (B3) |
| Per-sample Python shield | `filter_batch`, portable C or ONNX artifact, cross-runtime equivalence tests; `proposed_action` recorded (§14.2) |
| B15 rate limits, revocation automation, consent tombstones in P2 | Moved to P3 (Direction) per R8-8 |

### 17.3 Revision 9 (closing)

Applies R9-7 with the R9-2 and R9-9 consequences (B11, B12, B15, B16, B20, B21, B23, B25, B28 and §5.2, §5.3, §7.1, §9, §14.1, §15); see the round-3 disposition.

## 18. Notes for the overview and sibling documents

1. **Signing phase wording (resolved):** manifests signed from P0; plug-in signing and the index in P3. **Possible upstream ask.** Direct data-plane access by non-first-party plug-ins needs per-actor field capabilities in NeMo RL's data plane; until then such plug-ins stay first-party (§5.1).
2. **01 (resolved):** 01 §8.1 signs manifests and enforces safe formats, the allow-list, and tag checks on shared clusters, and refuses `safety_critical` data; `proposed_action` and the training rule are in 01 §5.3.
3. **03 (resolved):** 03 §8.1's operator row excludes `decide_gate`; 03 §10.2 and §10.6 adopt the administrator-set marker and signed break-glass IDs (§8.5).
4. **04 and 05 (resolved):** blocking clauses that rely only on an LLM or VLM judge (for example `av_story_eval`) block on human evaluation from P2 (04 §10.3, 05 §6.1).
5. **Site prerequisites (G0, resolved):** overview §3 lists per-team accounts and QOS caps, exclusive nodes and the prolog firewall, and a second OS user or service node for the adapter signing key among the G0 site prerequisites. Revision 9 adds the site-local mirror and the KMS/PKI operator (R9-1).

## Finding disposition (round 1)

| ID | Section | Resolution |
|---|---|---|
| GOV-1 | §6, §4 B9/B17 | In-trainer code is first-party only with two-person review, SBOM, and provenance from P0, a digest allow-list from P1, and signing in P3. Revision 8 adds the T0-V vendored-reviewed tier for pinned third-party simulators. |
| GOV-2 | §9, §4 B20 | Egress is declared and enforced, distinguishes download from upload (revision 8), caches and captures inherit input tags, and a P2 exit test proves restricted samples cannot reach external rewards or telemetry. |
| GOV-3 | §10, §4 B18 | The platform decides whether output is executed; code-executing environments are blocked until P8 in every profile. |
| GOV-4 | §8 | OIDC identity, team mapping, server-side priority, separation of duties, and a signed evaluation-owner declaration. Revision 8 narrows the operator allow-list. |
| GOV-5 | §14 | Validator-enforced `safety_critical` tag with quorum, signed safety case, mandatory shield, no automatic resume; revision 8 adds admission rules (§14.4) and HIL interlocks (§14.5). |
| GOV-6 | §11 | Consent is a lineage tag checked at inputs (P1) and export (P2); tombstones and deprecation move to P3 (Direction); the labeler mapping is separate. |
| GOV-7 | §8.2, §8.4 | Engine tenant isolation is mandatory; the audit log is append-only with a WORM anchor from P2 (revision 8). |
| GOV-8 | §7 | Per-workload identities with short-lived credentials; revision 8 removes store credentials from jobs (B23). |
| RT-1 | §5.1, §4 B1–B4 | Field-level write ACLs, an unreachable controller, per-run credentials; the provenance MAC moves to C-ext wiring and is verified by the adapter (revision 8). |
| RT-2 | §4 B5–B8 | Exclusive nodes, an allocation firewall, Ray TLS, disabled dashboard and job API, and an authenticated data plane from P1. |
| RT-3 | §3.3, §4 B12, B21 | Adoption is bound to the team's submission identity and the stored nonce; revision 8 replaces the shared P2 account with per-team accounts. |
| RT-4 | §3.7, §4 B9/B11 | Signed manifests with verified digests, safe formats only, and write-once prefixes from P0. |
| RT-5 | §12.1, B26 | Judges see only final media; injection canaries and a non-LLM blocking evaluator are required from P2 (revision 8, previously P4). |
| RT-6 | §12.2, §6 | Re-scoring audits and trigger probes are phased in P3 (Direction) before external release; P2 releases are private only (B25). |
| RT-7 | §13 | Hard GPU-hour caps via Slurm QOS from P1, runner cap from P2, measured costs; rate limits in P3. |
| RT-8 | §12.3, §12.4, B19 | Exporter-only exports and inherited PII flags; membership inference in P3 before external release. |
| EMB-9 (safety shield, logs, safety case) | §14.2, §14.3 | A versioned `SafetyShield` contract, now batched and portable (revision 8), plus log and safety-case artifact types; step fields are in 01. |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| RT2-1 | §7.1, B21 | Per-team Slurm accounts and UIDs; `run_root` owned by the team with mode 0700; no shared framework account. |
| RT2-2 | §5.2, B22, TB10 | Per-attempt contract paths; adapter-signed (Ed25519) verbs and pointers verified by the run; MACed outputs; the store accepts outputs only matching signed committed cuts; NeMo RL reads no lease. |
| RT2-3 | §5.3, B23 | `issue_epoch` with a job-bound launch token replaces store credentials; it refuses non-current, cancelled, and over-cap requeues; "do not requeue" exit and Slurm requeue limit. |
| RT2-4 | §15, B24 | Research profile enforces safe formats, signing, code block, allow-list, and research-use tags on shared clusters; outputs `unpromotable`; approver-signed import stage. |
| RT2-5 | §5.4, B14 | `stop` and safety pauses always honored; fenced and paused time counts against caps; `fenced_compute_max_s = 0` for `safety_critical`. |
| RT2-6 | §8.5, B29 | Signed, single-use, expiring, job-bound break-glass IDs with a consumed-nonce record in the guard (P3, Direction). |
| RT2-7 | §6 rule 2, B30, B3 | Separate OS user wherever non-first-party T1 plug-ins are admitted; MVP admits first-party T1 only; residual risk stated. |
| RT2-8 | §12.4, §9, §13, B25 | P2 releases private only; re-scoring, trigger probes, and membership inference in P3 before external release; egress distinguishes download from upload; QOS cap from P1. |
| GOV2-1 | §14.4, §14.5, B28 | `safety_critical` only in validated workflows (P6a depends on P2 for such data); research and standalone fused runs refuse it; HIL session approval and an independent hardware emergency stop. |
| GOV2-2 | §15, B20, B24 | Research-use tag checks on data loaders and `unpromotable` outputs prevent laundering restricted data or weights. |
| GOV2-3 | §12.1, B26 | LLM and VLM judges require injection canaries and a non-LLM blocking evaluator from P2; human evaluation where none exists (with 04 and 05). |
| GOV2-4 | §6 | T0-V vendored-reviewed tier: pinned source, two-person diff review on upgrades, no egress, read-only credentials, separate checkpoint writer in fused mode. |
| GOV2-5 | §8.3, §8.5, B27 | Operator allow-list excludes `decide_gate`, release, and safety resume; safety break-glass needs a safety officer; break-glass never bypasses no-auto-resume. |
| GOV2-6 | §8.2, §12.4, B16, B25 | WORM-anchored append-only audit log from P2 under production operations; P2 releases private only. |
| GOV2-7 | §6 rule 3, §1 principle 6, B9, B17 | Allow-list changes through reviewed Git merge requests from P1; the research profile signs manifests, resolving the contradiction with 01. |
| GOV2-8 | §4 B20, §16 | Input policy-tag checks from P1 on shared clusters. |
| EMB2-2 | §14.2 | `filter_batch` tensor API, portable C or ONNX artifact, cross-runtime equivalence tests; `proposed_action` and executed action recorded (fields in 01). |
| EMB2-4 (watchdog, interlocks) | §5.4, §14.5 | `fenced_compute_max_s = 0` for `safety_critical`; bench local watchdog safe-stops immediately on heartbeat loss; interlocks specified. Bench lease timing is in 03. |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| RT3-1 | §3.7, §7.1, B11 | Committed cuts in an adapter-owned prefix (or write-once storage); copy-on-write restore, no hard links. |
| RT3-2 | §5.2, §7, §7.1, B21 | Team UID = trust domain as accepted MVP risk; per-run ACLs; MAC keys and launch tokens issued through the adapter's user; no store credential in jobs (R9-9). |
| RT3-3 | §3.3, §7.1, B12 | One runner per team limited to `sbatch`, `scontrol release`, `scancel` on its account. |
| RT3-4 | §5.3, B23 | Authoritative `squeue -j`; first line of the batch script; P2-full only. |
| RT3-5 | §13, B15 | Minimal per-principal rate limits and a per-run paid-judge cap in P2. |
| RT3-6 | §9 | Site-local model and data mirror; direct Hugging Face and object-store egress denied. |
| GOV3-1 | §9, B20 | Launcher-issued tag-scoped read credentials on all governed storage, regardless of node tenancy. |
| GOV3-2 (MVP part) | §7.1, B21 | Per-run ACLs; keys issued through the adapter's user. |
| GOV3-4 | §14.1, B28 | Set by dataset owners; inferred for HIL/SITL/shield/vehicle export; removal needs a safety-officer signature. |
| GOV3-5 | §15 | Research lineage inputs recorded by the launcher. |
| GOV3-6 | §8.2, §12.4, B16, B25 | WORM anchor streamed or ≤ 5 min; PII- or consent-trained models not released before P3; P2 rate limits (RT3-5). |
| RT3-7, GOV3-2 (fused writer) | `open-issues.md` (P6a) | Deferred. |
| GOV3-3 | `open-issues.md` (P3) | Deferred. |
