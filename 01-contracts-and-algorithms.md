# 01. Contracts and Algorithms

**Status:** The design set is at revision 25; this document was last changed in revision 9. A prototype (`gatedtrain` 0.4.0) implements revision 24 except the training-seed unit of revision 20 (05 §3.3). Frozen until G0 closes; deferred items in open-issues.md.

Revision 9 applies round-3 decisions R9-3, R9-4, R9-7 (research-profile items), and R9-9 (rule-2 wording) from the round-3 review record (not published); revision 8 implemented R8-3, R8-10, R8-12–R8-15. Overview decisions D2–D5 and D15 and the hook list in overview §5 are authoritative; where the overview and the review decisions differ, the decisions win.

**Covers:** package placement of contracts and math; NeMo RL baseline and upstream hooks; the decision step; the step-field schema; `ModelAdapter`, `StepForward`, primitives, and capability records; algorithm plug-ins on the existing loss path; the research profile; the plug-in model and conformance kit.

**Code baseline:** NeMo RL `4f39fd8`; flow_grpo, dancegrpo, diffusionnft, verl `31cfccb`, and verl-omni `3580c1a` from local clones of their public repositories. Paths below are relative to each repository root.

## 1. Package placement

| Concept | Package | Imports |
|---|---|---|
| Step types, `StepFieldSchema` (field names, shapes as symbols, dtypes as strings), per-primitive capability records, `AlgorithmSpec` and estimator declarations (by name), contract emitter, contract transport interface with POSIX-directory and object-store implementations (02), `ConsumptionRecord` schema and outcome enum, file formats | A `gatedtrain-contracts` | Pure Python; `pyarrow` and the object-store client optional. No torch, Ray, or NeMo RL |
| `StepBatch` tensor view, `ModelAdapter`, `StepForward`, `StepOutputs`, loss-kernel signature and kernel registry, SDE step log-probs, Gaussian KL forms, segment-level ratios, external per-step KL (former hooks 2 and 3), step selectors, group samplers, row filters, DiffusionNFT and Flow-GRPO kernels | B `gatedtrain-denoise` | torch and A |
| `STEP_LOGPROB`, `GlobalNormCounts` plumbing, field registration, `unclassified`, callbacks, emitters, and the other hooks in §3 | C-up (upstream NeMo RL PRs) | NeMo RL only; never imports A |
| `LossFunction` wrappers around B kernels, estimator registrations, `KVBatchMeta` ↔ `StepBatch` adapter, emitter wiring | C-ext `nemo-rl-gatedtrain` | NeMo RL, A, B |
| AutoModel `StepForward` for diffusion, SDE rollout engine, rewards, recipes | D `nemo-rl-diffusion` | C-ext, B, A |
| Fused vectorized execution, continuous-control `StepForward`, kernel wrappers | H `nemo-rl-embodied` | A, B; C-ext optional |
| verl / verl-omni wrappers (path V) | verl-side adapter | verl, A, B |

**Rules** (overview §4).

1. A is importable without torch. Anything typed with `torch.Tensor` or `nn.Module` is in B; A refers to B objects only by registered name.
2. A function in B receives tensors and a `StepBatch` (a mapping from schema field names to tensors plus the schema version). Only C-ext, D, and H touch `KVBatchMeta`, `BatchedDataDict`, or `DataPlaneClient`.
3. C-up PRs contain only generic NeMo RL changes; framework code that adapts NeMo RL to A and B lives in C-ext. Under path V (overview §3), A and B are used unchanged through the verl-side adapter.

Overview dependency rule 2 reads: **signing is outside the trainer; verification is in A's library.** The in-allocation agent `gatedtrain-runtime-agent` (R) signs verbs and pointers with the per-attempt key; the trainer verifies them only by calling A's verification functions from C-ext's emitter wiring, and never holds a signing key (R9-9).

## 2. NeMo RL baseline at `4f39fd8`

### 2.1 Reused as-is

| Area | Current implementation (verified) | Use here |
|---|---|---|
| Data plane | `DataPlaneClient` (`nemo_rl/data_plane/interfaces.py:524`), the "stable boundary" (`nemo_rl/data_plane/README.md:53`); `KVBatchMeta` carries `fields` and per-row `tags` (`interfaces.py:348-371`) | Carrier for step fields (§5) |
| Controller | `SingleController` (`algorithms/single_controller.py`, 6,685 lines) reads metadata only (`:18`); PPO by branches on `_is_ppo` (`:473-484`); optimizer-commit fence `_begin_optimizer_commit` (`:5327`) set under `_checkpoint_save_lock` (`:3772-3773`) and cleared at `:3965`, with PPO early refit inside it (`:3788-3808`) | Unchanged; extended by callbacks (§7.7) |
| Loss path | `LossFunction` protocol with `loss_type`, `input_type`, `global_valid_seqs`, `global_valid_toks`, `**kwargs` (`algorithms/loss/interfaces.py:62-85`); `LossInputType` members `LOGIT`, `LOGPROB`, `OPD_FULL`, `DISTILLATION`, `DISTILLATION_CROSS_TOKENIZER`, `DRAFT` (`:53-59`); optional `metric_normalizations` (`:28-50`); call site `self.loss_fn(data=..., global_valid_seqs=..., global_valid_toks=..., **loss_input)` (`loss/wrapper.py:156-161`) | C-ext wrappers register here (§7.1); hook 1 dispatches on `input_type` (§3.2) |
| Clipped PG | `ClippedPGLossFn` (`loss/loss_functions.py:278`); `__call__` (`:698-709`) has named parameters and **no** `**kwargs`; it shifts every input by `[:, 1:]` in its body (`:754-763`). Configures PPO, GRPO, REINFORCE, GSPO, CISPO; truncated IS (`:210-226`); `force_on_policy_ratio` (`:237`) | Stays token-only |
| KL | `calculate_kl` with `k1`, `k2`, `k3` (`algorithms/utils.py:56-104`); default `k3` (`loss_functions.py:193`) | Token KL unchanged |
| Advantages | GRPO, GDPO, REINFORCE++, raw reward, GAE, OPD (`algorithms/advantage_estimator.py:94-666`), chosen by `if` chains in `_create_advantage_estimator` (`grpo.py:3081`, chain `:3096-3130`; `ppo.py:1241`, chain `:1263-1277`); groups keyed by the `group_id` tag (`data_plane/schema.py:37`) | Registry is upstream hook 10 (§7.2) |
| GAE | Token-level, `done`-masked, zero terminal bootstrap; one scalar `gae_gamma` (`advantage_estimator.py:85`, used at `:649`); no `terminated`/`truncated` distinction | Per-row bootstrap and per-step discount are a gated ask (P6a) |
| Dynamic sampling | DAPO-style filtering of zero-variance groups with regeneration: `dynamic_sampling` (`grpo.py:2167`) and `_apply_dynamic_sampling` on the data plane (`grpo_sync.py:166`) | LLM path unchanged; `post_rollout` generalizes it (§7.7) |
| Old log-probs | `grpo_train_sync` recomputes `prev_logprobs` unless `force_on_policy_ratio` (`grpo_sync.py:966-1003`) | `recompute` strategy (§7.4) |
| Distillation, weight sync, buffers, fleet health, actor envs | `get_topk_logits` (`models/policy/interfaces.py:108`); `WeightSynchronizer` (`weight_sync/interfaces.py:77`); `TQReplayBuffer`, `DataPlaneCheckpointBarrier` (`algorithms/async_utils/replay_buffer.py:1158`, `:300`); `GenerationFleetHealth` (`models/generation/fleet_health.py:133`); `PY_EXECUTABLES` (`distributed/ray_actor_environment_registry.py:20`) | `topk_logits` primitive; rest unchanged (02, 07) |
| Failure taxonomy | `classify_rollout_failure` maps unknowns to `DATA` (`experience/failures.py:315-345`); callers test `is FailureClass.INFRA` (`experience/rollout_manager.py:434`, `:1443`, `:1704`, `:2626`, `:2804`; `environments/gym_checkpoint_coordinator.py:109`) | Hook 5 |

### 2.2 What must be generalized

| Contract | Current shape | Limitation |
|---|---|---|
| Sample model | `DatumSpec` with message logs (`data/interfaces.py:63`) | No step types, decision segments, or episode boundaries |
| Policy log-probs | `PolicyInterface.get_logprobs` (`models/policy/interfaces.py:73`): one causal forward over a packed sequence | Diffusion and flow heads need a forward per step on stored state |
| Loss input | `ClippedPGLossFn` requires `next_token_logprobs [B, S-1]` and `global_valid_toks` (`loss_functions.py:700-751`) | Token shift and normalization are wrong for non-token steps; handled by B kernels, not by changing this class |
| Environment I/O | `observations: list[dict[str, str]]` (`environments/interfaces.py:44`) | Text only |

### 2.3 Trainer path and guarantee levels

The data plane is opt-in (`data_plane.enabled: false`, "the legacy grpo_train trainer never engages this", `examples/configs/grpo_math_1B.yaml:701-706`; backends `simple` and `mooncake_cpu`, `:707`). Only `grpo_sync.py`, `single_controller.py`, `sft_v2.py`, `opd.py`, and `advantage_actor.py` use it; `dpo.py`, `rm.py`, and `sft.py` do not. Step fields and `STEP_LOGPROB` therefore exist only on data-plane trainers; artifact lineage and lockfiles (05) apply to every trainer. Who maintains `grpo_train_sync` parity is recorded at G0 (overview §3).

### 2.4 Corrections to earlier baselines

- Revision 7 claimed existing losses ignore an extra keyword via `**kwargs`. False: `ClippedPGLossFn.__call__` (`loss_functions.py:698`) has no `**kwargs`, and the call site (`wrapper.py:156-161`) passes keywords, so an unconditional extra keyword would break every LLM recipe. Hook 1 is corrected in §3.2.
- Revision 7 said the estimator registry needs no hook. It replaces the `if` chains in `grpo.py` and `ppo.py`, so it is upstream hook 10 (§7.2).
- From revision 6: `calculate_kl` supports `k2`; the NeMo-Gym multimodal dedup limit is enforced by `_validate_multimodal_dedup_capability` (`grpo.py:599-625`, called at `:773`); NeMo AutoModel ships Wan 2.1 and 2.2 flow recipes (`nemo-automodel/examples/diffusion/finetune/wan2_1_t2v_flow.yaml`, `wan2_2_t2v_flow.yaml`); the run controller is not an extension of `SingleController` and reports only through the run contract (02).

## 3. Upstream hooks (overview §5)

Each hook is a separate behavior-preserving PR in C-up with its own tests. With every hook disabled or unused, LLM recipes produce identical losses and metrics.

### 3.1 Hook list (revision 9)

| # | Hook | What it changes in NeMo RL | When | Used by |
|---|---|---|---|---|
| 1 | `LossInputType.STEP_LOGPROB`; `GlobalNormCounts`; `MetricNormalizer.STEPS` | §3.2 | Before G0 | §7.1; **path N only** (path V registers kernels through verl) |
| 4 | Optional field registration | Partitions declare step types; only their required fields are registered; `TOKEN`-only partitions unchanged | Before G0 | §5 |
| 5 | `unclassified` failure class | `classify_rollout_failure` may return `UNCLASSIFIED`, treated as `DATA` by all callers | Before G0 | 07 |
| 6 | Fleet-health reporting callback | `GenerationFleetHealth` calls a registered callback on restart actions | After G0 | 03, 07 |
| 6b | Post-refit weight digest | Opt-in per-shard digest after refit, after a prototype | P4 | 02 |
| 8a-1 | Run-contract emitters and `stop` | Heartbeat with progress counter, completion manifest, cut manifest; `stop` over the authenticated contract transport (A; used by R's agent); controller epoch recorded once per cut manifest and ledger-segment header, not per row | By P2; may ship first as a C-ext shim | 02 |
| 8a-2 | `pause_and_checkpoint` | The verb with `hold`/`continue`, which makes it a controller state machine | After 8a-1 and a prototype | 02 |
| 8d | Per-step provenance export | Callback exporting each optimizer step's consumed row IDs and versions to the ledger writer; drop reasons from the several drop sites follow in a later PR | P2 (consumed rows); may ship first as a C-ext shim | 05 |
| 11 | Authenticated data-plane transport | Only if the transport cannot be configured to authenticate | P1b | 06 |
| 7 | Rollout and step callbacks | `post_rollout`, `pre_advantage`, `post_optimizer_step`; ordering in §3.3; worker RPC | P5, after a DiffusionNFT prototype | §7.7 |
| 8b | Contract-mode checkpoint storage | Attempt- and epoch-scoped storage, no run-side deletion, restore from a given cut | P3 | 02 |
| 9 | `CheckpointParticipant` | Extracted from `environments/gym_checkpoint_adapter.py:178` | P4 | 02 |
| 10 | Advantage-estimator registry | Replaces the `if` chains in `grpo.py` and `ppo.py` (§7.2) | P5 | §7.2 |

**C-ext shims (R9-3).** 8a-1 and 8d may ship first as C-ext shims against the pinned NeMo RL, so P2 does not wait on upstream merge pace; the upstream PRs later replace the shims with the same emitted records.

**Gated asks**, raised only in the named phase: per-row bootstrap value and per-step discount for GAE (P6a, §5.3); context-parallel slice fetch from the data plane if a D adapter cannot provide it (P4).

**Retired:** hook 2 (external per-step KL) and hook 3 (segment-level ratios) stay in B, inside kernels, because no NeMo RL token-path user needs them (NRL2-3); version packing as `(lineage_epoch, counter)` (versions stay NeMo RL integers; lineage identity is `(controller_epoch, counter)`, 02); in-run leases and self-fencing (the run adapter fences, 02); a `SingleController` stage runner (§7.7). Variable `dp_size` stays with P12.

### 3.2 Hook 1, corrected

- Adds the enum member `LossInputType.STEP_LOGPROB` and `MetricNormalizer.STEPS` (so split-API trainers can rescale step-normalized metrics).
- Adds a `GlobalNormCounts` value (data-parallel-global counts of valid sequences, tokens, and trained steps per step type) computed by the train loop.
- The call site at `wrapper.py:156-161` passes `global_norm_counts=` **only** when `loss_fn.input_type is LossInputType.STEP_LOGPROB`. All other losses, including `ClippedPGLossFn`, receive exactly today's arguments. No existing signature changes.
- `ClippedPGLossFn` stays token-only and keeps its `[:, 1:]` shift. Non-token losses are C-ext wrappers around B kernels (§7.1).
- Test: every `LossInputType` other than `STEP_LOGPROB` sees byte-identical keyword sets before and after the PR.

### 3.3 Hook 7 ordering and worker RPC

Order within one iteration of `grpo_train_sync` and `SingleController`:

1. Rollout completes and rewards are attached.
2. `post_rollout(ctx)`: may drop rows (recorded in the ledger with a reason) and request more generation up to a declared `max_extra_gen_batches`; it returns before any advantage is computed. It cannot change rows already used by an optimizer step.
3. `pre_advantage(ctx)`: extra scoring (teacher, cost critic, online reward model) writing declared fields.
4. Advantages, training microbatches, optimizer step.
5. `post_optimizer_step(ctx)` runs **inside the optimizer-commit fence**: in `SingleController`, after the optimizer step returns, while `_optimizer_commit_in_progress` is set, and before any refit, including PPO early refit (`single_controller.py:3788-3808`); snapshots skip while the fence is set (`:3768-3773`), so no snapshot or cut sees a half-updated EMA. In `grpo_train_sync`, which has no such fence, it runs **synchronously** after the optimizer step and before both refit and `save_checkpoint` (part 3 below). `ctx.last_step_of_iteration` marks the last step.
6. Refit (early refit inside the fence, or the normal refit) pushes the role named by `AlgorithmSpec.rollout_role` (§7.1), always after step 5.

Callbacks run in the controller. State that lives in sharded trainer workers (EMA copies) is updated through a **worker RPC**: `ctx.run_on_policy_workers(name, args)` calls a function registered in the trainer image on every policy worker and waits for all. A callback exception fails the step before the fence clears; no partial update is committed. Adapter fencing (02) does not suppress callbacks, because built-in callbacks have no side effects outside the attempt; the run never knows it is fenced, so a callback with external side effects must declare them and route them through the contract (02 §2.5), where the adapter withholds them while fenced.

**Hidden parts of hook 7 (R9-3).** Three changes are not visible in the callback signatures; each is prototyped in C-ext before the PR:

1. **Role refit.** `WeightSynchronizer.sync_weights` takes only `timer` and `kv_scales` and transfers the policy weights; it has no role parameter (`nemo_rl/weight_sync/interfaces.py:93-99`). Pushing a `rollout_role` other than `policy` therefore needs either a role argument added in hook 7, or a package-D adapter swap (activate the `rollout_role` adapter on the policy workers through the worker RPC, then run the normal refit).
2. **Worker-state save and restore.** State held only in policy workers (the `old` and `eval_ema` EMA copies) is registered with save and restore functions that the checkpoint path calls on every worker, so EMA state is in every cut (§7.8, 02) and a restore resumes it instead of reinitializing it.
3. **Ordering on `grpo_train_sync`.** `post_optimizer_step` completes before refit and `save_checkpoint` (step 5). Test: a checkpoint taken at any step boundary contains that step's EMA state, and refit pushes the post-callback weights.

## 4. The decision step

### 4.1 Definition

A trajectory is a sequence of decision steps. Each step has a type, a position on one or more nested axes, an action, and, where defined, a log-probability under the sampling policy.

| Model family | Step type | Log-probability | Nested axes |
|---|---|---|---|
| LLM, VLM, CAD-as-code, speech tokens, tokenized actions | `TOKEN` | Softmax | Turn → token |
| Diffusion / flow generation | `DENOISE` | Gaussian transition density under an SDE sampler; flow_grpo mean-reduces over latent dimensions (`flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:89`) | Sample → denoise step |
| Flow action head (VLA) | `DENOISE` inside `ACTION` | Per denoise step | Env step → action chunk → denoise step |
| Continuous control | `ACTION` | Gaussian or squashed Gaussian (tanh Jacobian, stored pre-squash action) | Episode → env step |
| Hybrid (reasoning tokens then an action chunk) | Mixed | Per step type | Env step → decision segment |
| No tractable density | `SAMPLE` | None | Sample |

### 4.2 Rules

1. **Step type is per step.** A row may contain several decision segments of different types.
2. **The credit axis is declared by the algorithm:** `step`, `env_step`, or `sample`. Advantages and values are `[B, S]`; outer-axis credit is broadcast to inner steps by a declared rule.
3. **Bootstrapping and discounting are explicit.** Every outer step carries `terminated`, `truncated`, and `horizon_cut`; truncations and horizon cuts carry `bootstrap_value`; a per-step `discount` or `step_duration` (§5.3) sets γ for that step. GAE runs only along the credit axis.
4. **Normalization follows the algorithm's aggregation mode** (§7.3), computed from `GlobalNormCounts`. The `DENOISE` default divides by trained steps per sample, so subsampling timesteps does not change the effective learning rate.
5. **Mixed-type rows combine by declared per-type weights**; the effective per-type weight is logged every iteration.
6. **The policy declares log-prob reduction and which KL quantities it can compute; the algorithm chooses the KL estimator** (§7.3).

### 4.3 `decision_segment_id` versus NeMo RL segments

NeMo RL's `LogicalRowTag.segment_index` and `segment_count` (`nemo_rl/data_plane/schema.py:219-228`) index a **physical row** among the "forest" rows one logical rollout is split into, and `forest_segment_id(owner_id, i)` (`schema.py:231-233`) is that row's string identity. The new `decision_segment_id` is a `[B, S]` int32 index of a contiguous run of same-type steps **inside one row**; decision segments never span physical rows. No package-A field is named `segment_*` alone; B's segment-level ratio mode is keyed on `decision_segment_id`.

### 4.4 Worked examples (P0 reference tests)

**PPO for flight control, vectorized simulator.** One row per rollout horizon T in one environment instance; with auto-reset a row may span episodes. `step_type = ACTION`; credit axis `env_step`; per-step rewards and, if constrained, `costs.<name>`; `terminated` on a crash; `truncated` at the time limit; `horizon_cut` at T-1 if the episode continues. Each truncation and horizon cut carries `bootstrap_value[k] = V(s_{k+1})`, computed from the sparse `final_observation` (§5.3). GAE uses `discount[k]` and resets at `terminated`. Log-probs include the tanh correction from the stored pre-squash action. A recurrent policy stores `policy_state_init` at each BPTT chunk start and resets its state after every episode boundary inside a chunk (§5.3).

**VLA with a flow action head.** One row per environment rollout; each environment step k emits a chunk from D denoise steps: K × D steps with `step_type = DENOISE`, `env_step_index = k`, `decision_segment_id = k`, `inner_index = d`. Credit axis `env_step`: advantages computed per k are broadcast to its D steps. With chunk-level ratios, the B kernel aggregates log-ratios over the segment's D steps before clipping.

Both examples are executable tests in the conformance kit (A declarations, B kernels) and must pass in P0 and P1a respectively.

## 5. Step-field schema

### 5.1 Mechanism

The schema is a declarative, versioned document in package A (`StepFieldSchema`, semantic version `0.x`, `experimental: true`). Each field declares name, shape symbol, dtype, the step types that require it, the roles allowed to read it, a storage mode (`dense` or `sparse_at_boundaries`), and its carrier: a data-plane **column** or a per-row **tag** in `KVBatchMeta.tags`. A partition declares its step types; hook 4 registers only the required fields. The layout is frozen only after an `ACTION` family is built (overview §8); readers reject unknown major versions.

**Token partitions carry zero new fields.** For `TOKEN`-only partitions the schema maps step concepts to existing fields by alias: `step_logprobs` → `generation_logprobs`, `step_mask` and `train_step_mask` → `token_mask`, `step_type` implied by the partition. P0 measures zero new bytes per token.

### 5.2 Core fields

| Field | Shape | Carrier | Required for |
|---|---|---|---|
| `step_type` | `[B, S]` int8 | Column, only in mixed-type partitions | Mixed |
| `decision_segment_id`, `env_step_index`, `inner_index` | `[B, S]` int32 | Column | `DENOISE` in `ACTION`, mixed |
| `step_logprobs` | `[B, S]` | Column | Any with a density |
| `store_step_mask`, `train_step_mask`, `step_mask` | `[B, S]` bool | Column (`train_step_mask` may be written at train time) | Non-token |
| `step_state` | `[B, S_stored + 1, ...]` | Column, dtype from the `StepForward` record | Non-token |
| `row_cost` | `[B]` | Column | Non-token data-parallel balancing |
| `rewards.<name>`, `costs.<name>` | `[B, K]` or `[B, S]` | Column | Per declared reward and cost |
| `outer_mask`, `terminated`, `truncated`, `horizon_cut`, `bootstrap_value` | `[B, K]` | Column | `ACTION` |
| `advantages`, `values` | `[B, S]` | Column (existing) | All |
| `group_id`, `noise_group_id`, `agent_id` | Tag | `KVBatchMeta.tags` (`schema.py:37`) | Grouped estimators, shared noise, multi-agent |
| `weight_version`, `sampler_config_digest` | Tag | `KVBatchMeta.tags` | Provenance (02, 05) |

Scalars per row are tags and add no tensor bytes. `controller_epoch` is constant within an attempt, so it is recorded once per cut manifest and ledger-segment header (hook 8a-1), not as a row tag. Composition identifiers are not in this schema; composition uses a generic `parent_ref` (04).

### 5.3 Embodied fields

| Field | Shape | Meaning |
|---|---|---|
| `policy_state_init` | `[B, C, ...]` | Recurrent state at the start of each of C BPTT chunks, recorded at rollout |
| `privileged_obs` | `[B, K, ...]` | Simulator ground truth readable only by `critic` and `teacher` roles |
| `critic_obs` | `[B, K, ...]` | Critic input for asymmetric actor-critic; defaults to an alias of the policy observation |
| `costs.<name>` | `[B, K]` | Constraint costs, estimated separately from rewards; shield behavior in 06 |
| `discount` or `step_duration` | `[B, K]` float | Per-step γ, or the step's duration Δt with γ_k = γ^{Δt_k} (semi-Markov options, variable control rates); exactly one may be declared; default γ constant |
| `env_params_version` | `[B, K]` tag | Domain-randomization and curriculum parameter version (08) |
| `final_observation` | sparse | Observation before an auto-reset, stored only at boundary indices: values `[M, ...]` plus `final_observation_index [M, 2]` (row, k) for the M truncations and horizon cuts in the batch; never dense `[B, K, ...]` |
| `obs_normalizer_version` | `[B]` tag | Normalizer statistics version used at rollout (08) |
| `intervened` | `[B, K]` bool | The safety shield replaced the policy's action at this step (06) |
| `proposed_action` | `[B, K, ...]` | The policy's action before the shield, stored next to the executed action wherever `intervened` can be true |

**Intervention training rule.** An algorithm that reads `ACTION` steps with possible interventions must declare `intervention_rule`:

- `train_on_proposed_plus_cost`: the ratio uses `proposed_action`'s log-prob, and the intervention adds a declared cost to `costs.shield` at that step.
- `mask`: intervened steps are excluded from the policy loss (still used for value targets) and counted as untrained in `GlobalNormCounts`.
- `executed_with_is`: the executed action is trained with importance weight π(a_exec)/μ(a_exec), where μ is the shield's declared behavior density, truncated at a declared bound.

**Out-of-support actions.** An executed action with zero density under the policy (for example a squashed-Gaussian boundary value with no pre-squash preimage, or an action outside declared bounds) is never trained with `executed_with_is`: the step falls back to `mask`, is counted in `out_of_support_steps`, and the count is logged and checked against a declared limit. A missing `intervention_rule` is a validation error when the environment declares a shield.

**Read sets** are enforced: `StepForward.step_logprobs` for the `policy` role receives a `StepBatch` without `privileged_obs`.

**Recurrent policies** require the **sequence-chunk microbatch constraint**: a microbatch holds whole chunks of L consecutive environment steps of one row, in order, starting at a stored `policy_state_init`; padding to L is masked. **Hidden-state reset inside a chunk:** at step k+1 after `terminated[k]`, `truncated[k]`, or an auto-reset, the recurrent state is replaced by the policy's initial state, both at rollout and in the training rerun; the reset mask is derived from the boundary flags, not stored. Conformance compares rollout-time and train-time hidden states across a mid-chunk reset.

**Storage rule.** Per-step state stays inside the row: `get_samples` requires every requested field on every requested row, and per-step rows would multiply controller metadata by S (256 rows × 100 env steps × 10 denoise steps = 256,000 rows per iteration for VLA). Fields marked `sparse_at_boundaries` are the exception allowed by the schema. Payload sizes are in 08.

## 6. `ModelAdapter`, `StepForward`, and capability records

### 6.1 `ModelAdapter` (package B)

```python
class ModelAdapter(Protocol):            # package B; implemented in D, H, or by existing workers
    family: str
    def build(self, backend_cfg) -> nn.Module: ...
    def sharding_plan(self, backend: str) -> ShardingPlan: ...      # incl. within-step context parallelism
    def weight_name_map(self, target: str) -> Mapping[str, str]: ... # rollout engine or exporter
    def export_spec(self) -> ExportSpec: ...  # formats; the framework exporter (05/06) writes artifacts
    records: AdapterRecords  # package-A dataclass: parallelism limits, precisions, weight transports, versions
```

An adapter without a `StepForward` is enough for SFT. Export goes through the framework exporter so manifests, hashes, and egress policy (06) apply uniformly.

### 6.2 `StepForward` (package B)

```python
class StepForward(Protocol):             # package B; tensors and StepBatch only
    step_types: frozenset[StepType]
    def microbatches(self, batch: StepBatchMeta, budget: MemoryBudget,
                     constraints: MicrobatchConstraints) -> Iterable[StepMicrobatch]: ...
    def step_logprobs(self, model, mb: StepMicrobatch, role: str) -> StepOutputs: ...
    # StepOutputs: log-probs [n] plus, when declared, transition mean and std for KL
    primitives: Mapping[str, PrimitiveCapability]
```

Optional primitives, each with its own record: `raw_prediction(role, mb)`; `noised_input(mb, timesteps, noise)`; `topk_logits(mb, k)`; `value(mb, obs_field)`; `action_distribution(mb)`; `sft_loss(kind, mb)`.

`TOKEN` implementations (in C-ext) wrap `get_logprobs`, `get_topk_logits`, and the existing value model. `DENOISE` implementations live in D on AutoModel and call B's per-step math, which any verl-omni adapter shares.

### 6.3 Capability records (package A)

```python
@dataclass(frozen=True)
class PrimitiveCapability:               # pure Python; validated without importing torch
    name: str                            # "step_logprobs", "raw_prediction", ...
    contract_version: str
    step_types: frozenset[StepType]
    roles: frozenset[str]                # policy, reference, old, eval_ema, teacher, critic
    params: Mapping[str, Any]            # {"reduction": "mean", "kl_quantities": {...}}
    fusable_with: frozenset[str]         # roles computable in the same microbatch pass (§7.6)
    reference_strategies: frozenset[str] = frozenset()  # "lora_disabled", "separate_model"
```

Validation matches an algorithm's requirements against these records and the adapter records: primitives and roles, log-prob kind, KL quantities, step selection mode, credit axis, microbatch constraints, weight transport, and parallelism limits. Anything else fails at runtime with an error naming the missing record.

### 6.4 Rollout engine and environment

The rollout engine writes `store_step_mask` at sampling time, receives a `GroupPlan` from the group sampler (§7.5), records `sampler_config_digest` on every row, and serves the weights of `AlgorithmSpec.rollout_role` (§7.1). `Environment` and `RewardFn` keep revision 6 §6.5 minus `context` (moved to 04); `EnvStepResult` gains `costs`, `privileged_obs`, `proposed_action`, and `intervened`. Code-executing environments declare `sandbox: required` (06).

## 7. Algorithm plug-ins

### 7.1 One spec, one B-level loss kernel

The math of an algorithm is a **loss kernel** in package B:

```python
def loss_kernel(role_outputs: Mapping[str, StepOutputs],  # per role: policy, old, reference, ...
                batch: StepBatch,       # advantages, masks, rewards, and other declared fields
                cfg: KernelConfig,      # hyperparameters from the spec: clip ranges, KL coef, β, ...
                counts: NormCounts) -> tuple[Tensor, dict]: ...   # -> (loss, metrics)
```

Kernels read role outputs by name, declared batch fields, and hyperparameters, so DiffusionNFT's three-role mix fits without a signature change. `NormCounts` is B's framework-neutral count type; wrappers fill it from NeMo RL's `GlobalNormCounts` or the trainer's own counts. `AlgorithmSpec` (package A) names the kernel; it does not reference NeMo RL's `LossFunction`:

```python
@register_algorithm("flow_grpo")
class FlowGRPO:
    spec = AlgorithmSpec(
        loss_kernel="gatedtrain_denoise.flow_grpo_clip",   # registered B kernel
        advantage_estimator="grpo", kl_estimator="gaussian_mean_sq",
        aggregation="sample_mean_step_mean",
        old_policy="rollout", on_drift="error", rollout_role="policy",
        step_selector="fraction_prefix", group_sampler="independent_noise",
        row_filter="zero_advantage_pad", credit_axis="sample", broadcast="same_for_all_steps",
        roles={"reference": "lora_disabled"}, fuse=frozenset({"reference", "policy"}),
        ablations={"kl": "kl_coef>0", "row_filter": "row_filter!=none"},
    )
```

**Wrappers.** Each trainer adapts kernels to its own loss protocol with one generic wrapper, not one per algorithm: C-ext's `StepLogprobLoss` is a `LossFunction` with `input_type = STEP_LOGPROB` that unpacks `data` and `global_norm_counts` into kernel arguments; D and H reuse it or their own equivalent for the fused mode; the verl-side adapter registers the same kernel through `register_policy_loss` (`verl/trainer/ppo/core_algos.py:50-53`). verl-omni keeps a separate `DIFFUSION_LOSS_REGISTRY` (`verl_omni/trainer/diffusion/diffusion_algos.py:146-165`); this design keeps one kernel registry for all trainers.

**`rollout_role`** (default `policy`) names the role whose weights are pushed to the rollout engine at each refit. DiffusionNFT samples with the EMA `old` adapter: `set_adapter("old")` precedes the sampling call (`diffusionnft/scripts/train_nft_sd3.py:640`), so its spec sets `rollout_role="old"`. Validation checks that the role is served by the adapter's weight transport and refits after `post_optimizer_step` (§3.3).

**Inline math in a user package.** A user package may define its kernel inline in the same file as its spec and register it under its own namespace (`user.my_algo`). It then works on every trainer that has the generic wrapper. Upstreaming the kernel to B is a later, separate step that requires B's parity tests; promotion to a production workflow does not require it.

### 7.2 Advantage-estimator registry (hook 10)

`register_advantage_estimator(name)` replaces the `if` chains in `_create_advantage_estimator` (`grpo.py:3081`, `ppo.py:1241`). Because it changes those files it is an upstream ask (hook 10, P5), not framework code. Existing estimators register under their current names, so configs are unchanged. Until hook 10 lands, production C-ext supports only the built-in estimators on NeMo RL, plus estimators computed in `pre_advantage` into the `advantages` field once hook 7 exists; the research profile uses its local shim (§8.3); the fused mode in H uses the registry directly. Estimators take tensors and tags (`group_id`, credit axis, `outer_mask`, bootstrap and discount fields) and return `[B, S]` advantages and optional returns.

### 7.3 What the algorithm declares

| Declaration | Values | Notes |
|---|---|---|
| Loss kernel | Registered B or user-package kernel | §7.1 |
| KL estimator | `k1`, `k2`, `k3` (existing `calculate_kl`); `gaussian_mean_sq`; `gaussian_exact` | `gaussian_mean_sq` is flow_grpo's `‖μ − μ_ref‖² / (2σ²)` averaged over latent dimensions (`flow_grpo/scripts/train_sd3.py:900-901`): exact only with equal σ, and 1/d of the summed KL. `gaussian_exact` uses both variances. Computed in the B kernel (former hook 2) |
| Aggregation mode | `token_mean`, `seq_mean_token_mean`, `seq_mean_token_sum`, `sample_mean_step_mean`, `step_mean_per_type` | From `GlobalNormCounts`; names match verl's `agg_loss` where they overlap (`core_algos.py:1140-1194`) |
| Ratio level | `step`, `decision_segment`, `sequence` | `decision_segment` is computed in the B kernel (former hook 3) |
| Old-policy strategy, drift policy | `rollout`, `recompute`, `lagged`, `ema`; `on_drift: error \| recompute` | §7.4 |
| Rollout role | `policy` (default), `old`, or another declared role | §7.1 |
| Step selector, group sampler, row filter | Registered functions | §7.5 |
| Credit axis, broadcast rule, per-type weights | §4.2 | |
| Intervention rule | `train_on_proposed_plus_cost`, `mask`, `executed_with_is` | §5.3; required with a shield |
| Roles, residency, primitives | §7.8 | Validated against records (§6.3) |
| Microbatch constraints | `pair`, `prompt_group`, `sequence_chunk(L)` | |
| Callbacks | `post_rollout`, `pre_advantage`, `post_optimizer_step` | §7.7 |

Clip ranges and KL coefficients are per step type: flow_grpo's mean-reduced log-probs do not share the token scale.

### 7.4 Old-policy strategy and drift

- **`rollout`** uses log-probs recorded by the sampler as the ratio denominator; the Flow-GRPO default (D5). flow_grpo computes rollout log-probs with the module it trains (`train_sd3.py:643-661`) and forms `ratio = exp(log_prob − sample["log_probs"][:, j])` (`:891`); DanceGRPO does the same (`dancegrpo/fastvideo/train_grpo_flux.py:595`). Precondition: shared modules, weights, dtype, and kernels. Saves the forward that `recompute` adds per trained step: **1/c of training compute**, where c is the forward-equivalents per trained step without that forward (33% at c = 3, 25% at c = 4, 20% at c = 5), or (R/c)/(1 + R) of the iteration's transformer compute; derivation and c values in 08 §4.4.
- **Drift check.** The conformance test (§9.2) and a periodic in-run probe measure the frozen-weight rollout/train log-ratio. If it exceeds the family threshold, the run **fails validation (or stops with reason `old_policy_drift`)** by default. Only with `on_drift: recompute` declared does it switch to `recompute`; the switch, the measured drift, and the step are recorded in the run's provenance and shown in the run summary, because it changes the algorithm. There is no silent fallback.
- **`recompute`** runs a trainer forward under pre-update weights: the LLM GRPO default (`grpo_sync.py:994-996`) and required for disaggregated engines with different numerics; generation log-probs then feed importance-sampling correction (`loss_functions.py:202`, `:210-226`). verl-omni recomputes unless the loss reads no log-probs (`verl_omni/trainer/diffusion/diffusion_trainer_utils.py:79-92`).
- **`lagged`** uses the previous iteration's weights. **`ema`** keeps an EMA copy with a declared decay schedule, updated in `post_optimizer_step`. **`eval_ema`** is not a ratio denominator; it is an EMA kept for evaluation and export.

### 7.5 Stored steps, trained steps, groups, and filters

- `store_step_mask`: chosen by the rollout engine; only these steps' states go into `step_state` (Flow-GRPO-Fast stores its SDE window).
- `train_step_mask`: chosen by the **step selector** from stored steps, at rollout or train time, per iteration or per epoch.

```python
StepSelector = Callable[[StepSelectContext], Tensor]   # -> train_step_mask [B, S]
GroupSampler = Callable[[GroupContext], GroupPlan]     # seeds, initial noise, noise_group_id
RowFilter    = Callable[[RowFilterContext], Tensor]    # -> keep mask [B], logged to the ledger (05)
```

| Reference behavior | Expressed as |
|---|---|
| flow_grpo trains the first `timestep_fraction` of stored steps (`train_sd3.py:345`, `:869`) | Selector `fraction_prefix` |
| DanceGRPO draws a fresh per-sample timestep permutation at train time (`train_grpo_flux.py:551-570`) | Selector `train_time_permutation`, seeded from the iteration RNG |
| DanceGRPO `init_same_noise` shares one initial latent (`train_grpo_flux.py:333-348`) | Group sampler `shared_initial_noise`; `noise_group_id` tag |
| flow_grpo drops all-zero-advantage samples, then re-admits random dropped samples to fill the batch (`train_sd3.py:801-821`) | Row filter `zero_advantage_pad`; dropped and re-admitted rows in the ledger |
| DAPO drops zero-variance groups and generates more until the batch is full (`grpo.py:2167`, `grpo_sync.py:166`) | LLM: existing dynamic sampling. Other families: `post_rollout` callback (§7.7) |
| DiffusionNFT re-noises clean latents at selected timesteps (`diffusionnft/scripts/train_nft_sd3.py:868`) | Train-time selector plus `noised_input` |

A row filter only drops rows inside the current batch; generating more rows needs `post_rollout`. Normalization by trained steps uses the count of `train_step_mask`.

**Shared noise and group normalization (RES3-5).** Group-mean centring includes each sample's own reward in its baseline, so it scales the gradient by (G−1)/G relative to a leave-one-out baseline, with or without `shared_initial_noise` (common random numbers). Group-std normalization is biased with or without CRN, because 1/σ̂ depends on the sample's own reward. CRN defaults for composition are in 04 §10.2.

### 7.6 Fused forwards

When the `StepForward` record lists roles in `fusable_with`, the trainer loads a microbatch's `step_state` once and runs those forwards back to back: reference (no grad), old (no grad, only while weights equal the pre-update weights), and policy (with grad). flow_grpo computes the reference mean under `disable_adapter()` in the same loop as the policy forward (`train_sd3.py:878-883`). Fusion is a scheduling choice: results must equal the unfused path within tolerance.

### 7.7 Callbacks and the stage-runner trigger

Hook 7 (P5) provides three callbacks with the ordering in §3.3:

- `post_rollout(ctx)`: DAPO-style resampling for non-token families, online reward-model updates from fresh rollouts, and merging several rollout batches into one training batch. Resampling is bounded by `max_extra_gen_batches`; exceeding it stops with a declared reason.
- `pre_advantage(ctx)`: teacher or cost-critic scoring.
- `post_optimizer_step(ctx)`: EMA updates and diagnostics, through the worker RPC for sharded state.

DiffusionNFT maps exactly: `eval_ema` steps after each gradient step (`train_nft_sd3.py:1023-1028`); the `old` adapter is blended once per iteration with `return_decay` (`:169`, `:1033-1038`) in `post_optimizer_step` with `ctx.last_step_of_iteration`; then refit pushes `old` (`rollout_role`). Hook 7 is requested only after a DiffusionNFT prototype in C-ext validates this ordering.

**Stage-runner trigger.** A `SingleController` stage runner is proposed upstream when **one committed user** (a pilot user with a written success criterion) needs ordering these three callbacks cannot express, not after three algorithms.

### 7.8 Model roles and residency

| Role | Trained | Strategies / notes | Default residency |
|---|---|---|---|
| `policy` | Yes | The `StepForward` being trained | GPU |
| `reference` | No | `none`, `lora_disabled`, `separate_model` | GPU when fused, else declared |
| `old` | No | §7.4; read in every training forward under `ema` | GPU |
| `eval_ema` | No | Read only for evaluation and export | CPU |
| `critic` | Yes, own optimizer | `value`; may read `critic_obs` and `privileged_obs` | GPU |
| `teacher` | No | Frozen, possibly another family; may read `privileged_obs` | Declared |

**Residency follows read frequency.** A role read in every training forward, or named as `rollout_role`, defaults to GPU; a role read only at evaluation, export, or checkpoint defaults to CPU. A spec may override per role; validation warns when an every-forward role is placed on CPU. The DiffusionNFT `old = ema` case is therefore GPU-resident and `eval_ema` CPU-resident; memory and transfer numbers are in 08 §5.3. Every role's weights and state are part of the checkpoint cut (02).

### 7.9 Expressibility

| Algorithm | Roles and rollout role | Primitives and declarations | Status | Phase |
|---|---|---|---|---|
| GRPO, DAPO (LLM) | policy, reference, old = `recompute`; rollout `policy` | Existing token path, zero new fields; DAPO via existing dynamic sampling (`grpo.py:2167`) | Existing | P0 parity |
| Flow-GRPO | policy, reference = `lora_disabled`, old = `rollout`; rollout `policy` | `step_logprobs` with Gaussian moments; `gaussian_mean_sq`; `fraction_prefix`; `zero_advantage_pad`; fused ref+policy | Kernel in B, `StepForward` in D | P1a (B), P1b (D) |
| DAPO-style resampling (diffusion) | as Flow-GRPO | `post_rollout` drops zero-variance groups and requests more generation | Needs hook 7 | P5 |
| DanceGRPO | as Flow-GRPO | `train_time_permutation`; `shared_initial_noise` | New selector and sampler | P5 |
| PPO with critic (token) | policy, reference, old = `recompute`, critic | `value`; existing GAE | Existing | P0 parity |
| PPO with critic (continuous control, asymmetric critic, costs, shield) | policy, old = `recompute`, critic | `action_distribution`, `value(critic_obs)`; `env_step` GAE with bootstrap and `discount`; `intervention_rule`; `sequence_chunk(L)` if recurrent | New in H | P6a |
| VLA flow head | policy, old, reference | `DENOISE` in `ACTION`; `decision_segment` ratios | New in H with a named rollout engine | P6b |
| DPO | policy, reference | `step_logprobs`; `pair` | Needs data-plane DPO trainer | P7 |
| Diffusion-DPO | policy, reference | `raw_prediction`, `noised_input`; `pair`; offline recipe in D | New in B and D | P5 |
| Token distillation | policy, teacher | `topk_logits` | Existing | P0 parity |
| Privileged teacher → student | policy, teacher | Teacher reads `privileged_obs`; student read set excludes it | New in H | P6a |
| DiffusionNFT | policy, old = `ema` (GPU), reference = `lora_disabled`, eval_ema (CPU); **rollout `old`** | `raw_prediction` for three roles, `noised_input`; `post_optimizer_step` EMA updates via worker RPC; synchronous only | Kernel in B; D; hook 7 | P5 |

DiffusionNFT, as its reference code does: samples with the `old` adapter (`train_nft_sd3.py:640`); re-noises stored clean latents (`:868`); takes velocity predictions from the current and `old` adapters (`:876`) and a LoRA-disabled reference (`:897`); forms `β·v + (1−β)·v_old` and `(1+β)·v_old − β·v` (`:934-937`); regresses both to the clean latent with adaptive weights; mixes by a reward-derived weight `r` (`:959`). It needs no rollout log-probs. Asynchronous use is undefined (EMA versus stale rows) and is rejected at validation.

## 8. Research profile (D15)

### 8.1 Footprint

| Aspect | Research profile | Production profile |
|---|---|---|
| Services | One node; Ray local; no metadata store or workflow engine | 03, 04 |
| Manifests and formats | Manifests always written. On shared clusters: safe weight formats only and signed cut and artifact manifests (06 B9). On a dedicated node: digests recorded. Lineage inputs are recorded by the launcher from the credentials it issued, not self-reported by the job | Signed (06) |
| Run contract | Files in a local run directory (02); no supervisor | Supervised |
| Plug-ins | Shared cluster: allow-list enforced, code-execution block on. Dedicated node: local paths allowed, digest recorded | Allow-list (06) |
| Data | Policy tags are enforced on the storage side: the launcher issues tag-scoped read credentials that admit only data tagged for research use, on all governed storage regardless of node tenancy; loader code is not the control. Every research run refuses `safety_critical` inputs and outputs (06 §14.4) | Policy tags (06); `safety_critical` only in validated workflows |
| Gates | Exploratory gates on training rewards, **non-blocking**: they log and annotate, never stop, promote, or publish | Statistical gates (05) |
| Outputs | Tagged `unpromotable` in their manifests | Content-addressed (05) |

A "shared cluster" is any node not exclusively allocated to the submitting user for the run; the launcher decides from the cluster profile, not from a user flag. Node tenancy selects only the format, signing, and plug-in rows; it never relaxes data-tag enforcement (R9-7).

### 8.2 Promotion

The production validator rejects any checkpoint or artifact tagged `unpromotable` or lacking production lineage. Promotion has two routes:

1. **Production rerun** under the production profile. The rerun may reuse research artifacts as inputs (for example a reward model, a data filter list, or an initial checkpoint) when each has complete lineage to tagged inputs, safe formats, and signed manifests; otherwise the artifact is regenerated.
2. **Import stage** in a workflow: an approver other than the requester signs the import of a named research artifact after checking its lineage, formats, and data tags; the import is recorded in the audit log and the lineage graph (05, 06).

### 8.3 Features

- **Sweeps:** override sets over one recipe, run sequentially or packed, each with its own run directory and a sweep index file.
- **Ablation switches:** the `ablations` map in `AlgorithmSpec` names each component; toggling one changes only that component and is recorded in provenance.
- **Per-timestep diagnostics:** per stored timestep, ratio mean and standard deviation, clip fraction above and below (as flow_grpo logs, `train_sd3.py:906-929`), approximate KL, frozen-weight rollout/train log-ratio, zero-std group ratio (`train_sd3.py:133-167`), rows filtered, low-signal turns, and effective per-type loss weight.
- **Local estimator and callback shim (until hooks 7 and 10 land; RES3-1):** the research profile ships a C-ext `grpo_train_sync` subclass with a local estimator registry and the three callbacks of §7.7 in the §3.3 order, so researchers do not fork NeMo RL before P5. It runs only under the research profile; outputs stay `unpromotable`, and production uses the upstream hooks once merged.
- **Deterministic single-batch replay:** `--capture-batch N` saves one iteration's microbatch inputs, `GroupPlan`, RNG states, sampler digest, and weight hash; `--replay` reruns forward and backward and compares losses and gradient norms bitwise, or within a declared tolerance.

## 9. Plug-in model and conformance

**Extension points:** `ModelAdapter`, `StepForward` and its primitives, loss kernels, rollout engines, environments, reward functions, algorithm specs, advantage estimators, step selectors, group samplers, row filters, callbacks, data processors, and exporters. Workflow stages, gates, backends, scheduler plug-ins, and composition plug-ins are specified in 03–05.

### 9.1 Plug-in rules

1. Implements one extension-point protocol from package A (declarations) or B (torch protocols).
2. Ships a static manifest validated without importing code: name, semantic version, contract range, capability records, dependencies with version ranges, resources, failure classes, sandbox requirement, licence.
3. Runs in its own Ray actor with its own environment (`PY_EXECUTABLES` pattern), except code that must run inside trainer workers (`StepForward`, kernels, wrappers, selectors, worker-RPC functions), which ships in the trainer image.
4. Is referenced as `name@major.minor`; the lockfile resolves exact versions and digests. Until P3 a digest allow-list replaces a plug-in index; allow-list changes go through reviewed Git merge requests (06).
5. Trust, signing, and allow-listing follow 06.
6. Passes the conformance kit in CI and ships a single-GPU smoke config under the research profile.

### 9.2 Conformance kit

| Extension point | Required tests |
|---|---|
| `StepForward` | Frozen-weight rollout/recompute log-ratio below the family threshold (gates `old = rollout`; above it, validation errors unless `on_drift: recompute`); gradient check; declared reduction and KL quantities verified numerically; fused and unfused agree; `policy` cannot read `privileged_obs`; recurrent state resets at a mid-chunk boundary identically at rollout and train time |
| Loss kernels in B | Parity with flow_grpo, dancegrpo, and diffusionnft on recorded tensors; `gaussian_exact` = `gaussian_mean_sq × d` at equal variances; segment-level ratios match a reference loop; wrappers in C-ext, D, H, and verl produce identical losses from identical inputs |
| `ModelAdapter` | Build and shard on each declared backend; name map round-trips through rollout engine and exporter; `rollout_role` weights reach the engine |
| Rollout engine | Weight-sync round trip changes outputs; `store_step_mask` and `sampler_config_digest` on every row; `GroupPlan` honored |
| Environment, reward | Seeded statistical equivalence; timeouts return masked results; vectorized and scalar agree; `final_observation` present exactly at boundary indices; `proposed_action` present wherever `intervened` |
| Algorithm | Worked examples (§4.4); `TOKEN` algorithms reduce to `ClippedPGLossFn`; parity for claimed published methods; missing records rejected at configuration; ablation switches change only their component; each `intervention_rule` including out-of-support fallback; `discount` and `step_duration` GAE against a reference |
| Callbacks (hook 7) | Order of §3.3; `post_optimizer_step` completes before refit and no snapshot sees a partial update; on `grpo_train_sync`, a checkpoint at any step boundary holds that step's EMA state; `rollout_role` refit pushes that role's weights; callback failure commits nothing |
| Estimator, selector, sampler, filter | Registered names reproduce today's estimators exactly; masks within stored steps; filter and `post_rollout` drops appear in the ledger |
| Schema | `TOKEN`-only partitions add zero fields and bytes; readers reject unknown major versions; A imports without torch |

A supported-combination matrix (family × backend × rollout engine) runs nightly; other combinations are experimental.

## 10. Changes

**Revision 9 (closing):** see Round-3 disposition; also storage-side tag credentials and launcher-recorded lineage (§8.1) and rule-2 wording (§1).

**Revision 8:**

- Packages: A pure Python; `ModelAdapter`, `StepForward`, `StepBatch`, and the loss-kernel signature in B; C split into C-up and C-ext; emitter and ledger record schema in A (§1).
- `AlgorithmSpec` names a B-level kernel with generic wrappers in C-ext, D, H, and verl; inline kernels in user packages allowed; `rollout_role` added (§7.1).
- Hooks restaged (§3.1): hook 1 corrected (§3.2); hooks 2 and 3 retired into B; estimator registry is hook 10; version packing, in-run leases, and the stage runner retired; hook 7 gains `post_rollout`, fence ordering, and a worker RPC (§3.3, §7.7).
- Old-policy drift is an error unless `on_drift: recompute`, which is recorded (§7.4); residency follows read frequency (§7.8).
- Embodied: `proposed_action` with `intervention_rule` and out-of-support handling, `discount`/`step_duration`, sparse `final_observation`, hidden-state reset inside chunks (§5.3).
- Research profile: shared-cluster controls, `unpromotable` outputs, promotion by rerun with reuse or by an approved import (§8). Baseline corrections (§2.4); phases P1a/P1b and P6a/P6b.

**Revision 7** (from revision 6): `StepPolicy` split into `ModelAdapter` and `StepForward` with per-primitive records; algorithm-chosen KL estimator with `gaussian_mean_sq` and `gaussian_exact`; aggregation from `GlobalNormCounts`; Flow-GRPO old policy `rollout`; stored and trained steps separated with pluggable selectors, samplers, and filters; stage vocabulary replaced by callbacks; `decision_segment_id`; declarative, versioned step fields with zero new token fields; embodied fields; research profile; baseline corrections.

## 11. NeMo ecosystem reuse

Rule: wrap, then contribute through the tool's own extension mechanism, then build. Versions other than NeMo RL and AutoModel are as recorded in revision 6 §18 on 2026-11-04 and were not re-verified.

| Tool | Role here | Integration | Gaps |
|---|---|---|---|
| NeMo RL `4f39fd8` | Data plane, controller, weight sync, buffers, loss and estimators | Hooks in §3 via C-up; adapters in C-ext | Step schema, `STEP_LOGPROB`, per-step GAE discount |
| NeMo AutoModel | Trainer for `DENOISE` `StepForward` (FLUX, Qwen-Image, HunyuanImage, Wan 2.1/2.2, LTX in `examples/diffusion/finetune/`) | Through NeMo RL workers, in D | No diffusion RL |
| NeMo Evaluator 0.3.0 | Gate statistics and scorer rewards | Wrap in E-stats (05) | Non-text metrics |
| verl-omni `3580c1a` | Path V trainer; reference for diffusion losses | verl-side adapter to A and B | Separate stack |

Wrapped or reused unchanged: Megatron-Bridge v0.6.0 (largest models, through NeMo RL; no RL in its matrix), NeMo-Run v0.12.0 (launch, 03), NeMo Gym (environment service), NeMo Curator 1.4.0 (data preparation; filter ID wrapper in 05), NeMo Export-Deploy 0.8.0 (LLM and multimodal export; non-LLM exporters new), NeMo Guardrails 0.24.1 (safety filter), NeMo Lens v0.1.0 (telemetry, 07), NVRx v0.7.0 (async checkpointing; not integrated with Ray workers).

## Round-1 disposition

Rows superseded by revision 8 are marked "(rev 8)".

| ID | Section | Resolution |
|---|---|---|
| ARCH-3 | §1, §6, §7.1 | (rev 8) Declarations in pure-Python A, torch protocols and math in B on tensors and `StepBatch`; NeMo RL types only in C-ext, D, and H. |
| ARCH-4 | §6.1–§6.3 | `StepPolicy` is split into `ModelAdapter` and `StepForward`, and `Capabilities` into per-primitive records. |
| ARCH-7 | §4.3 | `decision_segment_id` is an in-row step run, distinct from `LogicalRowTag.segment_index` and `forest_segment_id`. |
| NRL-1 | §3.1, §7.7 | (rev 8) No stage runner is requested; callbacks cover EMA, diagnostics, and resampling, and a runner is proposed only for one committed user who needs it. |
| NRL-3 | §5.1, §5.2 | `TOKEN`-only partitions alias existing fields and carry zero new bytes; per-row scalars are tags. |
| NRL-4 | §3.2, §7.1 | (rev 8) No second loss protocol in NeMo RL: one generic `STEP_LOGPROB` `LossFunction` wrapper in C-ext calls B kernels. |
| NRL-5 | §3.1 | Variable `dp_size` is excluded and deferred to P12. |
| RES-1 | §7.4 | (rev 8) Flow-GRPO uses `rollout` log-probs when modules and kernels are shared; drift beyond threshold is an error unless `on_drift: recompute` is declared and recorded. |
| RES-2 | §4.2 rule 6, §7.3 | The algorithm chooses the KL estimator; `gaussian_mean_sq` is labeled equal-variance and mean-reduced; `gaussian_exact` is added. |
| RES-3 | §4.2, §7.3 | Aggregation is declared and computed from `GlobalNormCounts`; effective per-type weights are logged. |
| RES-4 | §7.5 | Stored and trained steps are separate; selectors, samplers, and filters express DanceGRPO and flow_grpo behaviors. |
| RES-5 | §7.1, §7.2, §7.7 | (rev 8) A new algorithm is one file with a spec and a kernel (inline allowed), plus optional estimator and callbacks. |
| RES-8 | §8 | (rev 8) The research profile runs without store or engine, with sweeps, ablations, diagnostics, replay, and non-blocking gates; on shared clusters it keeps safe formats, signing, allow-lists, and tag checks. |
| PERF-3 | §7.4 | `rollout` avoids the extra DiT forward per trained step. |
| PERF-4 | §7.6 | Fusable roles run in one microbatch pass that reads `step_state` once. |
| EMB-2 | §5.3 | `policy_state_init` and `sequence_chunk(L)` support recurrent policies; (rev 8) state resets at boundaries inside a chunk. |
| EMB-3 | §5.3, §7.8, §7.9 | `privileged_obs` and `critic_obs` with enforced read sets enable asymmetric critics and privileged teacher → student. |
| EMB-9 (fields) | §5.2, §5.3 | `costs.<name>` beside `rewards.<name>`; (rev 8) `proposed_action` and `intervention_rule`; shield in 06. |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| ARCH2-4 | §1 | A is pure Python (schema, records, specs, emitter, ledger schema); `ModelAdapter`, `StepForward`, and the loss-kernel signature are in B; C splits into C-up and C-ext. |
| ARCH2-5 | §7.1 | `AlgorithmSpec` names a B-level kernel `(StepOutputs, advantages, masks, NormCounts) -> (loss, metrics)`, wrapped generically in C-ext, D, H, and verl. |
| NRL2-1 | §2.1, §2.4, §3.2 | Hook 1 passes `GlobalNormCounts` only to `STEP_LOGPROB` losses at `wrapper.py:156-161`, since `ClippedPGLossFn.__call__` (`loss_functions.py:698`) has no `**kwargs`; `ClippedPGLossFn` stays token-only. |
| NRL2-3 | §3.1, §7.3 | External per-step KL and segment-level ratios are computed in B kernels; hooks 2 and 3 are retired. |
| NRL2-5 | §3.3 | `post_optimizer_step` runs inside the optimizer-commit fence before any refit, including early refit, so snapshots never see partial state; sharded EMA uses a worker RPC; failures commit nothing. |
| NRL2-6 | §2.4, §7.2 | The estimator registry is upstream hook 10 (P5) because it replaces `if` chains in `grpo.py` and `ppo.py`. |
| RES2-1 | §7.1, §7.9 | `AlgorithmSpec.rollout_role` selects the weights pushed at refit; DiffusionNFT uses `old`, per `train_nft_sd3.py:640`. |
| RES2-2 | §7.4 | Drift beyond threshold is a validation error or a stop unless `on_drift: recompute` is declared, and any switch is recorded in provenance. |
| RES2-3 | §7.1 | A user package may inline its kernel in one file; upstreaming to B is a later step and not required for promotion. |
| RES2-4 | §3.3, §7.5, §7.7, §7.9 | `post_rollout` supports DAPO-style resampling, online RM updates, and batch merges; the stage runner is triggered by one committed user. |
| RES2-8 | §8.2 | A production rerun may reuse research artifacts with complete lineage, safe formats, and signed manifests. |
| PERF2-1 | §7.8 | Residency follows read frequency: `old` on GPU, `eval_ema` on CPU; numbers in 08. |
| EMB2-3 | §5.3, §7.3, §9.2 | `proposed_action` is stored with a declared `intervention_rule` (`train_on_proposed_plus_cost`, `mask`, `executed_with_is`); out-of-support executed actions fall back to `mask` and are counted. |
| EMB2-5 | §4.2, §5.3, §3.1 | A `[B, K]` `discount` or `step_duration` field feeds GAE; NeMo RL's scalar `gae_gamma` makes this part of the gated P6a ask. |
| EMB2-7 | §5.3 | `final_observation` is stored sparsely at boundary indices with an index field, never dense. |
| EMB2-8 | §5.3, §9.2 | Recurrent state resets to the initial state after a boundary inside a BPTT chunk at rollout and training, with a conformance test. |
| GOV2-7 | §8.1, §9.1 | The research profile no longer omits signing: shared clusters require safe formats and signed manifests (06); allow-list changes go through reviewed merge requests. |
| RT2-4 (research) | §8.1, §8.2 | On shared clusters the research profile enforces the allow-list and code-execution block; outputs are `unpromotable` and enter production only by rerun or an approved import stage. |
| GOV2-2 (research) | §8.1, §8.2 | Research data loaders admit only data tagged for research use, and research weights cannot reach production without lineage checks or a signed import. |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| NRL3-1 | §3.3 | Role refit is part of hook 7 (role argument) or a D adapter swap before a normal refit; prototyped first. |
| NRL3-2 | §3.3 | Worker-resident EMA state is registered for save and restore, so it is in every cut. |
| NRL3-3 | §3.1 | Hook 8d exports consumed rows first; drop reasons from the drop sites follow later. |
| NRL3-4 | §3.1 | 8a splits into 8a-1 (emitters and `stop`) and 8a-2 (`pause_and_checkpoint` with `hold`/`continue`, after a prototype). |
| NRL3-5 | §3.1, §5.2 | The controller epoch is recorded once per cut manifest and ledger-segment header, not per row. |
| NRL3-6 | §3.3, §9.2 | On `grpo_train_sync`, `post_optimizer_step` runs synchronously before refit and `save_checkpoint`, with a test. |
| RES3-1 | §7.2, §8.3 | Research-profile local C-ext shim until hooks 7 and 10 land. |
| RES3-3 | §7.1 | Kernels take `(role_outputs, batch, cfg, counts)`. |
| RES3-4 | §7.4 | 1/c of training compute, with 08 §4.4's derivation. |
| RES3-5 | §7.5 | Mean centring scales by (G−1)/G; std normalization is biased with or without CRN. |
| DA3-2 | §3.1 | Hook 1 is marked path N only. |
| RES3-2 | — | Deferred to P10; see `open-issues.md`. |
| ARCH3-5, ARCH3-6 | §1 | Package splits of B and A are a P0 packaging decision; see `open-issues.md`. The §1 table stands until then. |

GOV3-1 and GOV3-5 (research-profile parts, §8.1) and ARCH3-8 (rule-2 wording, §1) are applied here only for this sub-design's text.

## Notes for the overview

Both revision-8 notes are resolved: overview §4 and §5 show A as pure Python with torch protocols in B and the hook list of §3.1 with hook 1 as in §3.2; the overview's G0 row records whether verl-omni recomputed old log-probs.
