# 08. Families, Execution Modes, and Performance

**Status:** The design set is at revision 24; this document was last changed in revision 12. A prototype (`gatedtrain` 0.4.0) implements revision 24. Frozen until G0 closes; deferred items in open-issues.md. Revision 9 applies R9-5's P1b throughput baseline (PERF3-1) and the PERF3-5 wording from the round-3 review (record not published).

**Parent:** `design.md` (the overview). Decisions D12 (fused vectorized mode is first-class) and D16 (per-stage, per-iteration, and per-checkpoint targets) are specified here. Revision 8 implements the execution and placement parts of R8-13 and all of R8-15 from the round-2 review (record not published).

**Primary owner of findings:** round 1: EM-7, PERF-2, PERF-5–8, PERF-10, EMB-1, EMB-4, EMB-6 (placement and hardware), EMB-7, EMB-8, and the numbers behind PERF-1 (the persistence policy itself is in 02). Round 2: PERF2-1–PERF2-6, EMB2-1, EMB2-5 (hybrid mode), EMB2-6, EMB2-7 (rendering placement).

**Sources read for this file:** flow_grpo (`config/grpo.py`, `config/base.py`, `scripts/train_sd3.py`), diffusionnft (`scripts/train_nft_sd3.py`), verl-omni (Wan2.2 and Qwen-Image example scripts, `verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`), NeMo RL `4f39fd8` (`nemo_rl/data_plane/README.md`, `nemo_rl/data_plane/interfaces.py`, `nemo_rl/algorithms/advantage_estimator.py`), NeMo AutoModel diffusion examples, FastVideo Wan configurations (`fastvideo/configs/models/dits/wan_s2v.py`), and NeMo Gym `README.md`. Every number that is not read from code is marked **(estimate)** and lists its assumptions.

## 1. Scope and conventions

This file covers how stages execute (§2), what each model family needs (§3), how many bytes and FLOPs a realistic training configuration moves (§4), where components may be placed (§5), what embodied training adds (§6), and how performance is measured and gated (§7, §8).

Conventions used in all estimates:

- **GPU:** H100 SXM 80 GB, NVLink 4 within an 8-GPU node, one 400 Gb/s InfiniBand NIC per GPU.
- **Effective throughput:** 450 TFLOPS bf16 for large DiT forwards (about 45% of dense peak); 350 GB/s achievable NVLink all-to-all bandwidth per GPU; 40 GB/s achievable IB bandwidth per GPU; 10–25 GB/s host-to-device bandwidth per GPU when all eight GPUs of a node copy at once. All are assumptions to be replaced by the P0–P1b measurements in §8.
- **S** is the number of sampling steps per rollout; **W** is the number of trained steps per row (the SDE window, or all steps when no window is used). Flow-GRPO stores W+1 latent states in fp32 per row (01 owns the field layout).
- **Forward-equivalents per trained step, c** (per CFG branch): forward 1; backward 2 for full fine-tuning or about 1 for LoRA (no weight-gradient GEMMs for frozen weights); plus 1 when activation recompute is on; plus 1 for a reference forward when β > 0; plus 1 when `old = recompute` (overview D5) or when an `old` role is evaluated by a separate forward (DiffusionNFT). Estimate; measured in P1b (§8 item 5).

## 2. Execution modes

### 2.1 The five modes

| Mode | Shape | When to use | Package |
|---|---|---|---|
| **M1. Synchronous colocated** | Rollout and training share GPUs and alternate; NeMo RL `grpo_train_sync` | Default for LLM GRPO and for all diffusion in P1b (§5.2) | C-ext, D |
| **M2. Asynchronous single-controller** | `SingleController` with rollout engines, staleness-bounded sampling, and the data plane | Reward or environment latency of seconds or more; video with VLM judges (P4) | C-ext, D |
| **M3. Environment as a service** | Simulators, CAD kernels, code sandboxes, or frozen planners in their own processes or containers behind the `Environment` contract; NeMo Gym is the template (its environment servers "manage sessions, verification, cleanup, and failures", `nemo-gym/README.md:53`) | Long-horizon episodes, heavy or licensed simulators, rendering GPUs, HIL benches, frozen VLM planners (§2.5) | C-ext, H |
| **M4. Fused in-process vectorized** | Policy, GPU-batched simulator, GAE, and PPO update in one process per GPU; rollout buffer is a set of GPU tensors; no data plane and no weight sync | Massively parallel continuous control (thousands of environments, small MLP or small transformer policies, ms-level control loops) | H |
| **M5. Wrapped external library** | An existing trainer (rsl_rl, rl_games, a vendor stack) runs unchanged as one workflow stage behind the run adapter | When M4 is slower than the reference library, or a team will not migrate | R (run adapter), A (contract emitter library) |

Package H depends on A and B, and optionally on C-ext; M4 needs only the contract emitter library from A and the out-of-process run adapter from R.

### 2.2 Fused in-process vectorized mode (package H)

The data-plane row model is the wrong unit for massively vectorized PPO (EMB-1). A representative locomotion configuration steps 4,096 environments for 24 steps per iteration, which is about 98,000 transitions of a few hundred bytes each; the per-row metadata and transfer cost of the data plane would exceed the payload, and the whole iteration takes well under a second of policy compute (estimate; representative of Isaac Lab–style tasks, to be confirmed with the embodied pilot user).

Design of M4:

1. **One process per GPU.** The simulator steps a batch of environments on the GPU; the policy reads observations from the same device memory.
2. **GPU-resident buffer.** Observations, actions, log-probs, values, rewards, `terminated`, `truncated`, the optional discount or step-duration field (§2.5), and `proposed_action` where a shield is active (01) are preallocated tensors of shape `[T, N_env, ...]`. `final_observation` is stored sparsely at boundary indices (01), not as a dense `[T, N_env, ...]` copy. GAE runs on these tensors on the GPU at the end of the rollout. Step semantics (01) are the same as on the data plane; only the storage differs.
3. **Light policy backend.** A plain-torch policy backend (MLP, small CNN or transformer, optional recurrent state) with `torch.compile` and CUDA graphs where the simulator allows. No Megatron, no FSDP, no Ray inside the loop. Flow-matching action heads reuse package B for step log-probs.
4. **Algorithm plug-ins reused.** PPO, normalization, and loss aggregation are the same plug-ins as elsewhere (01); M4 calls them with tensors.
5. **Run contract at iteration granularity.** The heartbeat progress counter is the iteration index; `pause_and_checkpoint` and `stop` are checked between iterations; cuts are written only at iteration boundaries by a separate writer process (06), and the out-of-process run adapter (R) signs and commits them. Under fenced behavior (02, 03) the run keeps computing for at most `fenced_compute_max_s` without committing cuts; `safety_critical` runs use `fenced_compute_max_s = 0`. A standalone fused run refuses `safety_critical` inputs and outputs; they are accepted only when the fused run is a stage of a validated P2 workflow (06).
6. **Provenance at iteration granularity.** Each iteration records one `ConsumptionRecord` (schema in A, 05) per environment batch, with a `chunk` naming the batch and `precision = step_range` (05 §11.2, §12.5): `weight_version` before the update, `env_version`, the environment-parameter version (§6.1), the normalizer version (§6.2), per-replica seed ranges, transition count, masked-transition count, and quarantined seeds. Individual transitions are not recorded.
7. **Multi-GPU replicas (EMB2-6).** Replicas are data-parallel, each owning its environments, with gradients all-reduced per minibatch. State that must agree across replicas is synchronized at iteration boundaries:
   - **Observation normalizer.** Each replica accumulates count, mean, and sum of squared deviations over its own transitions; at the boundary one all-reduce in fp64 combines them with the parallel (Chan) update, so every replica holds identical statistics and the normalizer version increments once globally.
   - **Curriculum.** Per-instance levels stay with the owning replica; level-promotion statistics (successes and attempts per level) are all-reduced, and promotion is a deterministic function of the reduced statistics, so every replica makes the same decision. Quarantine lists (§6.4) are reduced as a union.
   - **Seeds.** Replica r draws from a disjoint range `[base + r·M, base + (r+1)·M)`; the iteration's ledger entry lists each replica's range and transition counts.
   - **Check.** After each boundary the replicas compare digests of normalizer and curriculum aggregates; a mismatch fails the iteration. The reduced payload is KB to a few MB per iteration (estimate), negligible next to the gradient all-reduce.
   - **Cuts.** The writer stores the global normalizer and curriculum aggregates once, and per-replica instance state and RNG state as shards.

### 2.3 Guarantees per mode

| Guarantee | M1 sync colocated | M2 async single-controller | M3 env as service | M4 fused vectorized | M5 wrapped library |
|---|---|---|---|---|---|
| Provenance granularity (`precision`, 05) | `row` | `row`, with per-row `weight_version` and staleness | `row`, plus `env_version` and environment parameters | **`step_range`**, one record per environment batch and iteration (§2.2 item 6) | `stage`: inputs, config, seeds, outputs |
| Cut granularity | Optimizer-step boundary | Per 02: non-quiescing rollout snapshots, buffer persisted or dropped by the 02 policy (§4.3), samplers `restart_only` or `drain` (§7.4) | As M1/M2 plus environment state through `CheckpointParticipant` or episode restart | **Iteration boundary only** | The library's own checkpoints, registered as cuts by the adapter |
| In-flight work at a cut | None | Token rollouts in the recovery ledger; non-token rollouts per 02 and §7.4 | Episodes snapshotted if the environment supports it, otherwise reset and counted | Episodes reset on restore unless the simulator supports state save; counted as lost transitions | Whatever the library does; declared by the adapter |
| Weight sync | In-device reshard | `WeightSynchronizer` pushing `AlgorithmSpec.rollout_role` (01); post-refit digest only where the opt-in digest ask has landed (P4) | As M1 or M2 | None (same tensors) | None visible |
| Failure domain | Whole job | Trainer gang; rollout engines restartable | Per environment instance (§6.4) | **Shared:** a simulator CUDA fault kills the policy process; restart from the last cut | Whole stage |
| Data plane | Yes | Yes | Yes | No | No |
| Gates and lineage | Full | Full | Full | Full at stage level; sample-level queries resolve to iterations | Stage level only |

### 2.4 Wrapping an external library as a stage (M5)

A library such as rsl_rl or rl_games is wrapped by an adapter of a few hundred lines (estimate) that:

1. Writes the run-contract records through the contract emitter library (A) and the contract transport (A interface, 02): a heartbeat whose progress counter is the library's iteration counter, a cut manifest after each library checkpoint (the adapter hashes the files the library wrote under the attempt- and epoch-scoped path), and a completion manifest. The cut manifest is written only when the checkpoint is complete under the completeness rule the adapter declares per library version (the files that must exist and the file the library writes last); an incomplete directory gets no cut manifest, and the conformance kit tests this with a checkpoint interrupted mid-write. Signing, lease handling, and fencing stay in the run adapter (R).
2. Maps the two verbs: `stop` and `pause_and_checkpoint` are delivered through the transport and checked in the library's per-iteration callback or logger hook; if the library has no hook, the adapter can only stop the process, and `pause_and_checkpoint` is declared unsupported in the capability record.
3. Records inputs (environment version, task config, seeds, container digest) and outputs (policy checkpoint, normalizer statistics, exported model) as artifacts with manifests (05).

What the wrapped stage gets: lineage at stage level, artifact hashing and signing, statistical gates on its outputs (05), budgets and placement (03), and restart from its own last checkpoint. What it does not get: per-iteration provenance (unless the library logs it), consumption records, environment-instance failure handling (§6.4), or the M4 guarantees. P6a compares M4 against these libraries (§7.5); if M4 is slower, M5 remains the supported path.

### 2.5 Declared hybrid mode: frozen planner service plus fused controller (EMB2-5)

A frozen VLM planner cannot live inside the M4 process: it is a large model with its own serving stack and failure modes, and the M4 step loop runs at ms scale. Hybrid mode is declared in the stage spec (`execution: hybrid`) and combines:

- **Controller in M4.** The trained controller, simulator, GAE, and PPO run fused as in §2.2.
- **Planner as a co-located M3 batch service.** The frozen planner is served on GPUs of the same node (for example 1–2 GPUs per node serve the planner while the rest run M4 replicas; estimate). It is called at planner rate, every `k_p` controller steps (for example a 1–2 Hz planner over a 50 Hz controller gives `k_p` = 25–50; estimate), with one batched request per replica covering every environment due for a new plan. Observations reach it through shared memory or CUDA IPC.
- **Latency.** Synchronous calls add `t_planner / (k_p × t_step)` to the iteration time; the stage declares a budget (proposed ≤ 10%, estimate) and the P6a measurement reports it. A stage may instead declare a one-period lag (plan computed from the observation at the previous planner boundary); the lag is part of the environment semantics and is recorded in the stage config.
- **Provenance.** The planner's output (subgoal tokens or embedding) is part of the controller's observation and stored in the M4 buffer. The planner artifact is pinned as a stage input, and each iteration's consumption record carries a `used` edge to it (05, every model role).
- **Discounting.** When a decision step spans a variable duration (action chunks, variable frame skip, event-triggered control, or a policy acting at planner boundaries), the environment writes 01's `[B, K]` discount (or step-duration) field. H's GAE then uses `δ_t = r_t + γ^{Δt_t} (1 − terminated_t) V(s_{t+1}) − V(s_t)` and `A_t = δ_t + (γλ)^{Δt_t} (1 − done_t) A_{t+1}`, with `r_t` the discounted reward accumulated within the step. Without the field γ is constant. Through NeMo RL's GAE (M2/M3) the same input is part of the gated GAE ask (P6a, overview §5).
- **Failure domain.** A planner-service failure is an M3 failure: the service restarts, the controller waits up to a declared timeout and then pauses at the iteration boundary; the M4 process is not killed.
- **Not in scope.** Training the planner in this mode (joint training is a non-goal); planner fine-tuning is a separate stage.

Hybrid mode is a P6a extension, built only when the embodied pilot user needs a planner. Because the planner is a token VLM, an existing token serving engine can host it.

## 3. Family notes

### 3.1 Image diffusion and the P1b family choice (EM-7)

**Choice rule.** The P1b family is the one the diffusion pilot user trains (G0, overview §3), chosen from families with an external Flow-GRPO reference:

| Family | Flow-GRPO reference | SFT in AutoModel | Comment |
|---|---|---|---|
| SD3.5 Medium (2.5B) | flow_grpo `pickscore_sd3` (`config/grpo.py:205`) and verl-omni `examples/flowgrpo_trainer/sd35/` | No SD3.5 config in `nemo-automodel/examples/diffusion/finetune/` | Canonical parity case for package B; dated relative to what users train |
| Qwen-Image (20B) | verl-omni `examples/flowgrpo_trainer/qwen_image/` (LoRA and 64-GPU full fine-tune); flow_grpo `pickscore_qwenimage` (`config/grpo.py:692`) | `qwen_image_t2i_flow.yaml` | **Recommended default** if the pilot user has no other preference |
| FLUX (FLUX.1/2/3) | verl-omni DanceGRPO `examples/dancegrpo_trainer/flux1/` | `flux*_t2i_flow*.yaml` | Second family candidate (overview adoption metric) |

Recommendation: **Qwen-Image is the P1b product family** unless G0 names an SD3.5 user, because it has two independent references and AutoModel already trains it. **SD3.5 Medium remains the parity case for package B (P1a)** in every outcome, because flow_grpo is the reference implementation of the math. A FLUX family is the second-family onboarding test.

Family facts carried forward:

- **Training vs deployment sampler.** RL trains with an SDE sampler at few steps (flow_grpo: 10 training and 40 evaluation steps for SD3.5, `config/grpo.py:212-213`; Qwen-Image: 10 and 50, `:699-700`). Checkpoints record both samplers; gates run under the deployment sampler (05).
- **Real training configurations are small.** SD3.5 trains at 512 px and 10 steps with 24 images per prompt (`config/grpo.py:212,216,218`).
- **Trained steps.** `timestep_fraction = 0.99` (`config/grpo.py:226`) gives `int(10 × 0.99) = 9` trained steps (`scripts/train_sd3.py:345`). Window configurations train 2–3 steps: `sde_window_size = 2` in `[0, 5)` for Qwen-Image (`config/grpo.py:718`) and the verl-omni Qwen-Image and Wan2.2 scripts; 3 for the SD3.5 "fast" configurations (`config/grpo.py:189`).
- **Activation recompute.** flow_grpo turns it on for Qwen-Image (`config/grpo.py:722`) and off by default (`config/base.py:31`), so c differs by family (§4.4).
- **Ratio denominator.** flow_grpo uses the stored rollout log-prob (`scripts/train_sd3.py:891`); the reference forward runs the same transformer with the LoRA adapter disabled (`:882`). Overview D5 applies.
- **Rollout role.** DiffusionNFT samples with the EMA `old` adapter (`diffusionnft/scripts/train_nft_sd3.py:640`) and reads it again in every training forward (`:873-876`); the `old` blend happens once per iteration (`:1033-1038`) and `eval_ema` steps after each gradient step (`:1029`). `AlgorithmSpec.rollout_role` (01) selects `old` for weight sync; residency follows in §5.3.
- **Stored dtype.** verl-omni stores W+1 window latents in fp32 (`qwen_image_flow_grpo/vllm_omni_rollout_adapter.py:437-438, 459, 520`). Latents stay fp32 unless a family shows bf16 storage keeps the rollout-versus-train log-ratio within the P1b margin.

### 3.2 Video diffusion

- **Reference configurations.** flow_grpo Wan2.1 uses the 1.3B model at 33 frames, 240×416, 20 sampling steps, with the 14B path commented out (`config/grpo.py:29-42`). verl-omni Wan2.2 uses TI2V-5B at 704×1280, 8 frames, 10 steps, `sde_window_size = 2` in `[0, 5]`, with FSDP parameter and optimizer offload (`examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_v1.sh:58-59,72-83`). Neither reference trains 14B at 81 frames; that configuration is the P4 planning case and an extrapolation.
- **Architecture used below.** Wan 14B: 40 layers, hidden 5120, **40 attention heads** of dimension 128 (FastVideo `configs/models/dits/wan_s2v.py:62-69`, Wan2.2-S2V-14B; the Wan2.1-T2V-14B count is assumed equal because it shares the 5120 hidden size).
- **Clip cost (estimate).** Per forward at 81 frames, 480×832: 32,760 video tokens (21×30×52 after 1×2×2 patching); linear 2 × 14B × 32,760 ≈ 0.92 PFLOP; attention 4 × 32,760² × 5120 × 40 ≈ 0.88 PFLOP; total ≈ 1.8 PFLOP. At 450 TFLOPS effective, VAE decode and reward excluded:

| Sampler | Forwards | GPU·s per clip | Use |
|---|---|---|---|
| Training SDE, 20 steps with CFG | 40 | about 160 | P4 planning case (§4.3) |
| Deployment, 50 steps with CFG, 480×832 | 100 | about 400 | Gate evaluation; partner clips in composition (04) |
| Deployment, 50 steps with CFG, 720×1280 (75,600 tokens, ≈ 6.8 PFLOP per forward) | 100 | about 1,500 | High-resolution partner or gate |
| Same as row 2 at 150 TFLOPS effective (unfused attention or serving overhead) | 100 | about 1,200 | Reproduces the 1,200 GPU·s figure in 04 |

  The 04 planning example should state which row it uses; 400 GPU·s at 480p and 450 TFLOPS is this file's planning value.
- **Context parallelism.** AutoModel uses Ulysses attention through diffusers with `cp_size` in the config (`nemo-automodel/examples/diffusion/finetune/wan2_1_t2v_flow.yaml:74-78`); valid CP sizes and placement in §5.1.
- **Rewards.** VLM judges and video OCR read decoded frames; frames cross process boundaries only as compressed video (§4.1).

### 3.3 VLA and autonomous driving

- Tokenized action heads reduce to the token path; flow-matching action heads use package B and the `ACTION` step type (01).
- Flow-head rollout needs a sampler that runs the flow head and records step log-probs. vLLM has no flow-matching action head, so P6b names a rollout engine before it starts (a package-B-based torch sampler service in H, or the pilot user's engine) (§7.5).
- Closed-loop driving simulators with camera sensors run as M3 services on rendering GPUs (§5.6); episodes are long, so M2 is required.
- Open-loop log replay with learned rewards can run in M1.

### 3.4 Flight control and locomotion

- `ACTION` steps with squashed-Gaussian log-probs and truncation bootstrap (§6.3).
- Plug-ins: policy, vectorized simulator with domain randomization (§6.1), M4 rollout, rewards, SITL stage, HIL gate with human sign-off (06), embedded exporter with equivalence tests (§6.5).
- Small-policy simulator RL is bound by simulator throughput; if M4 is slower than the reference library (§7.5), the library is wrapped as an M5 stage.

### 3.5 CAD and speech

- CAD-as-code uses the token path plus a sandboxed CAD-kernel environment (M3, after the P8 sandbox). Rewards run cheap rule checks first (executes, valid solid), then geometric match and manufacturability rules.
- Speech-token TTS uses the token path; rewards decode audio and run ASR, speaker similarity, and quality predictors. Audio crosses process boundaries compressed (Opus or FLAC).

### 3.6 Deployable components per family

`model_manifest_hash` (05 §11) covers exactly these components; resume-only state, such as optimizer and data-loader state, is covered only by the cut manifest (02 §2.4).

| Family | Deployable components |
|---|---|
| LLM and VLM | Weights, model config, tokenizer or processor, generation config |
| Image and video diffusion | Transformer or UNet weights, or a LoRA adapter with `base_ref`; scheduler config; text-encoder references; VAE reference |
| Speech | Weights, feature extractor, tokenizer |
| Control and VLA | Policy weights, observation-normalizer statistics (§6.2), `env_version` |

## 4. Payload and performance model

### 4.1 Per-row bytes

Latent sizes are exact from the shapes; encoder lengths are from the cited configs; compressed sizes are estimates.

| Item | SD3.5 Medium, 512 px | Qwen-Image, 512 px | Wan 14B, 81 f, 480×832 |
|---|---|---|---|
| Latent shape (one state) | 16×64×64 | 16×64×64 | 16×21×60×104 |
| One state, fp32 | 0.26 MB | 0.26 MB | 8.39 MB |
| Full trajectory, S+1 states | 2.9 MB (S=10) | 2.9 MB (S=10) | 176 MB (S=20); 428 MB (S=50) |
| Window W=2 (3 states) | 0.79 MB | 0.79 MB | 25.2 MB |
| Window W=4 (5 states) | 1.3 MB | 1.3 MB | 42 MB |
| Prompt embedding, bf16 | 1.68 MB (77 CLIP + 128 T5 tokens × 4096; flow_grpo `max_sequence_length=128`, `scripts/train_sd3.py:625`); 2.73 MB at the diffusers default of 333 tokens | ≤ 1.83 MB (≤ 256 tokens × 3584; verl-omni `max_sequence_length=256`) | 4.19 MB (512 umT5 tokens × 4096; estimate of the padded length) |
| Log-probs, timesteps, advantages | < 1 KB | < 1 KB | < 1 KB |
| Decoded pixels, raw uint8 | 0.79 MB | 0.79 MB | 97 MB |
| Decoded pixels, compressed (estimate) | about 0.1 MB (JPEG q95) | about 0.1 MB | 1.3–3.1 MB (mp4 at 2–5 Mbit/s for about 5 s) |

Rules that follow:

1. **Prompt embeddings are stored once per group or by cache key, never per row.** flow_grpo stores `prompt_embeds` in every sample (`scripts/train_sd3.py:675`); with 24 images per prompt that duplicates 1.68 MB 24 times. Rows carry a key (encoder digest, prompt hash, sequence length) into an embedding cache (§5.4), tracked in lineage (05).
2. **Decoded pixels for rewards are compressed** (JPEG or PNG for images, mp4 for video) whenever they leave the producing process. Raw video at 97 MB per row is about 30–75 times the mp4 size. Rewards that need lossless input run colocated with the decoder.
3. **Only window states are stored.** Rollout-time step selection (01) stores W+1 states; full trajectories are stored only by algorithms that sample timesteps at train time (DanceGRPO-style, 01), and they declare it.

### 4.2 Per-iteration volumes

| Configuration | Rows per iteration | Latents | Embeddings per row (wrong) | Embeddings per group (right) |
|---|---|---|---|---|
| flow_grpo `pickscore_sd3`: 48 prompts × 24 images, S=10, W=9 | 1,152 | 3.0 GB (10 states) | 1.9 GB | 81 MB |
| Same, W=2 | 1,152 | 0.91 GB | 1.9 GB | 81 MB |
| Wan 14B, 32 prompts × 8, S=20, W=2 (estimate of a realistic batch) | 256 | 6.4 GB | 1.1 GB | 0.13 GB |

For SD3.5 at realistic windows the embeddings stored per row are **larger than the latents** (PERF-2).

### 4.3 Buffer sizes and the numbers behind the persistence threshold (PERF-1, PERF2-6)

02 decides whether buffered non-token rows are persisted in a cut or dropped and redispatched. This applies to M2 (P4); the P1b colocated mode takes full cuts only when no rollout is running (02). The inputs 02 needs, all estimates:

| Quantity | SD3.5 Medium, W=2 | Wan 14B, W=2 |
|---|---|---|
| Bytes per buffered row (latents plus amortized embedding) | 0.86 MB | 25.7 MB |
| Buffered rows (one iteration plus one stale iteration in M2) | 1,152–2,304 | **256–512** |
| Buffer size | 1.0–2.0 GB | **6.6–13.2 GB** |
| Regeneration cost per row (§3.1, §3.2 assumptions) | about 0.3 GPU·s (2.5B, about 1,230 tokens, 20 forwards) | about 160 GPU·s |
| Cost of dropping the buffer at a cut | 350–700 GPU·s | 41,000–82,000 GPU·s (11–23 GPU·h) |
| Wall time of that loss on 32 GPUs | 11–22 s | 21–43 min |
| Write plus read of the buffer at 8 GB/s aggregate (2 GB/s per node, 4 nodes) | 0.25–0.5 s | 1.7–3.3 s |
| Persist cost on 32 GPUs | 8–16 GPU·s | 53–106 GPU·s |

Threshold supplied to 02 (estimate formula; 02 owns the policy):

```
persist  if  (B / BW_write + B / BW_read) × N_gpu  <  k × N_rows × t_regen
```

with B the buffer bytes, BW the aggregate storage bandwidth available to the cut, N_gpu the GPUs idled by the cut, t_regen the rollout GPU·s per row, and k a safety factor (proposed 0.5). **Persistence wins for both families.** SD3.5: 8–16 GPU·s against a right side of 175–350 GPU·s (k × drop), about a 20× margin. Video: 53–106 GPU·s against 20,000–41,000 GPU·s, about 400×. Dropping is acceptable only as a temporary state before the persisted format exists, with the loss reported per cut.

### 4.4 Train-to-rollout FLOP ratio by window

Ratio of training to rollout transformer FLOPs, R = W × c / S, with c from §1. CFG cancels. VAE decode and reward cost are excluded and add to the rollout side.

| Configuration | S | W | c (terms) | R | Interpretation |
|---|---|---|---|---|---|
| flow_grpo `pickscore_sd3` (LoRA, no activation recompute, β = 0.01; `config/grpo.py:227`, `config/base.py:28,31`) | 10 | 9 | 3 (1 + 1 + ref 1) | 2.7 | Training dominates |
| Same with `old = recompute` | 10 | 9 | 4 | 3.6 | +33% training |
| SD3.5 fast, no CFG (LoRA, β = 0, W = 3) | 10 | 3 | 2 (1 + 1) | 0.6 | Rollout dominates |
| Qwen-Image `pickscore_qwenimage` (LoRA, activation recompute, β = 0; `config/grpo.py:713-722`) | 10 | 2 | 3 (1 + 1 + recompute 1) | 0.6 | Rollout dominates |
| Wan 14B full fine-tune, activation recompute, β > 0 | 20 | 2 | 5 (1 + 2 + 1 + ref 1) | 0.5 | Rollout dominates |
| Same | 20 | 4 | 5 | 1.0 | Balanced |

**Recompute overhead, one derivation.** `old = recompute` adds one forward per trained step, so it adds **1/c of training compute**: 33% at c = 3, 25% at c = 4, 20% at c = 5. As a share of the iteration's transformer compute it adds (R/c)/(1 + R): 24% for `pickscore_sd3`, 12.5% for Qwen-Image, 6.7% for Wan at W = 2. 01 §7.4 quotes this derivation (revision 9); planning documents quote 1/c with the c they assume.

Consequences: with windows of 2–3, about 60–70% of transformer FLOPs are rollout (1/(1+R) for R = 0.5–0.6), and about half at W = 4 for full fine-tuning; rollout efficiency (batching, CUDA graphs, attention kernels) is the main lever, and colocated time-sharing (§5.2) wastes little. R is logged per iteration (§8).

### 4.5 Getting rows onto context-parallel ranks

NeMo RL today fetches through one leader per data-parallel replica group: "replica group = all TP×CP×PP ranks of one DP rank; leader = its rank 0", the leader broadcasts "one field at a time, each field staged whole on GPU (CPU → GPU → NCCL)", and every rank holds the "full DP batch on CPU" before keeping its CP slice (`nemo_rl/data_plane/README.md:151-162`). For video at CP = 8, each of 8 ranks holds the full batch in host memory and the leader stages each whole field on its GPU.

Example (estimate): a DP micro-batch of 16 Wan rows at W=2 is 403 MB; with CP = 8 the host copies total 3.2 GB per node per fetch, versus 403 MB if each rank received only its slice.

Requirement for package D (path N): **CP-aware sharded gets.** Latent fields declare a token axis; each CP rank fetches its own token range (or the leader broadcasts only the slice each rank needs). Non-sharded fields keep the leader broadcast. This is an adapter in D over the existing data-plane client; if it cannot be expressed without changing NeMo RL, it is the gated CP slice-fetch ask (P4, overview §5).

## 5. Placement constraints

### 5.1 Video context parallelism stays within an NVLink domain (PERF-5)

Ulysses all-to-all volume for Wan 14B at 81 frames (estimate): one activation tensor is 32,760 tokens × 5120 × 2 B = **335 MB per sample per layer**; each attention layer exchanges Q, K, V, and the output, about 4 × 335 MB × (CP − 1)/CP across the group. Over 40 layers at CP = 8 that is about 47 GB per sample per forward across the group, or 5.9 GB sent per GPU.

| Link | Time per forward per GPU (estimate) | Compute per forward per GPU at CP = 8 | Exposed overhead |
|---|---|---|---|
| NVLink (350 GB/s) | about 17 ms | about 0.5 s (1.8 PFLOP / 8 at 450 TFLOPS) | about 3% |
| IB (40 GB/s) | about 150 ms | about 0.5 s | about 30% (all-to-all sits between projection and attention and is hard to overlap) |

Rules:

1. The CP group of a video model is placed inside one NVLink domain (one HGX node, or one NVL72 rack domain); data parallelism spans nodes. The placement broker (03) rejects a plan whose CP group spans a fabric boundary.
2. **Ulysses CP must divide the attention head count** (heads are split across ranks). For Wan 14B with 40 heads the valid sizes are 1, 2, 4, 5, 8, 10, 20, 40; inside an 8-GPU node that leaves 1, 2, 4, 8. **CP = 16 is invalid** (40/16 is not an integer, and it would also span nodes). Larger groups need an NVL72 domain (10, 20, 40) or ring attention. The sequence length must also divide by CP (32,760 / 8 = 4,095). The family plug-in declares `num_heads` and validation enforces both.

### 5.2 Diffusion defaults to colocated rollout and training (PERF-6)

- **Default M1 colocated** with parameter and optimizer offload during rollout, as verl-omni does for Wan2.2 and Qwen-Image (`run_wan22_5b_t2v_hpsv3_v1.sh:58-59`; `run_qwen_image_ocr_lora.sh:45-46`). Rollout and training use the same modules and kernels, which also satisfies the D5 condition for using rollout log-probs.
- **Refit is not a bottleneck.** LoRA adapters are tens to hundreds of MB (Qwen-Image rank 64, `run_qwen_image_ocr_lora.sh:37`, a few hundred MB in bf16, estimate). A full Wan 14B in bf16 is about 28 GB; in disaggregated M2 it moves in about 1–2 s over IB (estimate: sharded across a node's 8 NICs it is under 0.1 s at line rate; 1–2 s allows for fan-out and staging). Colocated refit is an in-device reshard. Against a Wan iteration of about 32 min (estimate: 256 rows × 160 GPU·s on 32 GPUs is 21 min of rollout, plus about 11 min of training at R = 0.5), refit is about 0.1%.
- **Disaggregate only when** reward or judge latency cannot be hidden (M2 with judges in P4), or rollout and training need different parallelism that offload cannot reconcile. Then `old = recompute` or a measured equivalence applies (D5), and the synchronizer pushes `rollout_role`.

### 5.3 Role residency by read frequency (PERF-7, PERF2-1)

Each role (01) declares `residency: gpu | cpu | cpu_prefetch`. The default follows read frequency: a role read at least once per trained step (policy, `old`, `lagged`, reference with β > 0) is `gpu`; a role read once per iteration or less (`eval_ema`, read only for evaluation and export) is `cpu`. `cpu_prefetch` (layer-wise host-to-device prefetch overlapped with compute) is allowed for a per-step role only after the measured overlap check (§8 item 12).

Memory for Wan 14B full fine-tuning under DiffusionNFT, per replica sharded over 8 GPUs (estimate: bf16 parameters, fp32 master weights, fp32 Adam moments, bf16 gradients, fp32 EMA copies):

| Role | Bytes per replica | Per GPU | Read frequency | Default residency |
|---|---|---|---|---|
| Policy: parameters, master weights, Adam moments, gradients | 28 + 56 + 112 + 28 = 224 GB | 28 GB | Every forward | gpu (offloaded during rollout, §5.2) |
| Reference (β > 0) | 28 GB bf16 (0 for LoRA: adapter disabled) | 3.5 GB | Every trained step | gpu |
| `old` = EMA (`train_nft_sd3.py:640, 873-876`) | 56 GB fp32 | 7 GB | Every rollout forward and every trained step; written once per iteration | **gpu** |
| `eval_ema` | 56 GB fp32 | 7 GB | Evaluation and export; written after each gradient step | **cpu** |
| Activations at 32,760 tokens, CP = 8, with recompute | remainder | about 20–30 GB | — | gpu |

All roles on the GPU: 28 + 3.5 + 7 + 7 = **45.5 GB** per GPU of fixed state before activations (revision 7 said 49 GB in error). With `eval_ema` on the CPU: 38.5 GB, leaving 41.5 GB for activations and buffers. The `eval_ema` update goes through the post-optimizer callback; each update moves the GPU's 3.5 GB bf16 shard to the host and updates 7 GB of fp32 there, about 0.3–1 s (estimate, PCIe plus host-memory bound), run on a side stream and finished before the next gradient step.

**Why `old` is not on the CPU (estimate).** A CPU-resident `old` sends the GPU's 7 GB shard over PCIe for each `old` forward: 0.28–0.7 s at 10–25 GB/s. One trained step at CP = 8 with one sample per micro-batch is about 5 forward-equivalents (old forward, policy forward, activation recompute, backward 2) × 0.5 s = 2.5 s. Unoverlapped, that is **11–28% overhead**. Layer-wise prefetch needs 175 MB per layer in the 12.5 ms the `old` forward spends on a layer, about 14 GB/s, which is at the edge of the assumed bandwidth; it is therefore a measured fallback, not the default. A bf16 read copy of `old` on the GPU (3.5 GB, refreshed once per iteration from an fp32 accumulator on the CPU) is a P5 option: the `old` forward already runs under autocast (`train_nft_sd3.py:871-876`), so matmuls see bf16 weights, but it ships only after the frozen-weight conformance check (01) passes. Residency changes are recorded in the run config and must not change results.

### 5.4 Frozen text encoders outside the trainer

umT5-XXL (about 5.7B parameters, about 11 GB bf16, estimate), T5-XXL, CLIP, and Qwen2.5-VL text encoders are frozen. They run in a separate precompute stage or encoder service, write an embedding cache keyed by (encoder digest, prompt hash, sequence length), and are never loaded into trainer workers. flow_grpo loads all three SD3 text encoders onto every training GPU (`scripts/train_sd3.py:411-413`); this design does not.

### 5.5 Data-plane backend limits (PERF-8)

| Backend | Mechanism (`nemo_rl/data_plane/README.md`) | Limit relevant here | Use |
|---|---|---|---|
| `simple` | ZMQ; data held in `SimpleStorageUnit` Python memory (README:870, 933-934) | Every put and get is served by Python threads; GB-scale rows live on the Python heap | LLM and SD3.5-scale image rows (≤ about 3 GB per iteration) |
| `mooncake_cpu` | One-sided RDMA into registered host segments; CPU staging slots of 512 MiB (README:459-463, 872-878) | Objects larger than a staging slot register per transfer | Video rows, judge payloads |
| `mooncake_cpu` with `use_gdr` | GPU staging buffer, 1024 MiB per CUDA client, serialized (README:880-887) | In the CPU-producer, GDR-receiver flow used by `SingleController`, **each tensor must fit in 1 GiB** | Allowed only when every stored tensor is ≤ 1 GiB |

Rules: rows are stored per row, not as one batched tensor (a batch of 64 Wan rows at W=2 is 1.6 GB and would break the GDR limit); full 50-step video trajectories (428 MB) fit but leave little margin; raw decoded video is never stored. The backend choice per family is a P1b/P4 measurement (§8).

### 5.6 Video buffer persistence on separate storage units (PERF2-5)

Video rows are held in data-plane storage units placed apart from the trainer gang and from token partitions (02 `storage_unit_placement`, option (b)): a small dedicated node group, or nodes whose loss already ends the run. At a cut each storage unit writes its own shard in parallel to its own storage target, so the 6.6–13.2 GB buffer write does not share NICs or file-system targets with the trainer's checkpoint write (§7.3).

While a data-plane save runs, callers must serialize saves and block clears (`nemo_rl/data_plane/interfaces.py:752-754`), so producers wait. This **barrier hold time** is measured in P4 (p50 and p99 against buffer size, per storage unit and per cut) and reported in `gpu_seconds_lost` (02); the persist cost in §4.3 assumes it equals the write time, which P4 confirms or replaces.

### 5.7 Simulators and heterogeneous hardware (EMB-6 placement, EMB2-7)

| Situation | Placement | Reason |
|---|---|---|
| Control loop at ms scale (50–1,000 Hz policy queries), GPU-batched physics | M4: simulator and policy in one process on the same GPU | Any network hop per step dominates the step time |
| Batched simulator service, no cameras, step times of tens of ms or more | M3 on the same node as the rollout engine; observations through shared memory or CUDA IPC | Keeps per-step latency low without fusing failure domains |
| Camera-heavy driving or manipulation simulators | Rendering GPUs (RT-core parts such as L40S-class); see the rule below | H100 lacks RT cores; GPU types are declared per pool in the cluster profile (03) |
| Frozen VLM planner (§2.5) | M3 batch service on the same node as the M4 replicas | Planner-rate calls; separate failure domain |
| Hardware-in-the-loop benches | Reservable exclusive resource with a calendar; only HIL gate stages use it | Physical benches are scarce and need human sign-off (06) |

**Rendering placement rule (resolves the conflict between "M3 on the same node as the rollout engine" and "rendering on L40S nodes").** In order of preference:

1. **Policy inference on the rendering nodes.** The rollout policy runs next to the renderer on the rendering nodes; only finished trajectories cross to the H100 training pool as M2 rows, and weights reach the rendering nodes by ordinary refit. Default when the policy fits in rendering-GPU memory and its inference throughput keeps up.
2. **Mixed nodes.** Nodes that carry both rendering GPUs and training-class GPUs, where the cluster has them; the M3 same-node rule then holds as written.
3. **Cross-node with a declared latency budget.** Renderer on rendering nodes, policy inference on H100 nodes. The service declares `colocate: same_fabric(budget_ms)`; the broker (03) admits the plan only if the measured p99 of encode + transfer + decode for one step's batch is within the budget (proposed ≤ 20% of the per-step render time, estimate), and the sampler keeps at least two environment batches in flight per renderer so inference of one overlaps rendering of the next. Example (estimate): 64 environments × 6 cameras × 224 × 224 × 3 uint8 is 58 MB raw per step, about 4.6 ms at 100 Gb/s or under 1 ms after JPEG, against render times of tens of ms.

Observation payloads are sized per family and encoded on the rendering GPU when they cross nodes. Rendering placement is a P6b concern; the measurement is §8 item 15.

## 6. Embodied specifics

### 6.1 Domain randomization and curriculum state (EMB-4)

- `env_version` (simulator code, assets, task definition) is pinned per run (02).
- **Environment parameters** (randomization ranges, curriculum level, terrain difficulty, per-instance sampled values) are **versioned per-row (or per-iteration in M4) data**, not run pins. Each row records `env_params_version` and, for sampled values, a seed or a compact parameter vector (01 owns the field).
- A change of environment parameters **does not start a reward epoch**: the reward function is unchanged; only the state distribution moves.
- Curriculum state is part of every cut; in multi-GPU M4 it is synchronized at iteration boundaries (§2.2 item 7).
- Gates evaluate on a fixed evaluation distribution declared in the gate's registration (`eval_distribution`: item set, `env_version`, `env_params_version`; 05 §3.7), independent of the training curriculum.
- **Without a dataset.** Environment parameters are materialized as an ordinary `DatasetSnapshot`, one shard per `env_params_version`, with seed ranges as rows where applicable; consumption records anchor to it, so rollups and impact queries work unchanged (05 §12.1).

### 6.2 Observation-normalizer statistics

Running statistics (count, mean, sum of squared deviations) are policy state:

- included in every cut and restored with the weights;
- versioned; each row (M2/M3) or iteration (M4) records the normalizer version used at act time;
- updated only at iteration boundaries in M4, all-reduced across replicas (§2.2 item 7), so all transitions of an iteration share one version;
- frozen and shipped with every exported policy (folded into the first layer or exported as constants); export-equivalence tests (§6.5) include them.

### 6.3 Final observations, truncation bootstrap, and discount

Vectorized simulators reset an environment automatically when it ends, so the observation returned after a terminal step is the first observation of the next episode. The terminal observation is kept as `final_observation`, stored sparsely at boundary indices (01).

- **Terminated:** bootstrap value 0.
- **Truncated (time limit, horizon cut):** bootstrap value V(`final_observation`).
- **Who computes it:** in M4 the critic computes V(`final_observation`) in the same batched forward as the next step's values, inside package H. In M2/M3 the trainer computes it at advantage time from the stored `final_observation`, because rollout engines may not hold the critic.
- **Variable step duration:** the discount field of §2.5 scales γ and γλ per step.

NeMo RL's GAE assumes a **zero terminal bootstrap**: the fast path relies on it ("With zero terminal bootstrap, the TD value terms telescope", `nemo_rl/algorithms/advantage_estimator.py:615`), and the general path initializes `next_values` to zeros (`:641-643`). Truncated episodes are therefore treated as terminated, which biases value targets for time-limited tasks. M4 uses package H's GAE, which accepts a per-row bootstrap value and a per-step discount. Embodied PPO through NeMo RL in M2/M3 needs both as GAE inputs: the gated GAE ask (P6a, overview §5).

### 6.4 Simulator failures (EMB-7)

Failures are handled per environment instance, not per run:

| Failure | Detection | Handling |
|---|---|---|
| Exception or NaN/Inf in state or observation | Per-step check on the batched state tensors | Reset the instance; mask its transitions since the last valid step; treat the partial episode as truncated with a bootstrap if the last state is valid, otherwise drop it |
| Physics explosion | Velocity, penetration depth, or energy jump beyond declared thresholds | As above |
| Bad seed or parameter set | The same seed or parameter vector fails k times (proposed k = 3) | Quarantine it (union across replicas, §2.2 item 7); record it in the iteration's ledger entry |
| Determinism drift | Periodic replay of a fixed seed and action sequence against a stored reference trajectory | Alert; fail the stage if drift exceeds the tolerance declared by the simulator plug-in |
| Resource leak | RSS or GPU memory growth per episode | Recycle the simulator process every N episodes or above a memory threshold (M3); restart from the last cut in M4 |
| High mask rate | Masked-transition share per iteration | Fail the iteration above a declared threshold (proposed 5%) so silent data loss cannot continue |

In M4 the simulator shares the process and CUDA context with the policy (§2.3): a CUDA fault takes both down, and recovery is restart from the last iteration-boundary cut. The M4 cut interval is registered like any other (§7.3).

### 6.5 Export equivalence (EMB-8)

Every exported policy (ONNX, TensorRT, or embedded C) passes, on the target hardware:

1. **Action equivalence:** on a recorded set of observations covering the evaluation distribution, the maximum absolute action difference between the training policy and the export is within a declared tolerance per precision (fp32, fp16, int8), with normalizer statistics applied.
2. **Closed-loop equivalence:** episode return of the export in simulation within a registered margin of the training policy (TOST, 05).
3. **Latency:** p99 inference latency on the target device below the control-period budget, measured with the production runtime.

These are gate types in 05; HIL sign-off (06) follows them.

## 7. Numeric targets (PERF-10)

### 7.1 Per-stage overhead (P0)

- Metric: time of each instrumented stage (fetch, forward, loss, optimizer step, refit, put) with the framework hooks versus without.
- Procedure: interleaved A/B (alternate hook-on and hook-off iterations, or paired runs alternating on the same nodes) for at least 50 pairs; report the median difference and its 95% confidence interval per stage.
- Target: ≤ 2% per stage, and zero new bytes per token on token partitions (overview P0). An end-to-end ratio is reported but not gated, because run-to-run noise exceeds 2%.

### 7.2 End-to-end throughput and context-parallel efficiency (P1b, P4)

- **Samples/s/GPU end to end,** counting every GPU the run holds: rollout, VAE decode, reward and judge services, refit, and training.
- **P1b criterion (PERF2-2, PERF3-1):** end-to-end samples/s/GPU **≥ 0.9× verl-omni**, the throughput baseline for the chosen family, on the same hardware and the same configuration (model, resolution, S, W, group size, batch, LoRA or full, β, old-policy mode as recorded in G0, GPU count). The quality reference may remain flow_grpo. At least three interleaved run pairs (framework and verl-omni alternating on the same nodes), each run measured over at least 20 iterations after warm-up; the ratio and its 95% interval are computed across run pairs, not across iterations of one run, and the criterion holds when the interval's lower bound is ≥ 0.9. Path N only; under path V verl-omni is the implementation.
- **Intra-node CP efficiency (P4):** per-GPU throughput at CP = k within one NVLink domain divided by throughput at CP = 1 (at a sequence length that fits CP = 1, or extrapolated from per-layer timings), for valid k only (§5.1). Target ≥ 80%.
- Four-node scaling is reported, not gated: with CP inside nodes it is data parallelism and is expected to be near-linear.

### 7.3 Checkpoint interval and goodput loss (P1b, P4; PERF2-3)

**The interval is registered before the run,** in the stage spec, using Young/Daly: τ = √(2 · C · MTBF), where C is the measured blocking cost of one cut (stall seconds) and MTBF is the expected time between job interruptions for the run's node count, taken from the cluster profile and the stage's own history (07). A registered interval more than 2× away from τ needs a recorded reason (for example a cap from 02's checkpoint-age rule).

**Two metrics are reported together:**

- **Per-checkpoint loss** = (stall time × GPUs idled) + regenerated rollout GPU-seconds (dropped rows × t_regen) + park and drain GPU-seconds (§7.4), divided by GPU-seconds between checkpoints. Target ≤ 2% at the registered interval (overview P4).
- **Expected GPU-seconds lost per failure** = (τ/2 + restart time) × GPUs held + in-flight sampler loss (§7.4). With it, the expected total overhead C/τ + (τ/2 + restart time)/MTBF is reported per run. Lengthening the interval lowers the first metric but raises this one, so the interval cannot be gamed.

Example (estimate): Wan 14B full fine-tune with the two DiffusionNFT EMA roles, 308 GB of weights, optimizer state, and roles (196 GB without them) at 8 GB/s aggregate write gives **C ≈ 39 s** (≈ 25 s without roles) on 32 GPUs, plus 1.7–3.3 s for the persisted buffer (§4.3). In the DiffusionNFT reference `old` is a LoRA adapter (`train_nft_sd3.py:640`), so full-size EMA roles on a 14B full fine-tune are an extrapolation, not a reference configuration. With an assumed job MTBF of 24 h and restart time of 10 min:

| Interval | C/τ (per-checkpoint) | Rework per failure | Expected total overhead |
|---|---|---|---|
| Young/Daly, τ ≈ 43 min | 1.5% | (22 min + 10 min) × 32 GPUs ≈ 16.9 GPU·h | 1.5% + 2.2% ≈ 3.7% |
| Hourly | 1.1% | (30 min + 10 min) × 32 GPUs ≈ 21 GPU·h | 1.1% + 2.8% ≈ 3.9% |

At an assumed MTBF of 6 h, τ ≈ 22 min and C/τ ≈ 3.0%, above the per-checkpoint target: the run must then lower C (asynchronous save, so only the device-to-host copy blocks) rather than lengthen the interval.

### 7.4 In-flight sampler work at snapshots: drain versus restart-only (PERF2-4)

02 gives sampler participants two snapshot modes: `restart_only` (in-flight items are not saved and are redispatched after a restore) and `drain` (stop admitting, let in-flight items finish, then snapshot). Costs for M2 video (estimate: Wan 14B, 20 steps, CP = 8, 160 GPU·s per clip, so about 20 s wall per clip per CP group; progress of in-flight items uniform; n CP groups on G rollout GPUs):

- `restart_only`: nothing at snapshot time beyond the park; on a restore it loses about **G × t_wall / 2** of partial work. With 8 groups (64 GPUs): about 640 GPU·s per restore.
- `drain`: on every snapshot, groups that finish early wait for the slowest, about **G × t_wall × (n/(n+1) − 1/2)** idle; nothing is lost on restore. With 8 groups: about 500 GPU·s and about 18 s of wall time per snapshot.

Rule: periodic snapshots use `restart_only`, because its cost is paid only on the rare restore. `pause_and_checkpoint` before a planned stop, preemption, or upgrade uses `drain` (a restore is certain, and drain is cheaper in expectation), as do participants whose in-flight items cannot be restarted. A `drain` deadline caps the wait; items still running at the deadline are restarted. Both costs enter `gpu_seconds_lost` (02).

### 7.5 P6a and P6b criteria (EMB-8, EMB2-1)

P6 is split. **P6a** is package H with fused continuous control: it depends on P0, and on P2 when `safety_critical` data is used (§2.2 item 5). An embodied pilot user may take P6a in place of P1b in the MVP. **P6b** is VLA flow heads: it depends on P3 (supervised recovery and heterogeneous placement, §5.7), P4 (asynchronous samplers with `CheckpointParticipant`, CP slice fetch), and a named rollout engine (§3.3).

| Phase | Criterion | Procedure |
|---|---|---|
| P6a | Reward-curve equivalence | PPO in M4 versus rsl_rl or rl_games on the same task, environment count, and hyperparameters; TOST on final return and on area under the return curve, five seeds or more if the power plan requires (05), margins registered in advance; repeated at 1 GPU and at the pilot user's GPU count |
| P6a | Environment throughput | Env-steps/s in M4 ≥ 80% of the reference library on the same GPU, including policy inference and update |
| P6a | Export equivalence | §6.5 on the target hardware |
| P6a | Simulator failure handling | Injected NaN, explosion, and leak faults are handled per instance (§6.4) without a run restart in M3, and with restart from the last cut in M4 |
| P6a | Bootstrap and replica consistency | A time-limit task shows truncation bootstrapped from V(`final_observation`); normalizer and curriculum digests agree across replicas at every boundary |
| P6a (if hybrid) | Planner overhead | Planner latency share of iteration time within the declared budget (§2.5) |
| P6b | Closed-loop success rate | TOST on closed-loop success rate against the pilot user's reference implementation on a fixed scenario set, margins registered in advance, intervals clustered by scenario (05) |
| P6b | Flow-head parity | B's `ACTION`-step log-prob parity tests pass for the chosen flow head |
| P6b | Export and placement | §6.5 on target hardware; rendering placement meets its latency budget (§5.7) |

## 8. Must measure in P0 to P6b

Each item is a recorded measurement that replaces an estimate above; the named phase does not exit until it is recorded.

| # | Measurement | Phase | Replaces |
|---|---|---|---|
| 1 | TransferQueue put/get GB/s for 0.3–50 MB rows on `simple` and `mooncake_cpu` (with and without GDR), single node and cross node | P0 | §5.5 backend choice |
| 2 | Host memory and time of the DP-leader broadcast under TP and CP (CP = 1, 2, 4, 8) | P0 | §4.5 |
| 3 | Per-stage overhead with interleaved A/B (§7.1) | P0 | P0 exit |
| 4 | Recompute versus rollout log-ratio and its drift over training (per timestep, D5) | P1b | D5 choice per family |
| 5 | Forward-equivalents c per trained step, with and without activation recompute, and `old_forward` cost fused and unfused (01) | P1b | §1, §4.4 c-values |
| 6 | Rollout-to-train FLOP and time ratio by window W = 2, 3, all | P1b | §4.4 |
| 7 | End-to-end samples/s/GPU against verl-omni on the same hardware and configuration, at least three interleaved run pairs (§7.2) | P1b | P1b exit |
| 8 | Cut blocking cost C, restart time, and the job MTBF used for the registered interval; per-checkpoint and per-failure loss, persist versus drop (§7.3) | P1b (image), P4 (video) | §4.3, §7.3 |
| 9 | LoRA versus merged-weight refit time, colocated and disaggregated | P1b | §5.2 |
| 10 | VAE decode and reward latency (PickScore, OCR, VLM judge) per sample, and compressed-transfer overhead; rollout GPU·s per sample for the P1b family and, for video, 14B at 81 frames under the training and deployment samplers | P1b, P4 | §3.2, §4.1 |
| 11 | Data-plane barrier hold time per storage unit and per cut against buffer size (§5.6) | P4 | §4.3 persist cost |
| 12 | `old` residency overhead: GPU, CPU, layer-wise prefetch, and bf16 read copy; host-to-device bandwidth with eight GPUs copying | P5 | §5.3 |
| 13 | Snapshot cost and restore loss for `restart_only` and `drain` samplers (§7.4) | P4 | §7.4 |
| 14 | Env-steps/s against the reference library, all-reduce cost of normalizer and curriculum state, and planner latency share in hybrid mode | P6a | §2.2, §2.5, §7.5 |
| 15 | Rendering-to-policy round trip p99 for each placement option (§5.7) | P6b | §5.7 |

## 9. Changes

### 9.1 From revision 6 (revision 7)

Revision 7 added M4 and M5 with a guarantee-per-mode table; let the pilot user pick the diffusion family (Qwen-Image default, SD3.5 Medium as the package B parity case); replaced revision 6's 768 px and 40–50 step sizing with real training configurations and per-row tables; made video-buffer persistence the default; added placement rules (CP in an NVLink domain, colocated diffusion, role residency, encoders outside the trainer, backend limits, CP-aware gets, simulator placement); added embodied versioning, normalizer, bootstrap, simulator-failure, and export rules; and replaced end-to-end targets with per-stage, end-to-end-including-services, CP-efficiency, and per-checkpoint targets.

### 9.2 Revision 8

- **Residency (PERF2-1):** chosen by read frequency; `old` stays on the GPU because DiffusionNFT reads it in every rollout and training forward (verified at `train_nft_sd3.py:640, 873-876`); `eval_ema` moves to the CPU; CPU-resident `old` costs 11–28% (estimate); the 14B fixed state is 45.5 GB, not 49 GB.
- **P1b throughput (PERF2-2):** end-to-end samples/s/GPU ≥ 0.9× the reference on the same hardware and configuration.
- **Checkpoint interval (PERF2-3):** registered in advance with Young/Daly; expected GPU-seconds lost per failure reported next to per-checkpoint loss, with a worked example.
- **Drain mode (PERF2-4):** cost model and selection rule for `restart_only` and `drain` samplers.
- **Storage units (PERF2-5):** video buffers on separate storage units; barrier hold time measured in P4.
- **Arithmetic (PERF2-6):** Wan buffer rows 256–512 and 6.6–13.2 GB; persistence also wins for SD3.5 (8–16 versus 175–350 GPU·s); c includes activation recompute (Qwen-Image c = 3, R = 0.6; Wan full fine-tune c = 5, R = 0.5 or 1.0); one recompute-overhead derivation (1/c); clip-cost table reconciling 400 and 1,200 GPU·s; Ulysses CP must divide 40 heads, so CP = 16 is invalid; 14B memory 45.5 GB; Wan iteration about 32 min.
- **Embodied (EMB2-1, EMB2-5, EMB2-6, EMB2-7):** P6a/P6b split with dependencies and exit criteria; declared hybrid mode with a frozen planner as a co-located M3 batch service and the semi-Markov discount in GAE; multi-GPU M4 synchronization of normalizer, curriculum, and quarantine state with per-replica seed ranges; rendering placement rule.
- **Alignment with revision 8 elsewhere:** packages R, C-up, and C-ext in the mode table; `precision` per mode; `rollout_role` in weight sync; refit digest only where the opt-in ask landed; sparse `final_observation`; GAE ask covers bootstrap and discount (P6a); fused runs refuse `safety_critical` data outside a validated workflow; fused-mode cuts written by a separate writer process.

### 9.3 Revision 9 (closing)

- **P1b baseline (PERF3-1):** verl-omni, at least three interleaved run pairs on the same nodes, statistics across runs; flow_grpo may remain the quality reference.
- **Checkpoint example (PERF3-5):** C = 39 s with EMA roles, τ ≈ 43 min; full-size EMA roles on 14B are an extrapolation from DiffusionNFT's LoRA `old`.

## Notes for the overview and sibling documents

Revision 7 notes 1–5 (P1 reference per family, the GAE bootstrap ask, H's optional dependency on NeMo RL code, the CP slice-fetch ask, naming SD3.5 Medium) were resolved in the overview. The five revision-8 notes are also resolved: the overview phase table carries the P1b throughput criterion, the registered interval, and the P6a and P6b rows; 02 §4.4 uses the corrected sizing; 02 §4.2 follows §7.4 (`restart_only` for periodic snapshots, `drain` for planned pauses); 01 §7.4 quotes 1/c; 04 §10.7 states the sampler and efficiency behind 1,200 GPU·s.

## Finding disposition (round 1)

| ID | Section | Resolution |
|---|---|---|
| EM-7 | §3.1 | The P1b family is the pilot user's; Qwen-Image is the default, SD3.5 Medium stays the package B parity case, FLUX is the second-family test. |
| PERF-1 (numbers) | §4.3 | Superseded in revision 8 by PERF2-6: video buffers are 6.6–13.2 GB, persist in 53–106 GPU·s, and cost 11–23 GPU·h to drop; SD3.5 also persists; the threshold formula goes to 02. |
| PERF-2 | §3.1, §4.1, §4.2, §4.5 | Payloads use real training configs, embeddings are stored once per group or by cache key, reward pixels are compressed, and CP ranks use sharded gets. |
| PERF-5 | §5.1 | Video CP groups stay inside one NVLink domain (about 30% over IB versus about 3% over NVLink); revision 8 adds the head-count divisibility rule. |
| PERF-6 | §5.2 | Diffusion defaults to colocated rollout and training with offload; refit is about 0.1% of a Wan iteration. |
| PERF-7 | §5.3, §5.4 | Superseded in revision 8 by PERF2-1: residency by read frequency (`old` on GPU, `eval_ema` on CPU) with a corrected 45.5 GB budget; frozen text encoders run outside the trainer. |
| PERF-8 | §5.5 | `simple` is limited to image-scale rows, video uses `mooncake_cpu`, and GDR requires every tensor ≤ 1 GiB. |
| PERF-10 | §7 | Per-stage overhead, end-to-end samples/s/GPU including services, intra-node CP efficiency, and checkpoint goodput; revision 8 adds the P1b ratio and registered intervals. |
| EMB-1 | §2.2, §2.3 | M4 keeps rollout, GAE, and update GPU-resident without the data plane, with per-iteration provenance and cuts. |
| EMB-4 | §6.1–§6.3 | Environment parameters and curriculum are versioned without starting reward epochs; normalizer statistics are in cuts and exports; bootstrap ownership is defined. |
| EMB-6 (placement) | §5.7 | Simulators are fused or co-located for ms-level loops, run on rendering GPUs when needed, HIL benches are exclusive resources; revision 8 adds the rendering placement rule. |
| EMB-7 | §6.4 | Per-instance handling with reset-and-mask, seed quarantine, explosion and drift detection, recycling, and a mask-rate limit. |
| EMB-8 | §6.5, §7.5 | Superseded in revision 8 by EMB2-1: the criteria are P6a exit criteria; P6b adds closed-loop success-rate TOST. |

## Round-2 disposition

| ID | Section | Resolution |
|---|---|---|
| PERF2-1 (numbers) | §5.3 | `old` is read every forward and stays GPU-resident (CPU residency 11–28% overhead, estimate; layer-wise prefetch only after a measured overlap check); `eval_ema` is on the CPU; the 14B total is 28 + 3.5 + 7 + 7 = 45.5 GB. |
| PERF2-2 | §7.2 | P1b requires end-to-end samples/s/GPU ≥ 0.9× the reference on the same hardware and configuration, judged on the lower bound of a bootstrap interval. |
| PERF2-3 | §7.3 | Intervals are registered in advance with Young/Daly τ = √(2·C·MTBF); expected GPU-seconds lost per failure and total overhead are reported next to per-checkpoint loss. |
| PERF2-4 | §7.4 | `drain` and `restart_only` costs are modeled (about 500 GPU·s per snapshot versus about 640 GPU·s per restore for 64 rollout GPUs, estimate); periodic snapshots use `restart_only`, planned pauses use `drain`. |
| PERF2-5 | §5.6, §8 item 11 | Video buffers live on separate storage units that write in parallel; barrier hold time is measured in P4. |
| PERF2-6 | §3.2, §4.3, §4.4, §5.1, §5.3 | Rows 256–512 and 6.6–13.2 GB; SD3.5 persistence wins; c includes activation recompute; one 1/c recompute derivation; clip-cost table; CP must divide 40 heads (CP = 16 invalid); 45.5 GB. |
| EMB2-1 | §3.3, §7.5 | P6a (fused control; P0, plus P2 for `safety_critical` data; may replace P1b in the MVP) and P6b (VLA flow heads; P3, P4, named rollout engine) with separate exit criteria. |
| EMB2-5 (hybrid mode) | §2.5, §6.3 | Declared hybrid mode: frozen VLM planner as a co-located M3 batch service called at planner rate, controller in M4; 01's discount field feeds GAE. |
| EMB2-6 | §2.2 item 7, §6.1, §6.2 | Normalizer, curriculum, and quarantine state are all-reduced at iteration boundaries with digest checks; ledger entries record per-replica seed ranges. |
| EMB2-7 (rendering placement) | §5.7 | Policy inference on rendering nodes by default, mixed nodes where available, otherwise cross-node with a declared latency budget enforced by the broker. |

## Round-3 disposition

| ID | Section | Resolution |
|---|---|---|
| PERF3-1 | §7.2, §8 item 7 | The P1b throughput baseline is verl-omni, with at least three interleaved run pairs on the same nodes and the interval computed across run pairs; the quality reference may remain flow_grpo. |
| PERF3-5 | §7.3 | The Young/Daly example uses C = 39 s with EMA roles (τ ≈ 43 min); DiffusionNFT's reference `old` is a LoRA adapter, so the 14B full-fine-tune case is an extrapolation. |
| PERF3-2, PERF3-3 | §5.1, §4.3 | Deferred to P4; see `open-issues.md`. |
| PERF3-4 | §5.3, §8 | Deferred to P5; see `open-issues.md`. |
| EMB3-1–EMB3-6 | §5.7, §7.5 | Deferred (EMB3-1 blocks any HIL session; EMB3-2, EMB3-3, EMB3-5, EMB3-6 to P6a; EMB3-4 to P6b); see `open-issues.md`. |
