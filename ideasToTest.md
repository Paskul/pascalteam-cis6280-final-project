# ideasToTest — Planning-track ideas for `GoalPlanningAgent`

*CIS 6280 World Models (Fall 2026) final project · four-box stacking · planning track · compiled 2026-11-23*

**Scope.** Every idea below is a change *inside* `GoalPlanningAgent` (`assignment/training/planning/agent.py`, mirrored in cell 7 of `assignment/planning/colab.ipynb`). The world model (`jepa/smwm-c3/model.pt`), the policy (`policy/df-smwm-c1-sigma04/policy.pt`), the subgoal file and the evaluation loop stay frozen. Ideas that need anything outside the class (offline search, an extra fitted component) are flagged **[rules?]** — confirm with course staff before relying on them.

**Interface contract to keep.**
- The constructor is called as `GoalPlanningAgent(policy_path, 'cuda', **config)`, where `config` is the `--configs` JSON. New knobs must be popped before `super().__init__`, because `LatentPolicyAgent` rejects unknown kwargs.
- `reset(goal_frames, init_frame, seed)` must clear all new state. The evaluator reuses one agent object across episodes in each worker.
- `step(frame)` must return a `(5, 25)` array in [−1, 1]. Anything outside that range raises inside the environment and crashes the run.
- `self.index` must stay an int. The evaluator reads it for videos and the report.
- Every knob should default to the baseline, so a single `evaluate.py` call can A/B several configs.

**Notation.**
- k is the agent call, one 5-step block (0.5 s).
- m is the active subgoal index.
- g_j are the subgoal latents (192-D), and gap_j = ‖g_j − g_{j−1}‖².
- A candidate is 16 normalized action blocks c_1…c_16.
- ẑ_b is the candidate's predicted latent after block b (b = 1…16). Block 16 is one subgoal interval (80 control steps) ahead.
- C is the candidate's cost.

**Contents.**
1. Where the baseline fails
2. Literature review
3. How to test
4. The 48 ranked ideas
5. Suggested schedule
6. References

---

## 1. Where the baseline fails (the facts that drive the ranking)

Source: our analysis of the authors' per-episode reports (HF `eval/planning-val300/`, 900 episodes per agent) and of their earlier 100-scene logs.

- **Small and noisy gain.**
  - Planning reaches 1.41 mean boxes against 1.27 without it. The paired gain is 0.141 ± 0.042 (SE, 900 episodes).
  - Seed noise dominates: within-scene variance is 0.88, between-scene variance 0.29.
  - The paired SE is about 0.07 boxes for 300 scenes × 1 seed, and about 0.13 for 100 scenes.
- **Box-2 cliff.**
  - Success funnel: ≥1 box 85%, ≥2 41%, ≥3 12%, all 4 2.4%.
  - P(place 2 | placed 1) = 0.49, versus 0.44 with no planning. So part of the cliff is the context-1, memoryless policy itself.
- **On schedule, then late.**
  - Box 1 lands on the scripted reference's schedule (median step 316 vs 298).
  - From box 2 on, successful agents run 120–130 steps late: 760 / 1160 / 1576 vs 637 / 1024 / 1448.
- **The subgoal index stalls.**
  - The median episode ends one-third of the way through its ~19 subgoals. Only 11 of 900 ever reach the last one.
  - The largest bucket (394/900, stuck at one box) places box 1 by about step 320. It then never gets past subgoal ~6 (reference step ~480, approaching box 2) for the remaining ~1,475 steps.
- **Stacks degrade.** 152 of the 373 episodes that reached ≥2 boxes end with fewer boxes counted.
- **The world model is good locally and bad at a distance.**
  - The 16-block per-dimension MSE is 0.0059, about 3% of latent variance and 3× the 1-block error.
  - Its top-1 agreement with the simulator's best candidate is 77.6%.
  - Distant goals break the L2 score. With one goal per placement and 64 candidates, planning scored 0.80 mean boxes, against 1.31 with no planning (authors' 100-scene logs, older WM).
  - Dense every-80-step goals with 64 candidates reached 1.52 in the same logs.
- **Latent scale is uneven.** Per-dimension std ranges from 0.23 to 0.88, about a 14× spread in variance, so raw L2 weights a few dimensions heavily.

---

## 2. Literature review

### 2.1 Where our planner sits

Our planner is **best-of-N "sample, score, rank"**:
- a frozen generative policy proposes candidate action chunks;
- a frozen latent world model scores them against goal images;
- the winner is executed with receding-horizon replanning.

The same template appears in NWM's ranking mode, GPC-RANK, FOREWARN, V-GPS (scored by a Q-function instead of a world model), CAST, and the first iteration of LeWAM's noise-space search.

The literature improves this template along five axes, and the idea list follows them:
- **(a) Target:** what to aim at (subgoal choice and tracking).
- **(b) Score:** cost shape, metric, uncertainty, and robust selection.
- **(c) Proposals:** more, more diverse, warm-started, refined, or noise-optimized candidates.
- **(d) Execution:** commitment, replanning cadence, and recovery.
- **(e) Compute:** where and how to spend it.

The field's large efforts are scaling world models: V-JEPA 2, NVIDIA Cosmos and Cosmos Policy, Dreamer 4, and Genie 3. For a frozen-model planning track, the relevant threads are test-time search with compact JEPA-style latent models, and test-time search over world-action models.

### 2.2 JEPA / latent world models and planning (the lineage of our world model)

**LeWM** (Maes, Le Lidec, Scieur, LeCun, Balestriero; NeurIPS 2026) is the architecture source of our world model (see `THIRD_PARTY.md`).
- Encoder: ViT-tiny CLS token, then a BatchNorm projector, giving a 192-D latent.
- Predictor: 6 layers with AdaLN action conditioning and 10% dropout. The appendix reports that dropout 0.1 beats 0 by a wide margin.
- Planning: CEM with 300 samples × 30 iterations, 30 elites, a horizon of 5 action blocks, a **terminal** squared-L2 cost, and no warm start. The appendix reports CEM above Adam, RMSProp and SGD.
- Analysis: latent trajectories become temporally straight during training, and latent "surprise" spikes on physically implausible events.
- Our SMWM variant replaces SIGReg with an **inverse-dynamics regularizer**. Three 2026 papers below identify inverse-action supervision as the key ingredient for action-sensitive, plannable latents, so the model we were given is the "right" variant.

**DINO-WM** (Zhou, Pan, LeCun, Pinto; ICML 2025) runs CEM-MPC over frozen DINOv2 patch features with a terminal MSE cost. CEM beat gradient descent.

**V-JEPA 2-AC** (Meta FAIR, 2025):
- Energy: L1 distance at the **final** imagined state.
- CEM: 800 samples × 10 iterations, horizon 1.
- Actions are constrained to a small L1 ball, "since large actions are relatively out-of-distribution."
- **Pick-and-place uses three subgoal images switched on a fixed schedule** (4 / 10 / 4 planning steps), with no success check.

**PLDM** (Sobal, Zhang, Cho, Balestriero, Rudner, LeCun; NeurIPS 2025) shows that planning with JEPA latent dynamics generalizes better than model-free RL when learning from reward-free, suboptimal data.

**NWM** (Bar, Zhou, Tran, Darrell, LeCun; CVPR 2025) can plan from scratch or **rank trajectories sampled from an external diffusion policy** (NoMaD). That ranking mode is our template.

**What Drives Success in Physical Planning with JEPA World Models?** (Terver, Yang, Ponce, Bardes, LeCun; 2026):
- "CEM L2 performs best overall."
- L2 consistently beats L1.
- Gradient planners do poorly on navigation and DROID.
- Planning context W_p = 2 yields the best success rate while being cheaper, with a big gap between context 1 and 2. Planning context must not exceed training context.
- A 2-step rollout loss helps training.

**H-JEPA** (Zhang, Terver, Rabbat, LeCun, Balestriero; Oct 2026):
- A three-level hierarchy raises Visual AntMaze success from 18% to 73% with less planner compute.
- The paper reports that **subgoal-tracking costs are far more monotone along expert trajectories** than full-horizon goal costs.
- Execution uses commit schedules.
- An inverse-dynamics loss was needed on DROID.

**Mind the Gap: Hierarchical Planning in LeWorldModel** (Caselli et al.; Jul 2026):
- Naive hierarchical LeWM underperforms flat planning, because high-level CEM finds *off-support* macro-actions that score well on the terminal latent cost but make poor targets.
- Constraining the search to perturbations of empirical macro-actions (an inference-time support constraint) recovers much of the loss.
- The paper reports that **oracle future-latent subgoals raise success to ~73%, against ~53% flat**.
- Staged, fixed-duration execution helps at intermediate subgoal spacing and hurts at long spacing.

**Keeping JEPA World Models Plannable When Little of the Frame Moves** (Strohm, Wagner, Schwab, Huber; Oct 2026):
- When objects cover few pixels, encoders turn action-insensitive and cost surfaces go flat. An inverse-dynamics auxiliary loss fixes this.
- Test-time findings: a **terminal-only cost is "far worse" than a mean-over-horizon path cost**.
- Stage goals are switched when the latent distance falls to **1/10 of its entry value, or progress stalls for three steps** (reported).
- Our boxes are a small fraction of a 128×256 frame, so this is our regime.

**Decision-Metric Alignment in Latent World Models** (Wang et al.; Aug 2026):
- What predicts planning success is whether the latent cost *orders* candidate plans like the true cost (their "Plan-Real Spearman" diagnostic), not probe accuracy.
- The paper reports that rank correlation **collapses to ≈0 among CEM elites**. Over-optimizing the latent cost selects noise.
- Inverse-action supervision gives most of the ranking lift.

**Anisotropic Representations Improve Planning in JEPA World Models** (Kang et al.; Sep 2026) proves that with isotropic latents, a Euclidean latent cost equals an inverse-covariance-weighted state metric. That metric yields positive planning regret even with perfect prediction. The metric matters.

**When More Planning Hurts** (Chen; Oct 2026, a single-author PushT study):
- Extra optimization *lowered* the selected latent costs, yet caused failures.
- iCEM with 500 candidates solved 2 of 16 scenes. A learned updater or the released planner did far better.
- The paper also reports that carrying the previous plan forward mattered (later sections, unverified).

**SWIM** (Hu et al.; Oct 2026) trains JEPA world models with behaviour-weighted state metrics. It offers no test-time metric, but motivates weighting latent dimensions by task relevance.

### 2.3 Steering frozen generative policies at test time

**GPC** (Qi, Yin, Zhu, Du, Yang; RA-L 2026):
- GPC-RANK is best-of-N through a predictive world model. GPC-OPT refines a policy sample by gradients through the world model.
- A world model trained only on expert data cannot correct policy errors. Ours was finetuned on the policy's own candidate rollouts, which is the right data.

**LeWAM / Diffusion-Steering MPC** (Hegde, Popov, Aljalbout, Smolyanskiy; NVIDIA, Oct 2026). This is the closest published system to ours: a JEPA world-action model whose rectified-flow policy head is steered at test time.
- **CEM / MPPI / CMA-ES operate over the policy's initial noise**, not over actions.
- Budget: 64 samples × 3 iterations, 8 elites, plans of 5 chunks decoded at imagined states.
- **Projecting each noise sample onto the Gaussian sphere was critical**: Can success was 78.8% with projection and 23.0% without.
- **Action-space MPC collapsed** to 3–8% success.
- The goal is the demo frame 25 steps ahead of the nearest neighbour of the current latent.
- Planning *underperformed* the policy on the task where the policy was already strong.

**Golden Ticket** (Patil et al.; RAI Institute, 2026). One fixed, searched noise vector improves frozen diffusion and flow policies: 46 of 51 tasks in v4, with up to 79% absolute gain in simulation. Ticket quality varies from near 0% to near 100% success, and some tickets steer toward failure.

**DSRL** (Wagenmaker et al. 2025) runs RL in a diffusion policy's noise space. It requires training.

**DynaGuide** (Du, Song; NeurIPS 2025) guides diffusion-policy denoising with a separate dynamics model toward goal observations, or away from negative ones.

**ReGuide** (Lin, Shakkottai, Kalathil, Kumar; Jun 2026):
- Test-time guidance toward per-phase target sets, using a soft-min distance over ~50 centroids per phase.
- Guidance only fires inside a percentile gate.
- A phase advances when the next target is closer by a margin for K consecutive steps.
- Gains from test-time guidance alone were small: +0.01 to +0.05 absolute.

**ProgressVLA** (Yan et al.; 2026) uses a learned progress estimator plus world-model gradients to guide a diffusion policy. It also checks for monotonicity violations. Its components are trained.

**FOREWARN** (Wu, Tian, Swamy, Bajcsy; RSS 2025) imagines outcomes of sampled plans with a latent world model and verifies them with a VLM, reporting more than 30% improvement.

**V-GPS** (Nakamoto et al. 2024) re-ranks a generalist policy's samples with a learned Q-function.

**CAST** (Moroncelli, Roveda; Nov 2026):
- It samples N chunks from a frozen diffusion policy, certifies them, then selects among those that pass.
- A **viability** score from model rollouts also helps. The paper reports that a single rollout (M = 1) is nearly as good as four.
- A **guard** keeps the nominal sample unless every critic prefers another.
- **Best-of-N degraded as N grew** (Can at N = 32; collapse at N = 1024), because selection picks up critic errors.

**BID — Bidirectional Decoding** (Liu et al.; ICLR 2025):
- Longer action chunks capture temporal dependencies but lose reactivity.
- It samples many chunks and picks one by *backward coherence* with past decisions plus *forward contrast* against a weaker policy.
- It improved two generative policies across 7 benchmarks.

**RTC — Real-Time Chunking** (Black, Galliker, Levine; NeurIPS 2025) treats the next chunk as **inpainting**: the committed prefix is frozen and the remainder is generated, with soft masking. It needs no retraining for flow policies.

**RoboMonkey** (Kwok et al.; CoRL 2025):
- Action error falls roughly as a power law in the number of samples.
- Gaussian perturbation plus majority voting builds cheap proposals.
- A verifier selects the final action.

**MG-Select** (Jang et al.; ICLR 2026) selects without a verifier, using KL to a condition-masked reference. It needs condition-dropout training, so it doesn't apply to our policy.

**ELASTIC** (Li, Swamy, Bisk, Bajcsy; 2026) learns per-step compute allocation. Extra compute pays off at **grasps, placements and action-mode splits**.

**When Does Test-Time Search Help World Action Models? An Empirical Audit** (Liu et al.; Oct 2026):
- In **82% of query states, all candidates shared the same outcome**.
- A learned value selector did *worse than random*.
- Recommendation: measure selection headroom and scorer reliability before tuning the selector.

### 2.4 Sampling-based MPC and inference-time search for diffusion / flow models

**MPC machinery.**
- CEM in latent space: PlaNet (Hafner et al. 2019).
- iCEM (Pinneri et al. 2020): colored noise, elite reuse, and shifting the previous solution forward.
- MPPI (Williams et al. 2017).
- TD-MPC2 (Hansen, Su, Wang 2024): MPPI that mixes **policy-prior samples** with Gaussian samples and warm-starts from the previous solution.
- D-MPC (Zhou, Swaminathan, Murphy et al.; TMLR 2025): diffusion action proposals plus diffusion dynamics with sample–score–rank.
- Model-Based Diffusion (Pan et al. 2024): diffusion-style annealed trajectory optimization.

**Inference-time scaling of diffusion and flow models.**
- Ma et al. (2025): random search, zero-order search around good noises, and search over denoising paths.
- **FK steering** (Singhal et al. 2025): particles resampled by intermediate rewards.
- BFS/DFS search with local refinement (Zhang et al.; ICLR 2026).
- **Flash-BoN** (Rawal et al. 2026):
  - Cheap drafts, staged selection, then resuming refinement from the cached state.
  - Plain best-of-N matches or beats guided search at equal wall-clock time.
  - **Candidate diversity correlates with the best-of-N gain (r ≈ 0.75).**
- **IPR** (Kang, Yoon, Ahn; May 2026): random region-wise re-noising and regeneration. It requires a model trained with *mixed per-region noise levels*, which is exactly what Diffusion Forcing trains.
- SDEdit (Meng et al. 2022): partial re-noising.

**Diffusion Forcing (DF).** DF (Chen, Martí Monsó, Du, Simchowitz, Tedrake, Sitzmann; NeurIPS 2024) is our policy's recipe:
- Independent per-token noise levels and causal attention.
- Pyramid schedules.
- **Monte Carlo Guidance**, which plans by using sampled futures.

Related DF-based work:
- MCTD (Yoon et al. 2025): tree search over DF denoising.
- Streaming Diffusion Policy (Høeg, Du, Egeland; ICRA 2025): keeps partially denoised future actions across control steps.

### 2.5 Subgoal following, progress tracking, recovery

Published subgoal-switching rules:
- A fixed time schedule (V-JEPA 2-AC).
- Entry-relative distance or a stall rule (Strohm et al.).
- A margin held for K consecutive steps (ReGuide).
- Nearest-neighbour retrieval plus a fixed lookahead (LeWAM).
- **Distributional progress with causal peak selection** (REMORA, Wang et al.; Oct 2026). Among the candidate progress peaks, it picks the one closest to the previous estimate. This keeps continuity, but progress is allowed to *decrease* after errors and then re-ramp.

Classical counterparts:
- Pure pursuit (Coulter 1992): steer toward a point a fixed lookahead ahead on the path.
- **Model predictive contouring control** (Lam et al. 2010; Liniger et al. 2015): progress along the reference path is a decision variable, decoupled from time.
- DTW / TCC video alignment (Dwibedi et al. 2019).
- HMM forward filtering (Rabiner 1989), taught as belief-state inference in CIS 6280 L4.

### 2.6 Uncertainty, monitoring, and "when more planning hurts"

**Uncertainty estimates.**
- MC dropout as an approximate ensemble (Gal, Ghahramani 2016).
- PETS ensembles (Chua et al. 2018).
- EpicWorldModel (2026) uses ensemble disagreement inside CEM.
- Dream-MPC and EfficientTDMPC (2026) add pessimistic uncertainty penalties.

**Counter-evidence, so test these carefully.**
- Uncertainty penalties *increased* collisions in one 2026 study ("Learning from World Feedback").
- Epistemic estimates in latent dynamics models are biased ("Biased Dreams", 2026).

**Monitoring.**
- Sentinel (Agia et al.; CoRL 2025): temporal action consistency (STAC) plus VLM progress checks detect more than 97% of diffusion-policy failures.
- LeWM: latent surprise as violation-of-expectation.

**The cross-cutting warning.**
- Over-optimizing a noisy latent cost backfires. Evidence: elite-rank collapse, CAST's large-N degradation, "When More Planning Hurts", and the audit's worse-than-random selector.
- Proposals should stay on the policy's support. Evidence: LeWAM's sphere projection, Mind the Gap's empirical macros, V-JEPA 2's action ball.

### 2.7 Connections to Jiatao Gu's work and the CIS 6280 syllabus

**Context.**
- Gu is Assistant Professor at Penn CIS since 07/2025, and a member of technical staff at AMI Labs.
- Before that he was at Apple MLR (2022–2026) and Meta FAIR (2018–2022).
- He teaches this course. His group members include Yong-Hyun Park (the repo's author), Mutian Tong, Xinyue Ai and Enxin Song, who are also the course TAs.
- Recent talks:
  - "Towards Robust World Models" (GRASP, Fall 2025). Its thesis is that large diffusion video world models remain brittle.
  - "Learning World Models for Robust Physical Intelligence" (Apr 2026).
  - "Are Normalizing Flows Good Candidates for Interactive World Models?" (CVPR 2026 workshop).

The links below are analogies that motivate ideas. None of these papers addresses robot planning directly.

| Gu et al. work | What it does | What it suggests for our planner |
|---|---|---|
| **Non-autoregressive NMT + noisy parallel decoding** (ICLR 2018) | Samples many candidates in parallel from a fast generator and rescores them with a stronger autoregressive teacher | Our loop is noisy parallel decoding (the policy proposes, the WM rescores). Proposal diversity and rescoring quality are the levers → ideas 6, 13, 24 |
| **Levenshtein Transformer** (NeurIPS 2019), insertion-based decoding (TACL 2019) | Refines by editing (insert/delete) instead of regenerating | Repair the previous plan instead of resampling from scratch → ideas 7, 10, 16 |
| **Learning to Translate in Real-Time** (EACL 2017), **Monotonic Multihead Attention** (ICLR 2020) | Learns when to READ vs WRITE, and a monotonic alignment to a source stream | When to commit vs replan → ideas 8, 33. Monotonic, but recoverable, alignment of our trajectory to the reference subgoal stream → ideas 1–4, 11, 28 |
| **Trainable Greedy Decoding** (EMNLP 2017) | Steers a frozen decoder through its inputs toward an arbitrary objective | Steer the frozen policy through its noise → ideas 15, 40, 43 |
| **DART** (ICLR 2025), **Kaleido Diffusion** (NeurIPS 2024) | AR × diffusion hybrids; diversity from autoregressive latent priors | Per-token DF schedules and streaming denoising → ideas 23, 47. Candidate diversity → idea 24 |
| **TarFlow / STARFlow / STARFlow-V / Normalizing Trajectory Models** (2025–26) | Exact likelihood and few-step generation | Likelihood-aware selection → idea 42. Fewer denoising steps buy more candidates → idea 26 |
| **Multiplex Thinking** (2026) | Token-wise branch-and-merge reasoning | Branching search over chunks that merges near-duplicate branches → ideas 41, 13 |
| **Matryoshka Diffusion** (ICLR 2024), **BOOT** (ICML 2024) | Coarse-to-fine generation; fast sampling | Cheap draft screening, then full rollouts → idea 36. Speed → ideas 25, 26 |

**Syllabus mapping.**

| Lecture | Topic | Ideas |
|---|---|---|
| L4 | Filtering, belief states | 11 |
| L9 | PlaNet, MPC, CEM | 15, 16 |
| L10 | TD-MPC, model bias | 14, 34 |
| L11 | Diffusion and flow matching | 10, 16, 38 |
| L13 | Closed-loop drift | 22, 41 |
| L14 | Flows | 42 |
| L19 | World-action models | 15 |
| L21 | Test-time computation: branching, verification, adaptive compute | 6, 13, 20, 41 |
| L23 | Calibration, OOD, drift, failure | 12, 14 |

### 2.8 What this implies for us

1. **Subgoal management is the cheapest large lever.**
   - Our data shows the index stalling.
   - Published planners use lookahead retrieval, entry-relative or stall switching, time schedules, and distributional tracking.
   - → ideas 1–4, 9, 11, 19, 30.
2. **Terminal-only latent costs are fragile.** Path costs and subgoal-tracking costs are more monotone and work better (Strohm et al., H-JEPA). → ideas 5, 28, 29.
3. **Don't over-trust argmin on a noisy cost.** Elite-rank collapse, large-N degradation, and the audit all point the same way. Prefer robust selection, uncertainty, and staying on the policy manifold. → ideas 13, 14, 15, 17.
4. **Refine in noise space, not action space.** Use noise-space search, SDEdit or DF inpainting, never raw action perturbation (LeWAM, Mind the Gap). → ideas 10, 15, 16.
5. **Temporal consistency matters for chunked generative policies** (BID, RTC, CAST's guard, iCEM/TD-MPC2 warm starts). → ideas 7, 8, 10, 31, 32.
6. **Spend compute where decisions happen** (ELASTIC, the audit). → ideas 20, 21, 33.
7. **Measure headroom first.**
   - If no sampled candidate succeeds, invest in proposals (6, 15, 16, 21, 22).
   - If good candidates exist but aren't picked, invest in targets and scoring (1–5, 13, 14, 17).

---

## 3. How to test

### 3.1 Helpers to write once (many ideas reuse them)

```python
# (a) Per-token-noise Diffusion-Forcing sampler. Enables prefix inpainting (10), SDEdit refinement (16),
#     noise-space search (15), drafts (36), particles (37), streaming (47).
def _denoise(self, z, x, t0, n_steps):
    """z: (n, 1, 192) z-scored latent context; x: (n, 16, 125) tokens at per-token noise levels t0: (n, 16)
    (1 = pure noise, 0 = clean/frozen). Rectified flow: x_t = (1-t) a + t eps; the policy predicts v = eps - a."""
    t, dt = t0.clone(), t0 / n_steps         # every token reaches t = 0 after n_steps; t0 = 0 tokens never move
    for _ in range(n_steps):
        v = self.policy(z, x, t).float()     # DFPolicy.forward accepts per-token t of shape (n, 16)
        x = x - dt[..., None] * v            # same Euler update as DFPolicy.sample
        t = (t - dt).clamp(min=0)
    return x                                 # clean "draft" at any point: a_hat = x - t[..., None] * v
# Draw every noise tensor from self.generator so episodes stay reproducible.

# (b) MC-dropout rollouts (14). Dropout (p = 0.1) lives in the predictor; pred_proj contains BatchNorm -> keep eval.
self.encoder.predictor.train()   # repeat ctx / act_emb R times along the batch dimension, then call rollout once
...
self.encoder.predictor.eval()

# (c) Differentiable rollout (38-40): reuse finetune_onpolicy.rollout(model, ctx, acts, steps, grad=True)
#     inside `with torch.enable_grad():`, because GoalPlanningAgent.step is decorated with @torch.no_grad().
```

Also add an **opt-in trace** (`trace=True` knob) that appends a dict per call to `self.trace`. Fields:
- k and the current index;
- distances to subgoals m−2 … m+3, and the switching threshold;
- the chosen cost and the spread of candidate costs;
- surprise ‖z − ẑ_1(prev)‖² (the gap between the observed latent and what the WM predicted for it);
- candidate dispersion (spread of the candidates' predicted end latents).

A small dev script that mirrors `run_episode` can dump the trace. Most stall hypotheses (e.g., "the τ·gap threshold is too strict in slow phases") are cheap to verify from it before building anything.

### 3.2 Protocol

1. **Headroom check (dev script, not part of the agent).**
   - At sampled states, execute each of the N candidates in the simulator from a saved state, the way `collect_onpolicy.py` does.
   - Record a privileged outcome, such as box positions or placed count after 16 blocks.
   - Report the fraction of states where candidates differ, and the oracle gain. This decides between proposal-side and scoring-side work (§2.8.7).
2. **Offline proxy for scoring ideas (5, 13, 14, 17, 27).**
   - The released `onpolicy-smwm-c3/val-*.npz` shards hold 6,208 states × 16 candidates × 16 ground-truth future latents (1 GB).
   - Compare each scoring rule on predicted latents vs the same rule on the true latents (agreement / regret).
   - This tests robustness to world-model error. It does not test whether a cost shape leads to task success.
3. **Closed-loop ladder, always paired** (same scenes and same `--sample-seeds`):
   1. A 50-scene smoke test for bugs and timing.
   2. 300 scenes × seed 0 (paired SE ≈ 0.07).
   3. The full 300 × {0, 2, 3} (SE ≈ 0.04) for anything that will be reported.
   - Report mean boxes, ≥2 and ≥3 rates, P(2 | 1), the final goal-index fraction, and `agent_s`.
4. **Guard against overfitting the dev set.** Tune on val scenes 1–150 and confirm on 151–300. The competition uses held-out scenes.
5. **One knob per idea, default off.** Pass several `--configs` JSONs in one `evaluate.py` call; each config runs over all scenes. For example:

   ```
   --configs '{"candidates":16,"generate":16,"execute":1,"tau":0.25}' '{"candidates":16,"generate":16,"execute":1,"tau":0.25,"target_mode":"lookahead"}'
   ```

---

## 4. The ideas, ranked by importance and immediacy

**How the ranking works:** expected gain, judged from our failure analysis and the literature, times how fast the idea can be tested.
- **Tier 1:** cheap changes aimed at the measured index stall and at the cost shape.
- **Tier 2:** stronger search and robustness.
- **Tier 3:** refinements.
- **Tier 4:** expensive, exploratory, or needs a rules check.

**Effort:** S < ½ day (often config-only or under 50 lines), M = 1–2 days, L > 2 days. **Compute** is per replan, relative to baseline.

### Overview

| # | Idea | Targets | Effort | Compute | Key sources |
|---|---|---|---|---|---|
| 1 | Re-localize the subgoal index in a forward window (skip-ahead) | index stall | S | 0 | LeWAM, REMORA |
| 2 | Pure-pursuit lookahead target (nearest + 1 interval) | index stall | S | 0 | Pure pursuit, LeWAM |
| 3 | Stall timeout / no-progress forced advance | index stall | S | 0 | Strohm et al., V-JEPA 2-AC |
| 4 | Entry-relative switching threshold (+ τ sweep) | index stall | S | 0 | Strohm et al., authors' logs |
| 5 | Time-aligned trajectory (path) cost | cost shape | S–M | 0 | Strohm et al., H-JEPA, LeWM |
| 6 | Scale candidates (16 → 32 / 64 / 128) | proposals | S (config) | + / ++ | RoboMonkey, CAST, authors' logs |
| 7 | Warm-start incumbent + "keep unless clearly better" guard | dithering | M | + | iCEM, TD-MPC2, BID, CAST |
| 8 | Execute > 1 block per replan | dithering, compute | S (config) | − | BID, Diffusion Policy, Gu 2017 |
| 9 | WM-informed subgoal skipping (reachability) | index stall | S | 0 | Mind the Gap, FF-JEPA |
| 10 | DF prefix inpainting (committed candidates) | dithering, proposals | M | + | Diffusion Forcing, RTC, LevT |
| 11 | HMM / Bayesian filter over the subgoal index | stall, recovery | M | 0 | REMORA, CIS 6280 L4 |
| 12 | Surprise detection + non-monotonic recovery | recovery | M | 0 | LeWM, Sentinel, REMORA |
| 13 | Robust selection among near-ties (medoid / softmax / abstain) | scorer noise | S | 0 | Decision-metric, audit, MG-Select |
| 14 | MC-dropout WM ensemble (mean, optional pessimism) | scorer noise | S–M | + | Gal & Ghahramani, LeWM, PETS |
| 15 | Noise-space CEM / MPPI with sphere projection | proposals | M | ++ | LeWAM, Golden Ticket |
| 16 | SDEdit elite refinement (partial re-noising) | proposals | M | + | SDEdit, Ma et al., IPR |
| 17 | Normalized / whitened latent metric | cost metric | S | 0 | Anisotropic, Decision-metric |
| 18 | Planning context W_p = 2 or context ensemble | scorer | S | 0 / + | What Drives Success |
| 19 | Hysteresis and dwell for all switching rules | stall flicker | S | 0 | ReGuide |
| 20 | Adaptive compute from candidate dispersion | compute | M | ± | ELASTIC, audit, Sentinel |
| 21 | Stuck → exploration boost | stall | S–M | + (local) | RoboMonkey, Flash-BoN |
| 22 | One-chunk viability lookahead | long horizon | M | ++ | DF Monte Carlo Guidance, CAST, LeWAM |
| 23 | Pyramid noise schedule (`lag` > 0) | proposals | S (config) | 0 | Diffusion Forcing |
| 24 | Diversity-engineered noise (antithetic / orthogonal) | proposals | S | 0 | Flash-BoN, Kaleido |
| 25 | CUDA graphs / `torch.compile` the planner | compute | M | −− | engineering |
| 26 | Fewer Euler steps, more candidates | compute | S (config) | − | NTM, BOOT |
| 27 | Inverse-dynamics self-consistency penalty | scorer | S | + | Strohm et al., Decision-metric |
| 28 | Progress-along-path (MPCC-style) cost | cost, stall | M | 0 | MPCC |
| 29 | Min-over-trajectory "reach" cost + continuation | cost shape | S | 0 | H-JEPA |
| 30 | Reference-clock prior / maximum lag | stall | S | 0 | V-JEPA 2-AC |
| 31 | Backward-coherence penalty | dithering | S | 0 | BID |
| 32 | Within-mode ensembling of the executed block | jitter | S | 0 | ACT, MPPI |
| 33 | Adaptive commit length (READ / WRITE) | dithering, compute | M | − | Gu 2017, ELASTIC |
| 34 | Proposal portfolio (TD-MPC2-style mixture) | proposals | M | + | TD-MPC2, iCEM |
| 35 | Sampling temperature (initial-noise scale mixtures) | proposals | S | 0 | CAST, LeWAM |
| 36 | Draft-and-prune candidates (successive halving) | compute | M | ± | Flash-BoN, Matryoshka |
| 37 | Particle (Feynman–Kac / SMC) steering | proposals | L | ++ | FK steering |
| 38 | World-model-gradient guidance during denoising | proposals | L | +++ | DynaGuide, ReGuide, ProgressVLA |
| 39 | Gradient refinement with a trust region (GPC-OPT) | refinement | M | ++ | GPC |
| 40 | Gradient optimization of the initial noise | refinement | L | +++ | LeWAM, D-Flow |
| 41 | Beam / tree search over chunk sequences | long horizon | L | +++ | MCTD, Multiplex Thinking |
| 42 | Policy-likelihood (typicality) prior | scorer | M | ++ | NTM / STARFlow, FFJORD |
| 43 | Golden-ticket noise bank **[rules?]** | proposals | L (offline) | 0 | Golden Ticket |
| 44 | Offline latent-dimension weighting **[rules?]** | cost metric | M (offline) | 0 | SWIM |
| 45 | Inverse-dynamics "latent servoing" proposals | proposals | S | + | Latent Diffusion Planning |
| 46 | Pixel-level box-layout cues for stage tracking **[rules?]** | stall, recovery | M | 0 | Strohm et al. |
| 47 | Streaming (rolling) DF denoising with particles | compute, consistency | L | − | Streaming DP, DF, DART |
| 48 | Final-stage "finish and protect" mode | endgame | S | 0 | — |

---

### Tier 1 — do first (cheap, aimed at the measured failure)

**1. Re-localize the subgoal index inside a forward window (allow skip-ahead)** · S · +0 compute
- *Change.*
  - In `goal_index`, after the baseline loop, compute d_j/gap_j for j ∈ [m, m+W] (W ≈ 3).
  - If some j > m is relatively closer than m for K consecutive calls, jump `self.index` to j.
  - The index still only moves forward. Backward moves are idea 12.
- *Why.*
  - Stuck episodes freeze near subgoal ~6 for ~1,475 steps.
  - When the agent's route or timing deviates from the scripted reference, the active subgoal can become unmatchable while later ones are already closer.
  - LeWAM picks its goal by nearest-neighbour retrieval on the demo, plus a lookahead.
  - REMORA tracks progress by choosing the hypothesis nearest the previous estimate, rather than demanding an exact hit per stage.
- *Test.* `relocalize_window` ∈ {0 (off), 2, 3, 5}, `relocalize_k` ∈ {1, 2, 3}. Watch the final-index fraction and P(2 | 1).
- *Risk.* Premature skips between look-alike subgoals (hand hovering at the same spot). Normalize by gap and require dwell (idea 19).
- *Refs.* LeWAM; REMORA; TCC alignment.

**2. Pure-pursuit lookahead: aim one interval ahead of where you are** · S · +0
- *Change.*
  - Localize p = argmin over j ∈ [m−1, m+W] of d_j (optionally with a fractional position, see idea 28).
  - Plan toward g_{min(p+L, M−1)} with L = 1.
  - Set `self.index` to the target, which is what the video shows.
- *Why.*
  - A chunk is exactly one subgoal interval long (16 blocks = 80 steps), so its end should match "one subgoal past where I am". The baseline instead aims at "the first subgoal not yet matched within τ".
  - Pure pursuit and LeWAM's goal rule (nearest frame + 25 steps) both aim a fixed distance ahead of the localized point. This removes the need for exact hits, and keeps targets local, unlike the stage-goal failure (0.80 mean boxes).
- *Test.* `target_mode` ∈ {"threshold", "lookahead"}, `lookahead` ∈ {1, 2}.
- *Risk.* If localization is wrong, the target is wrong. Pair with ideas 11 and 19.
- *Refs.* Coulter 1992; LeWAM; V-JEPA 2-AC.

**3. Stall timeout: force progress when the target stops getting closer** · S · +0
- *Change.*
  - Track the number of blocks spent on the current index and the best distance to the active target so far.
  - Advance (or re-run idea 1 with a wider window) if either holds:
    - the index hasn't moved for S blocks (S ≈ 24–32, i.e. 1.5–2 intervals);
    - the best distance hasn't improved by ≥ 5% over the last K replans (K ≈ 6–8).
- *Why.*
  - Stuck-at-1 episodes waste most of the episode.
  - Strohm et al. switch stage goals when the distance has fallen far enough *or progress stalls*.
  - V-JEPA 2-AC switches its pick-and-place subgoals purely on a time budget. Time-based escape hatches are standard.
- *Test.* `stall_blocks` ∈ {off, 24, 32, 48}, `stall_patience` ∈ {6, 8}. Watch time to box 2.
- *Risk.* Skipping a subgoal the agent still needs, such as finishing a grasp. Pair with idea 12 so it can come back.
- *Refs.* Strohm et al. 2026; V-JEPA 2-AC.

**4. Entry-relative switching threshold (and a τ sweep)** · S · +0
- *Change.*
  - Store d_entry = ‖z − g_m‖² when m becomes active.
  - Advance when d ≤ ρ·d_entry **or** d ≤ max(τ·gap_m, τ_min·median(gap)).
- *Why.*
  - Hypothesis: τ·‖g_m − g_{m−1}‖² becomes extremely strict in slow phases (descend, close, release), where consecutive subgoal images barely differ. Then tiny permanent differences from the scripted scene, such as box offsets or finger pose, block advancement forever. Verify with the trace from §3.1.
  - Strohm et al. use an entry-relative rule.
  - The authors' logs show τ = 0.25 beat τ = 0.5 at 64 candidates (1.52 vs 1.43).
- *Test.* `tau` ∈ {0.15, 0.25, 0.35}, `rho_entry` ∈ {off, 0.1, 0.2, 0.3}, `tau_min` ∈ {0, 0.1}.
- *Risk.* Rules that are too lenient let the target outrun the agent. Track index vs the reference clock.
- *Refs.* Strohm et al.; authors' `eval/goal.jsonl`.

**5. Score the whole predicted trajectory against time-aligned targets** · S–M · +0
- *Change.* Replace ‖ẑ_16 − g_m‖² with C = Σ_b w_b ‖ẑ_b − ĝ(s_0 + b/16)‖².
  - ĝ(s) linearly interpolates the subgoal latents at fractional position s.
  - s_0 is the current localized position (ideas 2 and 28).
  - w_b ∝ b, or uniform.
- *Variants.*
  - **"tail4"**: the mean over blocks 13–16 against the target.
  - **"two_goal"**: ‖ẑ_8 − g_m‖² + ‖ẑ_16 − g_{m+1}‖² (the README's "look ahead to several subgoals").
- *Why.*
  - Strohm et al. find terminal-only cost "far worse" than a mean-over-horizon path cost for LeWM-family planners.
  - H-JEPA reports subgoal-tracking costs are much more monotone along expert trajectories.
  - Our cost uses 1 of 16 predictions, and the one with 3× the error.
  - LeWM latents are temporally straight, which supports interpolating between nearby subgoals.
- *Test.* `cost` ∈ {"final", "tail4", "aligned", "two_goal"}. Run the offline proxy (§3.2) first, then closed loop.
- *Risk.* A single static target over the whole horizon rewards "reach g_m and stop". Always use advancing targets.
- *Refs.* Strohm et al.; H-JEPA; LeWM; repo README.

**6. Scale the number of candidates (config only)** · S · + / ++
- *Change.* `candidates` ∈ {16, 32, 64, 128}. Both the policy and the WM are batched, and the GPU is underused at 16.
- *Why.*
  - In the authors' logs, 64 candidates with every-80 goals reached 1.52 vs 1.31 without planning.
  - RoboMonkey finds a power-law gain from more samples.
  - **But** CAST degrades at large N when the scorer is noisy. Our WM's top-1 agreement is only 77.6%, so expect diminishing returns.
- *Test.* Plot mean boxes and `agent_s` against N on the same 300 scenes.
- *Risk.* Winner's curse. Combine with ideas 13 and 14 at N ≥ 64.
- *Refs.* RoboMonkey; CAST; authors' logs.

**7. Warm-start incumbent with a "keep unless clearly better" guard** · M · +
- *Change.*
  - Store the chosen chunk `self.plan`.
  - Next call, form the incumbent: `plan[1:]` padded to 16 blocks, by repeating the last block or with idea 10.
  - Score the incumbent from the new context, alongside the fresh candidates.
  - Switch only if C_new < (1 − δ)·C_incumbent.
- *Why.*
  - Resampling from scratch every 0.5 s lets the chosen plan flip between modes (box A vs B, approach angle), which costs time. Our successful agents run ~120 steps late from box 2 on.
  - Precedents:
    - iCEM and TD-MPC2 shift the previous solution forward;
    - BID selects for coherence with past decisions;
    - CAST's guard keeps the nominal sample unless every critic prefers another;
    - "When More Planning Hurts" reports that carrying the plan forward mattered.
- *Test.* `warm_start` on/off, `guard_margin` δ ∈ {0, 0.05, 0.1, 0.2}. Watch placement time and the change in the first executed block between consecutive calls (jitter).
- *Risk.* Committing to a failing plan. Release the guard on surprise (idea 12) or a stall (idea 3).
- *Refs.* iCEM; TD-MPC2; BID; CAST.

**8. Execute more than one block per replan (config only)** · S · −
- *Change.* `execute` ∈ {1, 2, 3, 4}, which is already supported. Also test at matched compute (execute = 2 with 2× candidates).
- *Why.*
  - Open-loop commitment trades reactivity for consistency (BID; Diffusion Policy's action horizon). This is the READ/WRITE trade-off of Gu's simultaneous-translation work.
  - It halves compute, which funds ideas 6 and 15.
- *Risk.* Slower reaction at contact (grasp precision).
- *Refs.* BID; Diffusion Policy; Gu, Neubig, Cho, Li 2017.

**9. Let the world model pick the subgoal (reachability-based skipping)** · S · +0
- *Change.*
  - From the rollouts already computed, take r_j = min over n and b of ‖ẑ_{n,b} − g_j‖²/gap_j, for j ∈ {m, m+1, m+2}.
  - If r_{m+1} < r_m, the current subgoal is already passed or unreachable: advance, then re-score.
  - Optional "furthest-along" cost: min over j of [d(ẑ_16, g_j) − β·(j − m)].
- *Why.*
  - The imagined futures already contain the information needed to judge which subgoal is reachable.
  - Mind the Gap shows good subgoals matter enormously (oracle future-latent subgoals ~73% vs ~53% flat).
- *Test.* `skip_by_reach` on/off, β ∈ {0, 0.1·median(gap)}.
- *Risk.* WM optimism about later subgoals. Require a margin and dwell (idea 19).
- *Refs.* Mind the Gap; FF-JEPA; Multiplex Thinking (the branch-and-merge analogy).

**10. Prefix-conditioned candidates via Diffusion-Forcing inpainting** · M · +
- *Change.*
  - Use `_denoise` (§3.1) with t0 = 0 on the first p blocks of the incumbent plan, and t0 = 1 on the rest.
  - Produce K "committed" candidates plus N − K fresh ones.
  - *Soft variant:* re-noise the prefix to t_p ≈ 0.2 so it can adapt to the new observation, like RTC's soft masking.
- *Why.*
  - DF trains every token with an independent noise level, so a clean prefix with a noisy suffix is in-distribution.
  - The policy's causal attention lets the suffix condition on the prefix.
  - RTC shows inpainting keeps flow-policy chunks consistent with no retraining.
  - The result commits to the current intent while still exploring futures. It is the Levenshtein-Transformer move: edit, don't regenerate.
- *Test.* `prefix_blocks` p ∈ {2, 4, 8}, `committed_frac` ∈ {0.25, 0.5}, `prefix_t` ∈ {0, 0.2}.
- *Risk.* A stale prefix after a slip. Disable on surprise (idea 12).
- *Refs.* Diffusion Forcing; RTC; Levenshtein Transformer.

### Tier 2 — high value, moderate effort

**11. Bayesian (HMM) filter over the subgoal index** · M · +0
- *Change.*
  - Keep a belief b over the M subgoals.
  - **Predict:** b ← Tᵀb, where T stays with p_s, steps +1 with p_1, +2 with p_2, and −1 with a small p_back.
  - **Update:** b_j ∝ b_j·exp(−d_j / (2σ_j²)), with σ_j² ∝ gap_j.
  - **Target:** argmax(b) + 1, or minimize the belief-weighted cost Σ_j b_j·C(g_{j+1}).
- *Why.*
  - One principled mechanism handles timing differences, skips and regressions, using only latent distances.
  - It is a training-free cousin of REMORA's distributional progress tracking, and the exact tool from CIS 6280 L4 (filtering, belief states).
- *Test.* `index_mode` = "hmm". Sweep the σ scale and p_s. Compare belief trajectories with the reference clock.
- *Risk.* Distances are not calibrated likelihoods, so the emission temperature needs tuning.
- *Refs.* REMORA; Rabiner 1989; course L4.

**12. Surprise detection and non-monotonic recovery** · M · +0
- *Change.*
  - Keep the executed candidate's predicted ẑ_1. At the next call, surprise s = ‖z − ẑ_1‖², compared against its running median s̄.
  - If s > κ·s̄ (κ ≈ 4):
    - (a) re-localize over [m−3, m+2], allowing the index to *decrease*;
    - (b) drop the incumbent and prefix (ideas 7, 10);
    - (c) trigger idea 21.
- *Why.*
  - A monotone index cannot recover from drops or collapses. 152/373 episodes at ≥2 boxes lose counted boxes later, and the box must be regrasped.
  - Precedents:
    - LeWM's latent surprise detects implausible events;
    - Sentinel flags erratic failures via temporal consistency;
    - REMORA lets progress decrease when the evidence supports it.
- *Test.* `surprise_k` ∈ {off, 3, 4, 6}, `allow_regress` on/off. Inspect episodes with collapses or ground falls.
- *Risk.* False alarms at contact, where WM error is naturally higher. Use per-index running statistics.
- *Refs.* LeWM; Sentinel; REMORA.

**13. Robust selection among near-ties** · S · +0
- *Change.* Let S be the set of candidates within ε of the best cost. Choose:
  - (a) the medoid of S in predicted-end-latent space;
  - (b) the highest-mass mode under softmax(−C/λ) (MPPI-style);
  - (c) the incumbent (idea 7) if it is in S.
  - If costs are effectively tied (spread below a noise floor), abstain to the policy's first sample.
- *Why.*
  - Latent-cost rankings collapse to about 0 correlation among elites (Decision-metric).
  - The audit's learned selector was worse than random when headroom was small.
  - RoboMonkey and MG-Select use consensus.
  - argmin over noisy scores is a winner's-curse estimator, worsening as N grows (CAST).
- *Test.* `select` ∈ {"argmin", "medoid", "softmax", "incumbent_if_tied"}, ε ∈ {5%, 10%}.
- *Refs.* Decision-metric alignment; test-time search audit; RoboMonkey; MG-Select; CAST.

**14. Monte-Carlo-dropout world-model ensemble** · S–M · +
- *Change.* Enable predictor dropout (§3.1b). Roll each candidate R = 4–8 times in one batched call. Score by the mean cost, optionally adding κ·std for risk aversion.
- *Why.*
  - The predictor was trained with dropout 0.1, so MC dropout is a free approximate ensemble.
  - Averaging de-noises the scorer, which helps ideas 6 and 13. Pessimism (PETS, EpicWorldModel, Dream-MPC) avoids chunks the model is unsure about.
  - **Caution:** 2026 work shows uncertainty penalties can backfire, and latent epistemic estimates are biased. Test κ = 0 (mean only) first.
- *Test.* `mc_dropout` R ∈ {0, 4, 8}, κ ∈ {0, 0.5, 1}.
- *Refs.* Gal & Ghahramani; PETS; LeWM; EpicWorldModel; "Learning from World Feedback"; "Biased Dreams".

**15. Noise-space CEM / MPPI with sphere projection** · M · ++
- *Change.*
  - Make the initial noise ε ∈ R^{16×125} the decision variable, decoded deterministically with `_denoise`. Iteration 1 is today's best-of-N.
  - Then refit a diagonal Gaussian to the elite noises, sample, rescale each token's noise to norm √125, decode, and score.
  - Run 2–3 iterations with 4–8 elites.
- *Why.*
  - LeWAM is our architecture class (JEPA world model plus rectified-flow policy head). Noise-space CEM / MPPI / CMA-ES improved Lift and Square over the policy alone. Projection was decisive (Can 78.8% vs 23.0%), and action-space MPC collapsed to 3–8%.
  - Searching noise keeps actions on the policy's support. That matters because our WM only saw on-policy chunks during finetuning.
- *Test.* `noise_cem_iters` ∈ {0, 1, 2}, elites ∈ {4, 8}, samples per iteration ∈ {16, 32}. Always combine with idea 13.
- *Risk.* LeWAM saw planning hurt where the policy was already strong. Elite saturation means more iterations are not always better.
- *Refs.* LeWAM; Golden Ticket; DSRL; Ma et al. 2025.

**16. SDEdit-style elite refinement (partial re-noising)** · M · +
- *Change.*
  - Re-noise the top-k chunks: x = (1 − t*)a + t*ε′, with t* ∈ [0.2, 0.5].
  - Denoise back with ⌈t*·n_steps⌉ Euler steps, score, keep the best. Run 1–2 rounds.
  - Optionally re-noise only the later blocks (per-token t0), which keeps the near-term actions.
- *Why.*
  - It produces local, policy-like variations around good candidates: CEM's elite resampling, on the diffusion manifold.
  - Precedents: search over paths (Ma et al.), cycling top particles back to intermediate noise, and IPR's re-noising.
  - DF's mixed-noise training is exactly the backbone IPR needs.
- *Test.* t* ∈ {0.2, 0.35, 0.5}, rounds ∈ {1, 2}, k ∈ {2, 4}.
- *Refs.* SDEdit; Ma et al.; IPR; FK steering.

**17. Normalized (whitened) latent metric for switching and cost** · S · +0
- *Change.* Compute distances as ‖(z − g)/σ‖² with σ = `self.z_std`, which ships inside the policy checkpoint and needs no extra data. Variants: a diagonal σ from the episode's subgoal latents; cosine distance.
- *Why.*
  - Euclidean latent cost implicitly fixes a state metric, and that metric can misalign with the task (Anisotropic representations). What counts is ordering, not information content (Decision-metric).
  - Our dimensions span about 14× in variance, so raw L2 is dominated by a few of them.
- *Test.* `metric` ∈ {"l2", "zscore", "cosine"}, offline proxy first. (What Drives Success found L2 beats L1, so keep the squared form.)
- *Risk.* z-scoring amplifies low-variance, possibly noisy, dimensions.
- *Refs.* Anisotropic representations; Decision-metric alignment; What Drives Success.

**18. Planning context W_p = 2, or a context ensemble** · S · 0 / +
- *Change.* Roll out from the last 2 real latents instead of 3. Or roll with C ∈ {1, 2, 3}, average, and use the disagreement as an uncertainty signal for ideas 13 and 14.
- *Why.* What Drives Success found W_p = 2 gives the best planning success and is cheaper, with a big gap between context 1 and 2.
- *Test.* `wm_context` ∈ {2, 3}; `context_ensemble` on/off.
- *Refs.* What Drives Success.

**19. Hysteresis and dwell for every switching rule** · S · +0
- *Change.* A switching condition (ideas 1–4, 9, 12) must hold for K consecutive calls (K = 2–3), with a margin Δ (e.g. ≥ 10% closer), before the index moves.
- *Why.* Latent distances flicker (bf16 encoding, contact transients). ReGuide advances a phase only after a margin has held for K consecutive steps.
- *Test.* K ∈ {1, 2, 3}, Δ ∈ {0, 0.1}.
- *Refs.* ReGuide.

**20. Adaptive test-time compute from candidate dispersion** · M · ±
- *Change.* Measure the spread of the candidates' predicted end latents.
  - Low spread (candidates agree): use fewer candidates and/or execute 2 blocks.
  - High spread (a mode split: which box, grasp, place): double the candidates and enable ideas 15 or 16.
- *Why.*
  - ELASTIC finds sampling compute pays off at grasps, placements and mode splits.
  - The audit found most states have no outcome variation among candidates.
  - This is a training-free version of ELASTIC's allocation.
- *Test.* Set thresholds from dispersion quantiles in the trace. Compare at matched average compute.
- *Refs.* ELASTIC; test-time search audit; Sentinel.

**21. Stuck → exploration boost** · S–M · + (local)
- *Change.* When idea 3's no-progress signal fires (before forcing a skip), boost exploration for a few blocks: 4× candidates, a slightly higher noise scale, and/or idea 15.
- *Why.* If no sampled candidate makes progress, a more diverse proposal set may contain one that does. RoboMonkey shows sample scaling; Flash-BoN shows diversity drives the best-of-N gain.
- *Test.* `stuck_boost` on/off; boost size.
- *Refs.* RoboMonkey; Flash-BoN; ELASTIC.

**22. One-chunk viability lookahead** · M · ++
- *Change.*
  - For the top-K candidates (K ≈ 4), z-score the predicted end latent and sample M = 1–2 continuation chunks from the policy.
  - Roll the continuations from the last 3 predicted latents.
  - Add γ × (best continuation cost toward the next target) to each candidate's cost.
- *Why.*
  - A chunk can end near g_m in a configuration that can't continue, such as a poor grasp.
  - DF's Monte Carlo Guidance evaluates the present with sampled futures. CAST-V scores viability with model rollouts, and reports M = 1 is nearly enough. LeWAM decodes multi-chunk plans at imagined states.
- *Test.* K ∈ {2, 4}, M ∈ {1, 2}, γ ∈ {0.3, 0.5}.
- *Risk.* The WM was finetuned on 16-block rollouts, so beyond that is untested. Conditioning the policy on a (blurry) predicted latent is mildly out-of-distribution.
- *Refs.* Diffusion Forcing; CAST; LeWAM.

**23. Pyramid noise schedule (config only)** · S · 0
- *Change.* `lag` ∈ {0, 1, 2}, already supported by `DFPolicy.sample`.
- *Why.* DF's pyramid schedule denoises near-term blocks first, so the executed first block is sharpened conditioned on a more uncertain future. That fits executing only block 1.
- *Refs.* Diffusion Forcing.

**24. Diversity-engineered initial noise** · S · +0
- *Change.* Antithetic pairs (ε, −ε), Gram–Schmidt-orthogonalized noise across candidates, or quasi-random normals.
- *Why.* At fixed N, better mode coverage means more selection headroom. Flash-BoN reports diversity–gain correlation r ≈ 0.75; Kaleido Diffusion (Gu et al.) targets sample diversity directly.
- *Test.* `noise_scheme` ∈ {"iid", "antithetic", "orthogonal"}.
- *Refs.* Flash-BoN; Kaleido Diffusion.

**25. CUDA graphs / `torch.compile` the planner** · M · −− (faster)
- *Change.* In `__init__`, warm up and capture static-shape graphs for N-candidate sampling and the N-way rollout, via `torch.cuda.graphs` or `torch.compile(mode="reduce-overhead")`. Pad the context to 3 frames at episode start.
- *Why.*
  - Per-step compute is many tiny kernels, so latency is launch-bound. The authors' agent time was 324–727 s per episode.
  - Faster steps convert into more candidates or iterations. That matters if the competition announces compute limits.
- *Test.* `agent_s` at a fixed config, plus a bit-for-bit (or tolerance) output check.
- *Refs.* Engineering.

### Tier 3 — worth testing if time allows

**26. Fewer Euler steps, more candidates** · S (config) · −
- `n_steps` ∈ {5, 10, 20}, with N scaled inversely.
- Rectified flows are near-straight, so few-step sampling (Gu's BOOT and Normalizing Trajectory Models line) may keep quality while buying candidates.

**27. Inverse-dynamics self-consistency penalty** · S · +
- *Change.* For each predicted transition compute â_b = `self.encoder.inverse`(cat(ẑ_{b−1}, ẑ_b)). Add λ·mean_b ‖â_b − c_b‖².
- *Why.*
  - If the model's imagined transition doesn't match the commanded action, the prediction is likely off-distribution and exploitable.
  - Inverse-dynamics structure is what makes LeWM-family latents plannable (Strohm et al.; Decision-metric; H-JEPA), and our checkpoint ships the head.
  - Untested in the literature: a novel idea.
- *Risk.* The head was trained on encoder latents, not predicted ones.

**28. Progress-along-path (MPCC-style contouring) cost** · M · +0
- *Change.* Treat the subgoal latents as a polyline. Project each ẑ onto the nearest segment within a local window to get a progress coordinate s and a lateral error e. Use C = −α·s(ẑ_16) + β·Σ_b e(ẑ_b)². The fractional s also smooths ideas 2 and 5.
- *Why.* Model predictive contouring control makes progress along a path a decision variable, decoupled from time. That is our situation, where the agent is slower than the reference, and it removes hard switching altogether.
- *Risk.* Projecting onto far segments is meaningless (the stage-goal failure), so restrict to a window.
- *Refs.* Lam et al. 2010; Liniger et al. 2015.

**29. Min-over-trajectory "reach" cost plus continuation** · S · +0
- C = min_b ‖ẑ_b − g_m‖² + λ·‖ẑ_16 − g_{m+1}‖².
- It rewards passing through the subgoal at whatever pace, instead of exactly at block 16. A simpler cousin of idea 5, in the spirit of H-JEPA's intermediate subgoal terms.

**30. Reference-clock prior and maximum lag** · S · +0
- *Change.* The agent knows its call count k, so the reference index is j_ref = ⌊k/16⌋. Force-advance if the target lags j_ref by more than L_max intervals, or use j_ref as the prior inside idea 11.
- *Why.* V-JEPA 2-AC switches subgoals purely on time. We are on the clock for box 1 and about 1.5 intervals behind afterwards, so the clock is informative.
- *Risk.* It can push targets ahead of a slower agent.

**31. Backward-coherence penalty (soft commitment)** · S · +0
- C += λ·mean over b < p of ‖c_b − plan_prev_{b+1}‖² in normalized action space.
- This is BID's backward-coherence criterion. It is a softer alternative to idea 7's hard guard.

**32. Within-mode ensembling of the executed block** · S · +0
- Cluster the top-k candidates by predicted end latent. Within the best cluster, execute the softmax(−C/λ)-weighted average of their first blocks (raw, clipped).
- ACT's temporal ensembling and MPPI's weighted averaging reduce jitter. Averaging only within a mode avoids blending incompatible behaviours.

**33. Adaptive commit length (a READ / WRITE policy for replanning)** · M · −
- Execute 2–4 blocks when recent surprise is low and candidates agree. Replan every block when surprise or dispersion is high (ideas 12, 20).
- This is Gu et al.'s READ/WRITE framing of simultaneous translation, implemented with training-free thresholds. ELASTIC shows needs vary by phase.

**34. Proposal portfolio (TD-MPC2-style mixture)** · M · +
- At fixed total N, mix:
  - fresh samples;
  - the warm-start incumbent (idea 7);
  - prefix-inpainted candidates (idea 10);
  - SDEdit children of the last step's elites (idea 16);
  - structured noise (idea 24).
- TD-MPC2 mixes policy-prior samples into MPPI and warm-starts. A diversified portfolio hedges across failure modes.

**35. Sampling temperature: initial-noise scale mixtures** · S · +0
- Use noise scales s ∈ {0.8, 1.0, 1.2}, ideally *mixed* across candidates rather than shrunk globally.
- Lower s is mode-seeking; higher s explores. CAST ablates the noise scale.
- LeWAM's projection result warns that atypical noise norms hurt, so mixtures are safer.

**36. Draft-and-prune candidates (successive halving)** · M · ±
- Generate 4N candidates.
- At Euler step 4 of 10, take drafts â = x − t·v and score them with an 8-block rollout.
- Keep the top N, finish denoising, then score fully.
- This mirrors Flash-BoN's draft → staged selection → resume, and Matryoshka-style coarse-to-fine screening (Gu et al.).

**37. Particle (Feynman–Kac / SMC) steering through denoising** · L · ++
- Run N particles through the Euler steps. Every few steps, draft, roll the WM, weight particles by exp(−C/λ), and resample.
- FK steering and BFS/DFS diffusion search beat best-of-N at equal compute in many settings.

**38. World-model-gradient guidance during denoising** · L · +++
- *Change.* At each Euler step:
  - compute â = x − t·v;
  - run a differentiable rollout (§3.1c) and compute C;
  - nudge â ← â − η∇C, then convert back.
  - Only apply guidance inside a distance band (ReGuide's gate).
- *Why.* DynaGuide steers frozen diffusion policies with a separate dynamics model. ProgressVLA guides through a world model.
- *Expectations.* ReGuide's test-time-only gains were small (+0.01 to +0.05).
- *Risk.* Cost: backprop through 16 WM steps at every Euler step. Adversarial gradients.

### Tier 4 — exploratory, expensive, or needs a rules check

**39. Gradient refinement of the chosen chunk with a trust region (GPC-OPT)** · M · ++
- A few Adam steps on the best chunk's normalized actions, minimizing C under ‖a − a₀‖ ≤ δ, optionally followed by an SDEdit "projection" (idea 16).
- GPC reports gains from this.
- But DINO-WM, LeWM and What Drives Success all find gradient planners worse than CEM, and Mind the Gap and LeWAM show off-support exploitation. Keep δ small.

**40. Gradient optimization of the initial noise (D-Flow / ReNO-style)** · L · +++
- Backpropagate C through the 10-step decode and the WM rollout to ε. Re-project to the sphere after each step.
- This is idea 39 on the policy manifold: the noise-space analogue of LeWAM's search.

**41. Beam / tree search over chunk sequences** · L · +++
- Beam width B, depth 2–3 chunks (16–24 s). Children are policy samples at imagined latents; near-duplicate branches are merged.
- Precedents: Monte Carlo Tree Diffusion (DF-based), VLP's tree search, and Multiplex Thinking's branch-and-merge (Gu group).
- This could make the box-1 → box-2 transition visible to the planner.
- Risk: WM drift past 16 blocks (CIS 6280 L13).

**42. Policy-likelihood (typicality) prior** · M · ++
- Estimate log p_policy(a | z) per candidate with the flow's change-of-variables divergence integral (Hutchinson trace estimator, as in FFJORD). The cheap proxy is the noise norm ‖ε‖. Use C −= λ·log p.
- Gu's normalizing-flow line argues exact likelihood is valuable for interactive world models. A likelihood term discourages atypical chunks that exploit WM errors, the same intuition as LeWAM's projection.

**43. Golden-ticket noise bank** · L offline · 0 online · **[rules?]**
- Offline random search over fixed noise tensors on **train-split** scenes. Add the top 2–4 tickets as fixed extra candidates at every replan.
- Golden Ticket improved 46/51 tasks with one searched noise. Tickets vary wildly, and some steer toward failure. Using them as *candidates* lets the WM veto bad ones.
- Needs offline simulator search, so check the rules.

**44. Offline latent-dimension weighting (SWIM-inspired)** · M offline · 0 · **[rules?]**
- Fit per-dimension weights g on released latents (which coordinates predict actions or progress), and use √g ⊙ z in the cost.
- SWIM weights state coordinates by behavioural relevance and improves CEM. The Anisotropic paper shows the metric matters.
- This adds a fitted component, so check the rules.

**45. Inverse-dynamics "latent servoing" proposals** · S · +
- Set a one-block target ẑ* = z + (g_target − z)/16 and propose a* = `inverse`(z, ẑ*), repeated or with noise, as a few extra candidates. The WM scores them like any other.
- Latent Diffusion Planning extracts actions from planned latents with an inverse-dynamics model, and ours ships one trained on one-block transitions.
- Risk: off-manifold actions. Keep them strictly as extra candidates, clipped.

**46. Pixel-level box-layout cues for stage tracking** · M · 0 · **[rules?]**
- Color-threshold the three fixed box materials (blue, orange, purple) in both views of the current frame and the subgoal frames.
- Compare blob layouts: which boxes are in the goal zone, and their vertical order. Use this to confirm subgoal matches and to detect collapses.
- The 192-D latent is likely dominated by the large moving hand, while progress lives in a few box pixels (Strohm et al.'s regime).
- It uses only RGB, but check that it fits the spirit of the track.

**47. Streaming (rolling) DF denoising with particles** · L · −
- Carry a pyramid of partially denoised future blocks across calls: the front block is clean and later blocks noisier. Each call advances one level and appends a fresh noisy block. Keep several particles and select among them with the WM.
- Precedents: Streaming Diffusion Policy and DF's pyramid schedule amortize denoising. DART (Gu et al.) is the AR × diffusion analogue.
- This gives smoother chunks and frees compute for particles.

**48. Final-stage "finish and protect" mode** · S · +0
- Once the target is the last subgoal (the full-stack frame):
  - freeze the target;
  - allow regression on collapse (idea 12);
  - prefer chunks that release on the stack and then hover still.
- The metric counts the peak hands-free stack, so the release is what scores.
- Low value today (11/900 episodes reach the last subgoal), but it matters once Tier 1 works.

---

## 5. Suggested schedule

The course site lists final presentations on **Dec 1** and report + code on **Dec 14**. The page may be stale, so confirm on Canvas.

**By Nov 27.**
- Build the §3.1 helpers and the trace.
- Implement ideas 1–5 as knobs.
- Run config sweeps 6 and 8 on 300 scenes × seed 0.
- Run the headroom check (§3.2.1) to decide between scoring-side and proposal-side work.

**By Dec 1.**
- Take the best Tier-1 combination through the full 900-episode paired benchmark against the baseline.
- Pull failure analysis from traces and videos for the presentation.

**Dec 2–12.**
- Layer Tier 2 on the best combination, in this order: 7, 10, 12, 13, 14, then 15 or 16.
- Re-confirm on 900 episodes.
- Sync `agent.py` → the Colab cell, and keep defaults reproducing the baseline.
- Write up, with ablations as one row per knob.

---

## 6. References

*Numbers attributed to 2026 preprints come partly from partial reads of their arXiv HTML. Re-check against the PDFs before quoting them in the report.*

**Course and Jiatao Gu**
- CIS 6280 World Models (Fall 2026): https://www.cis.upenn.edu/~cis6280/
- Jiatao Gu homepage: https://jiataogu.me/
- "Towards Robust World Models", GRASP talk: https://www.grasp.upenn.edu/events/fall-2025-grasp-on-robotics-jiatao-gu/
- Non-Autoregressive Neural Machine Translation (noisy parallel decoding), ICLR 2018: https://arxiv.org/abs/1711.02281
- Levenshtein Transformer, NeurIPS 2019: https://arxiv.org/abs/1905.11006
- Insertion-based Decoding with Automatically Inferred Generation Order, TACL 2019: https://arxiv.org/abs/1902.01370
- Learning to Translate in Real-time with NMT, EACL 2017: https://arxiv.org/abs/1610.00388
- Monotonic Multihead Attention, ICLR 2020: https://arxiv.org/abs/1909.12406
- Trainable Greedy Decoding for NMT, EMNLP 2017: https://arxiv.org/abs/1702.02429
- DART, ICLR 2025: https://arxiv.org/abs/2410.08159
- Kaleido Diffusion, NeurIPS 2024: https://arxiv.org/abs/2405.21048
- STARFlow, NeurIPS 2025: https://arxiv.org/abs/2506.06276
- STARFlow-V, CVPR 2026: https://arxiv.org/abs/2511.20462
- Normalizing Trajectory Models, 2026: https://arxiv.org/abs/2605.08078
- Multiplex Thinking, 2026: https://arxiv.org/abs/2601.08808
- Matryoshka Diffusion Models, ICLR 2024: https://arxiv.org/abs/2310.15111
- BOOT, ICML 2024: https://arxiv.org/abs/2306.05544

**JEPA / latent world models and planning**
- LeWorldModel (LeWM): https://arxiv.org/abs/2603.19312
- DINO-WM: https://arxiv.org/abs/2411.04983
- V-JEPA 2 (V-JEPA 2-AC): https://arxiv.org/abs/2506.09985
- PLDM, Planning with Latent Dynamics Models: https://arxiv.org/abs/2502.14819
- Navigation World Models: https://arxiv.org/abs/2412.03572
- What Drives Success in Physical Planning with JEPA World Models?: https://arxiv.org/abs/2512.24497
- H-JEPA: https://arxiv.org/abs/2610.06805
- Mind the Gap, Hierarchical Planning in LeWorldModel: https://arxiv.org/abs/2607.12547
- Keeping JEPA World Models Plannable When Little of the Frame Moves: https://arxiv.org/abs/2610.03137
- Decision-Metric Alignment in Latent World Models: https://arxiv.org/abs/2608.18746
- Anisotropic Representations Improve Planning in JEPA World Models: https://arxiv.org/abs/2609.37441
- When More Planning Hurts: https://arxiv.org/abs/2610.37891
- SWIM: https://arxiv.org/abs/2610.36662
- Learning Goal-Conditioned Subgoals through Frozen Hierarchical World Models: https://arxiv.org/abs/2610.26179
- FF-JEPA: https://arxiv.org/abs/2606.09311
- EpicWorldModel: https://arxiv.org/abs/2610.05996

**Steering frozen policies and test-time selection**
- LeWAM, Diffusion-Steering-Based MPC with a JEPA World Action Model: https://arxiv.org/abs/2610.12407
- GPC, Generative Predictive Control: https://arxiv.org/abs/2502.00622
- Golden Ticket: https://arxiv.org/abs/2603.15757
- DSRL: https://arxiv.org/abs/2506.15799
- DynaGuide: https://arxiv.org/abs/2506.13922
- ReGuide: https://arxiv.org/abs/2606.28939
- ProgressVLA: https://arxiv.org/abs/2603.27670
- FOREWARN: https://arxiv.org/abs/2502.01828
- CAST, Certify Then Select: https://arxiv.org/abs/2611.19047
- Bidirectional Decoding (BID): https://arxiv.org/abs/2408.17355
- Real-Time Chunking (RTC): https://arxiv.org/abs/2506.07339
- RoboMonkey: https://arxiv.org/abs/2506.17811
- MG-Select: https://arxiv.org/abs/2510.05681
- ELASTIC: https://arxiv.org/abs/2606.31132
- When Does Test-Time Search Help World Action Models?: https://arxiv.org/abs/2610.41970
- Cited without links: V-GPS (Nakamoto et al. 2024); Latent Diffusion Planning (Xie et al. 2025); Dreamer 4 (Hafner et al. 2025); D-Flow (Ben-Hamu et al. 2024); ReNO (Eyring et al. 2024); VLP (Du et al. 2023).

**MPC, diffusion search, Diffusion Forcing**
- PlaNet: https://arxiv.org/abs/1811.04551
- iCEM: https://arxiv.org/abs/2008.06389
- TD-MPC2: https://arxiv.org/abs/2310.16828
- D-MPC: https://arxiv.org/abs/2410.05364
- Model-Based Diffusion: https://arxiv.org/abs/2407.01573
- MPPI: Williams et al., ICRA 2017 (cited without link)
- Inference-Time Scaling for Diffusion Models (Ma et al.): https://arxiv.org/abs/2501.09732
- FK steering: https://arxiv.org/abs/2501.06848
- Classical search for diffusion: https://arxiv.org/abs/2505.23614
- Flash-BoN: https://arxiv.org/abs/2607.04461
- IPR, Iterative Partial Refinement: https://arxiv.org/abs/2605.19317
- SDEdit: https://arxiv.org/abs/2108.01073
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- Monte Carlo Tree Diffusion: https://arxiv.org/abs/2502.07202
- Streaming Diffusion Policy: https://arxiv.org/abs/2406.04806
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT: https://arxiv.org/abs/2304.13705
- Diffuser: https://arxiv.org/abs/2205.09991

**Progress tracking, monitoring, uncertainty**
- REMORA: https://arxiv.org/abs/2610.39125
- Sentinel: https://arxiv.org/abs/2410.04640
- MC dropout: https://arxiv.org/abs/1506.02142
- PETS: https://arxiv.org/abs/1805.12114
- Dream-MPC: https://arxiv.org/abs/2605.04568
- Learning from World Feedback: https://arxiv.org/abs/2607.16591
- Biased Dreams: https://arxiv.org/abs/2604.25416
- TCC, Temporal Cycle-Consistency: https://arxiv.org/abs/1904.07846
- FFJORD: https://arxiv.org/abs/1810.01367
- MPCC (Liniger et al.): https://arxiv.org/abs/1711.07300
- Cited without links: pure pursuit (Coulter 1992); MPCC (Lam et al. 2010); HMM tutorial (Rabiner 1989).
