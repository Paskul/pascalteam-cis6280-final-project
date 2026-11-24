# notesForRunning — running the planning notebook locally

Verified 2026-11-23 on this machine (2× RTX A5000). No repo code was changed to get it running.

## Summary (2026-11-23)

**Results.**
- No idea beat the baseline in boxes stacked on new scenes. On 68 held-out scenes, the final planner scored 1.56 mean boxes vs 1.50 for the baseline (Δ +0.06 ± 0.15, not significant).
- Cell 7 now holds the final planner: the baseline + switching threshold τ = 0.5 + bf16 policy sampling.
  - **Speed:** 2.3× faster per replan (106 vs 241 ms).
  - **Stall fix:** τ = 0.5 removes the baseline's freeze on finished subgoals.
  - **Notebook scene:** 3 boxes vs 1 (a weak signal; bf16 alone also gets 3).
- **Hurt:** skip-ahead/pursuit tracking and the time-aligned path cost.
- **No help:** stall release, abstain, MC dropout, 64 candidates, warm start.
- Details: the table, "Final planner", "Findings".

**Next steps** (details in "Conclusions and what to try next"):
1. Spend the freed compute on a second imagined chunk (ideas 22/41) rather than more samples.
2. Detect when the policy picks a different box than the subgoal images show, and stop steering against it.
3. Judge ideas on several hundred episodes per variant: effects of 0.1–0.2 boxes are invisible at 32–68 scenes.
4. Fix the GPUs' "HW Power Brake Slowdown" (PSU / PCIe power), which slowed every run.

## How to run the notebook

1. **Free the GPUs.** Check `nvidia-smi`. If the Qwen LLM server is holding them, stop it with `systemctl --user stop qwen.service`. It comes back with `systemctl --user start qwen.service` and on every reboot.
2. **One-time setup (already done).** `bash scripts/setup.sh` builds `.venv` from the lock file and downloads the hand meshes. Then `.venv/bin/python -m pip install nbconvert ipykernel` so the notebook can run outside Colab.
3. **Edit cell 7, then run the notebook** from the repo root. It takes about 2 minutes:
   ```bash
   .venv/bin/jupyter nbconvert --to notebook --execute assignment/planning/colab.ipynb \
     --output-dir runs/<run-name> --output colab.executed.ipynb --ExecutePreprocessor.timeout=1800
   ```
   The executed copy, with all outputs and both videos, goes to `runs/<run-name>/` (git-ignored), e.g. `runs/idea01-skip-ahead/`. The timeout is per cell, in seconds; raise it for slower planners. The baseline run is in `runs/baseline-notebook/`.
4. **Read the result**, which is the table from cell 8:
   ```bash
   jq -j '.cells[8].outputs[] | select(.output_type=="stream") | .text | if type=="array" then join("") else . end' \
     runs/<run-name>/colab.executed.ipynb | tail -4
   ```

**Reading results**
- The notebook runs one scene (100038) twice: once with the policy alone ("no planning"), then with our planner.
- Runs on this machine are deterministic **when the GPU is not shared**. Unchanged code gives the identical result every time, so a different result comes from the code change. GPU 0 and GPU 1 give identical episodes. But while other GPU jobs run at the same time, an occasional episode takes a different path (seen on 2 of 8 checked scenes), so run the notebook with GPU 0 otherwise idle.
- Baseline here: no planning stacks **3** boxes and the planner stacks **1**. The notebook text says 1 and 2; those numbers came from the authors' GPUs. Rendered images differ by one brightness level on under 1% of pixels between GPUs, and that is enough to send an episode down a different path.
- One scene is one noisy sample, so a change can gain or lose a box by luck. Treat a single-scene win as promising, not proven.

## Dev harness: scoring a planner on many scenes

One scene can't separate ideas: the per-episode spread is about ±0.9 boxes. So every idea is also scored on validation scenes with a harness that runs **cell 7's code verbatim with a copy of cell 8's run loop** (same constructor arguments, environment settings and seeds). It changes no repo file. Everything lives in the git-ignored `runs/dev/`:

- `harness/nbeval.py`: the evaluator. `--source notebook` runs the current cell 7, `--source baseline` the committed (git HEAD) cell 7, or a `.py` file with a cell-7 variant. `--cfg '{...}'` adds keyword arguments to cell 8's constructor call. It writes one JSON line per episode to `results/<name>.jsonl`, and is resumable.
  ```bash
  cd runs/dev && ../../.venv/bin/python harness/nbeval.py --source notebook --first 0 --scenes 32 \
    --gpus 0,1 --workers-per-gpu 1 --out results/<name>.jsonl
  ```
- `harness/compare.py results/base_plan.jsonl results/<name>.jsonl`: a paired comparison on the common episodes. It prints the mean, Δ ± SE, wins/ties/losses, ≥k rates and P(2|1).
- `harness/stop_run.sh <name>.jsonl`: stops a run **including its worker processes**. Killing only the main process leaves orphaned workers that keep using the GPU.
- `harness/make_cell7.py` + `harness/sync_cell7.py` + `harness/run_nb.sh <run-name>`: build the cell-7 source for one idea, write it into the notebook (asserting that only cell 7 changes), and run the notebook into `runs/<run-name>/`.
- `harness/nb_batch.sh <spec>` / `harness/nb_single.sh <run-name> <cell.py>`: run the notebook for a list of ideas or one cell file. They freeze the dev workers on GPU 0 during each run so the notebook result is deterministic. `harness/nb_results.py <run-name>...` prints the table numbers.
- `harness/equiv_check.py A.py B.py '{cfg}'` feeds identical frames to two cell versions and reports the first differing step. `harness/time_step.py` times `agent.step()`.
- Agent versions: `harness/agent_v0.py`…`agent_v5.py` are the dev classes (all knobs). `cells/final_A.py` is exactly what is in cell 7 now. `cells/<run-name>.py` is the cell used for each notebook row.
- Validation: the harness reproduces the notebook exactly on scene 100038 (planner 1 box at step 297; no planning 3 boxes at steps 312/593/1133).
- Dev set: the first 32 validation scenes (sample seed 0) for screens. The final planner is checked on validation scenes 33–100, which no screen used, so choosing the winner can't flatter its score. All 100 are the scenes in the notebook's goals file. The paired SE is about ±0.25 boxes on 32 scenes and ±0.16 on 68.
- Speed on this machine: a baseline planner episode takes about 70 s of GPU time. Both GPUs report "HW Power Brake Slowdown" since boot (SM clock ~800–1600 MHz of 2100), so everything is slower than it should be. That's a hardware or PSU signal worth checking. CUDA MPS made throughput worse here, so each GPU is simply time-shared.

## The only thing we change

**Only `GoalPlanningAgent` in cell 7 of `assignment/planning/colab.ipynb`. Nothing else may be changed.**

- Keep its interface:
  - `reset(goal_frames, init_frame, seed)`
  - `step(frame)`, returning a `(5, 25)` float32 array in `[-1, 1]`
  - the `self.index` attribute, which cell 8 reads
- It may only use the camera frames and subgoal images it is given, never the simulator's state.
- Everything else stays fixed:
  - Every other cell, including cell 8's run loop and the arguments it passes (`candidates=16, generate=16, execute=1, tau=0.25`). Make setting changes inside the class, e.g. in `__init__`.
  - The downloaded world model, policy, and subgoal files.
  - The rest of the repo, including `assignment/training/planning/agent.py`. Its `LatentPolicyAgent` is the no-planning baseline that our class inherits from. Cell 7 defines its own copy of the planner, so edits to `agent.py` wouldn't reach the notebook anyway.

## What was tried already and its result

Add one row per idea run; idea numbers refer to `ideasToTest.md`. The no-planning run doesn't use our class, so it should always stay at 3 boxes. If it changes, something other than the planner changed.

The three number columns are the notebook run (scene 100038). That scene says little about the ideas: bf16 alone already gives 3 boxes there. The dev-set result in Notes is what separates ideas. Dev result files are in `runs/dev/results/`: `base_plan`, `base_noplan`, `s1_bf16_only` (control), `s2_pursuit_final`, `s3_pursuit_path`, `s4_tau05`, `s5_tau05_path`, `s6_tau05_warm`, `s12_warm`, `s7b_stall`, `s8b_abstain`, `s9b_mc4`, `s10_n64`, plus `ho_final_A` / `ho_bf16` for the held-out check.

| # | Idea | What changed in `GoalPlanningAgent` | Planner boxes | No-planning boxes | Planner min | Notes |
|---|---|---|---|---|---|---|
| 0 | Baseline | Nothing | 1 | 3 | 1.2 | 2026-11-23. Reference for every later row. Dev: 1.38 mean boxes on the 32 screening scenes (no planning: 1.06). |
| eng. | bf16 policy sampler | Run `policy.sample` under bf16 autocast (the baseline samples in float32) | 3 | 3 | 1.0 | `runs/idea-bf16-sampler/`. Sampling is ~4× faster (16 candidates: 209 → 54 ms per replan), with a 1.3% relative difference in the sampled actions. Dev: Δ +0.22 ± 0.19 (32 scenes, 13/11/8); held-out Δ +0.06 ± 0.15 (68 scenes). Neutral on boxes. **Kept** for speed. It is also the control for every later row. |
| 1–2, 30 | Pursuit / skip-ahead tracking + terminal cost | Localize on the subgoal path (window, ×2 rate limit, clock cap), aim one interval ahead | 0 | 3 | 0.9 | `runs/idea02-pursuit-final/`. Dev: Δ −1.00 ± 0.58 vs baseline (6 scenes, stopped early). The target raced ahead on look-alike subgoals; in 100038 it reached subgoal 14 although box 1 was never placed. |
| 1–2, 5 | Pursuit tracking + time-aligned path cost | As above, but score all 16 imagined latents against the interpolated subgoal path | 2 | 3 | 0.9 | `runs/idea02-pursuit-path/`. Dev: Δ −0.80 ± 0.58 (5 scenes, stopped early). |
| 4 | **τ = 0.5 (final planner)** | Switching threshold 0.5 instead of cell 8's 0.25, + bf16 sampler | **3** | 3 | **0.9** | `runs/final-tau05-bf16/`, now in cell 7. Dev: Δ **+0.28 ± 0.27** vs baseline (32 scenes, 14/9/9); +0.06 ± 0.22 vs the bf16 control. Episodes reaching ≥1/≥2/≥3/4 boxes: 29/13/9/2, vs 27/13/4/0 for the baseline and 30/14/7/0 for the bf16 control. **Held-out (68 new scenes): Δ +0.06 ± 0.15 vs baseline, +0.00 ± 0.13 vs bf16 control**, so the third-box gain did not replicate. Held-out result in "Final planner" below. |
| 4 + 5 | τ = 0.5 + path cost | Final planner + time-aligned path cost | 2 | 3 | 0.9 | `runs/idea05-tau05-path/`. Dev: Δ −0.69 ± 0.33 vs bf16 control (13 scenes, 2/4/7). **Harmful.** |
| 4 + 7 | τ = 0.5 + warm start | Final planner + incumbent plan (Diffusion-Forcing inpainted tail, guard 10%) | 2 | 3 | 1.2 | `runs/idea07-tau05-warm/`. Dev: Δ −0.22 ± 0.46 vs bf16 control (9 scenes, stopped early). |
| 7 | Warm start | bf16 + incumbent plan with a 10% guard, baseline tracker | 2 | 3 | 1.2 | `runs/idea07-warm/`. Dev: Δ −0.19 ± 0.26 vs bf16 control (16 scenes, 4/5/7); it read +0.22 at 9 scenes. Not helpful. |
| 3 | Stall release | bf16 + advance after 24 calls farther than 1.3·gap from the subgoal | 3 | 3 | 0.9 | `runs/idea03-stall-release/`. Dev: Δ −0.08 ± 0.08 vs bf16 control (12 scenes, 0/11/1). It fired in 7/12 episodes, but only around steps 975–1225, too late to add a box. |
| 13 | Abstain when the target is out of reach | bf16 + execute the policy's own sample after 24 calls farther than 1.5·gap | 3 | 3 | 1.0 | `runs/idea13-abstain/`. Dev: Δ −0.08 ± 0.08 vs bf16 control (12 scenes, 0/11/1). It fired in 3/12, late. |
| 14 | MC-dropout scoring | bf16 + mean cost over 4 dropout rollouts per candidate | 2 | 3 | 1.0 | `runs/idea14-mc-dropout4/`. Dev: Δ −0.44 ± 0.25 vs bf16 control (18 scenes, 2/8/8). Not helpful. |
| 6 | 64 candidates | bf16 + 64 candidates (about the baseline's time per replan) | 1 | 3 | 1.4 | `runs/idea06-64-candidates/`. Dev: Δ −0.25 ± 0.37 vs bf16 control (12 scenes, 5/2/5; stopped). No sign of a gain. |
| 17 | z-scored latent metric | (offline only) | – | – | – | Replayed on 41 closed-loop traces, it made subgoal tracking no more box-aware than raw L2 (τ = 0.5: 6.5% vs 2.0% "behind" calls), so it was not run. |
| 25 | CUDA graphs / MPS | (benchmark only) | – | – | – | Graphs gave 1.1× (sampler) and 1.5× (rollout) on this throttled GPU; MPS lowered throughput. Not used. |

## Final planner (now in cell 7)

`GoalPlanningAgent` in cell 7 is the baseline loop plus two changes. Both are keyword arguments with the baseline value documented in the docstring:

1. **`switch_tau=0.5`**: switch to the next subgoal once `||z − g_m||² < 0.5·gap_m` (the baseline uses cell 8's 0.25).
2. **`bf16_sampling=True`**: the policy sampler runs under bf16 autocast, about 4× faster. On GPUs without native bf16 (e.g. a Colab T4, where autocast would mean float16) it falls back to float32.

**Compute:** `agent.step()` takes **106 ms per replan vs 241 ms for the baseline** (2.3× faster). This is the median over 55 replans on the same frames, on an otherwise idle A5000 (`runs/dev/harness/time_step.py`). Adding 4-way MC dropout would cost 117 ms.

The cell also contains the other tested mechanisms as switches that are off by default: stall release, path cost, warm start, `n_candidates`, MC dropout. Changing their defaults in `__init__` reproduces those rows of the table (cell 8's call can't pass them). With every switch at its baseline value, the class gives action-for-action the same episode as the committed baseline (checked on 40 closed-loop blocks of scene 100038: identical md5 of the actions).

**Held-out check (68 scenes never used for screening: validation scenes 33–100, sample seed 0, finished 2026-11-23 21:29):**

| Planner | Mean boxes | Δ vs baseline (paired) | ≥1 / ≥2 / ≥3 / 4 boxes |
|---|---|---|---|
| Baseline (committed cell 7) | 1.50 | – | 59 / 32 / 9 / 2 |
| bf16 sampler only | 1.56 | +0.06 ± 0.15 (20/26/22) | 60 / 33 / 11 / 2 |
| **Final planner (τ = 0.5 + bf16)** | **1.56** | **+0.06 ± 0.15** (23/21/24), 95% CI −0.24 … +0.36 | 60 / 34 / 10 / 2 |

Final vs the bf16-only control: +0.00 ± 0.13 (19/30/19). Over all 100 dev scenes (screen + held-out; the screen half was used to pick the winner, so this is slightly optimistic): final 1.59 vs baseline 1.46, Δ +0.13 ± 0.14. Files: `runs/dev/results/ho_final_A.jsonl`, `ho_bf16.jsonl`, `base_plan.jsonl`.

**Verdict.**
- The final planner stacks **the same number of boxes as the baseline within noise**, and replans **2.3× faster**.
- The third-box effect that τ = 0.5 showed on the 32 screening scenes (9 vs 4 episodes with three boxes) **did not replicate** on the held-out scenes (10 vs 9).
- τ = 0.5 does remove the measured index freeze. The subgoal index ends 55% of the way through the sequence on average, vs 36% for the baseline. That just doesn't turn into more boxes at this sample size.
- No idea tried here produced a statistically confirmed increase in boxes. The baseline planner is already about +0.3 boxes better than no planning on these scenes (+0.31 ± 0.19, 32 scenes), and the remaining failures (box 2+, box-order conflicts) are not ones the subgoal rule, cost shape or candidate count could fix.

## Findings

Dev numbers are paired against the baseline planner on the same scenes and seeds. "Δ" is the mean difference in boxes, ± its SE. One run per scene gives a paired SE of about ±0.25 at 32 scenes. Readings swing a lot below ~30 scenes: τ = 0.5 read −0.46 at 13 scenes, +0.04 at 25 and +0.28 at 32. Every bf16 variant is also compared with the bf16-only control, which shares its sampler numerics and therefore diverges from it later (more ties, smaller SE).

**1. Why the baseline stalls (measured on traced runs, not assumed).** The subgoal index freezes because the policy never gets within `τ·gap` of the scripted frame around a placement. In every traced stall, the closest the agent got to the frozen subgoal was 1.4–2.6× the threshold. Example: scene 100057 places box 2 at step 693, but the index stays on subgoal 7 ("box 2 being lowered") for the remaining ~1,100 steps. The planner keeps steering back toward a configuration that is already in the past, and box 3 never comes. Consecutive-subgoal gaps vary about 9× (squared L2 of 15–136), so a fixed `τ = 0.25` is very strict in slow phases.

**2. τ = 0.5 removes the stall, but the gain in boxes did not hold up on new scenes.** Replayed on the baseline traces, τ = 0.5 cut the calls where the target sits behind an already-finished placement from 11% to 0%, while racing ahead of the real box count on only 3–4% of calls. In closed loop (32 scenes), boxes 1–2 are unchanged: ≥1 box in 29 episodes and ≥2 in 13, vs 27 and 13 for the baseline and 30 and 14 for the bf16 control. On the screen, the gain was the third and fourth box: 9 episodes reached three boxes (baseline 4, bf16 control 7), and 2 stacked all four (both references: none). On the 68 held-out scenes this disappeared: 10 vs 9 episodes with three boxes, Δ +0.06 ± 0.15 overall. So the screen result was most likely noise, or an effect too small to measure here. Stall release and abstain target the same stall but fire only after the agent has hovered far from the subgoal for 24 calls, which happens around steps 975–1225: too late to add a box.

**3. The latent is dominated by the hand, so lenient tracking races ahead.** The hand goes through similar poses for every box. For half of the subgoals, a non-adjacent subgoal is closer than 0.3× the subgoal's own gap. Retrying box 1 therefore looks like "doing box 2" in latent space. Pursuit / skip-ahead trackers (ideas 1–2, 30) followed those look-alikes: in scene 100038 the target reached subgoal 14 although box 1 was never placed (Δ −1.00 ± 0.58). τ = 0.5 stays safe because it only ever checks the *next* subgoal, never a window ahead.

**4. Changing the cost shape hurt.** The time-aligned path cost (idea 5: all 16 imagined latents against the interpolated subgoal path) scored Δ −0.69 ± 0.33 vs the bf16 control. Likely reasons, not verified here: interpolated latents are not real states, and the extra terms dilute the one comparison that matters, where the chunk ends.

**5. More candidates, averaged scores and plan commitment did not show a gain.** With 64 candidates (affordable at the baseline's time per replan thanks to bf16): Δ −0.25 ± 0.37 (12 scenes). MC-dropout scoring with 4 rollouts: −0.44 ± 0.25 (18 scenes). A likely reason, not verified here: picking the best of many noisy world-model scores mostly picks the noise (the "winner's curse" warned about in `ideasToTest.md` §2.6). Warm start (committing to the previous plan unless a fresh candidate is 10% better) did not help either: −0.19 ± 0.26 (16 scenes).

**6. Box-order conflict, a failure mode no subgoal rule fixes.** In scene 100038 the reference's box 2 is the blue box, but the policy goes for the other orange box. The baseline planner then spends ~1,000 steps scoring candidates against "blue box being placed", fighting the policy's own choice. The policy alone stacks 3 boxes there; the baseline planner stacks 1. The subgoal images encode one box order, and the planner has no way to accept a different but equally good one.

**7. Compute.**
- The bf16 sampler is the only free speedup: 4× on sampling and 2.3× per replan overall (106 vs 241 ms). On the notebook scene the whole planner episode takes 0.9 vs 1.2 min, because physics and rendering don't change.
- CUDA graphs gave 1.1–1.5× on this throttled GPU, so the work is compute-bound, not launch-bound.
- MPS reduced total throughput.
- Both A5000s report "HW Power Brake Slowdown" since boot, which caps the clocks. Fixing that (PSU / PCIe power) would speed up every run on this machine.

## Conclusions and what to try next

**What ended up working.**
- **bf16 policy sampling:** replanning is 2.3× faster (106 vs 241 ms), with no measurable change in boxes (held-out Δ +0.06 ± 0.15). It is free compute for whatever comes next.
- **τ = 0.5 subgoal switching:** it fixes a real, measured pathology. After a placement the baseline's subgoal index froze on a finished subgoal for the rest of the episode; with τ = 0.5 the index keeps advancing. On new scenes this did not translate into more boxes (+0.00 ± 0.13 vs its bf16 control).

**What did not.** Lenient or skip-ahead tracking, path costs, stall release, abstaining, MC dropout, 64 candidates and warm start (see the table). The likely reasons, from the traces:
1. The latent is dominated by the hand, so trackers can't tell "retrying box 1" from "doing box 2".
2. The remaining failures are physical: placing box 2 and later boxes, and box-order conflicts with the scripted subgoals. Neither a subgoal rule nor a different ranking of the same 16 policy samples changes those much.

**Next steps if more time is spent on this track:**
1. **Use the freed compute for lookahead, not more samples.** A second chunk imagined from the end of each first chunk (idea 22) or a small beam over chunk sequences (idea 41) would let the planner see the box-1 → box-2 transition. The risk is world-model drift past 16 blocks.
2. **Handle box-order conflicts.** Detect when the policy is working on a different box than the subgoal images show (the subgoal stays far while the policy's samples agree with each other), and stop steering until they agree again. "Abstain" was a first, too-late version of this.
3. **Measure before tuning.** At ±0.15 boxes per 68 scenes, effects of 0.1–0.2 are invisible. Confirming one would take several hundred episodes per arm (e.g. 100 scenes × 3 seeds: about 1.5 h per bf16 arm and 3 h for the float32 baseline on this machine).
4. **Fix the GPUs' "HW Power Brake Slowdown"** (PSU / PCIe power). The SM clock sat at roughly 40–80% of its 2100 MHz maximum throughout these runs, which slowed every experiment.
