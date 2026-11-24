# Four-box stacking — 24 new experiments for `GoalPlanningAgent`

CIS 6280 World Models · 24 November 2026 · Repository revision `79d1893b`

**Start with experiments 49 and 50.** Experiment 49 asks whether the planner selects a good first move or a move with one lucky imagined ending. Experiment 50 asks whether the carried box is centered over the box that must support it. They test different explanations of the second-box failure. Run them separately. If video shows a successful pickup of a different box from the one in the reference, use experiment 51 on that failure.

| Run | What changes | What would justify more runs |
|---|---|---|
| **49 · First move under several futures** | Give the same first block several plausible continuations. Rank the first block by its average result. | The selected move changes on ambiguous grasps, and more scenes reach a second box. |
| **50 · Center the carried box** | Use both live camera views to rank wrist motions near placement. | The box arrives closer to the center of its support and more releases add a box. |
| **51 · Accept an alternate box order** | When the hand carries a different box, score against an image of that placement with the two box colors exchanged. | More correctly identified alternate pickups become placements, with little change on matching-order scenes. |
| **52 · Check action clipping** | Compare what the model scores with the bounded command the simulator receives. | Clipping changes enough winners to explain a measurable difference in prediction quality. This check can precede the GPU runs. |
| **53 · Remember a failed grasp** | Record where an empty grasp came from. Try a different approach on the next attempt at that box. | Fewer identical retries and a higher chance of lifting the box on the next attempt. |

Numbers 49–72 continue the original 48-item list. All 24 are proposed experiments. The placement results below come from the prior runs. The image check below uses the repository's demonstration video.

Jump to [the previous results](#what-the-current-result-says), [the cell interface](#what-cell-7-can-see-and-change), [the run commands and evaluation plan](#how-to-run-the-next-round), or [the research sources](#literature-and-the-specific-ideas-it-supports).

## What the current result says

**There are two starting versions.** The reviewed Git commit adds `ideasToTest.md` and `notesForRunning.md`. Its notebook still defines the original `tau=0.25` planner and samples the policy in float32. The notes describe a local cell with `switch_tau=0.5` and native bf16 policy sampling. They identify `runs/dev/cells/final_A.py` as its saved copy. Git ignores `runs/`. Use that local cell as the Round 2 control. On a fresh checkout, reproduce the two settings inside cell 7 and check the result before comparing variants. The local episode files and final cell were unavailable in the commit. [Commit comparison] · [Notebook] · [Run notes]

| Evidence | Result | What it permits us to conclude |
|---|---|---|
| Course benchmark, 300 scenes × 3 sampler seeds | Planning averaged **1.41** peak boxes. Policy alone averaged **1.27**. With planning, 85.2% reached one box, 41.4% two, 11.8% three, and 2.4% four. | Almost every episode registered a grasp (99.1%). Only about 49% of episodes that reached one box reached two. First grasp alone is a poor target for this round. |
| Local notebook scene `100038` | Policy alone **3**, original planner **1**, bf16 only **3**, local final **3**. The notes identify an orange pickup when the reference calls for blue. | Use the video to inspect an order conflict. This scene cannot select a winner. |
| First 32 local validation scenes, sampler seed 0 | Policy alone averaged **1.06** and the original planner averaged **1.38**. Local final gained **+0.28 ± 0.27** over the original and **+0.06 ± 0.22** over bf16 alone. Nine scenes reached three boxes, versus four for the original. | The apparent third-box improvement needed replication. |
| Next 68 local scenes, sampler seed 0 | Original **1.50** (59/32/9/2 reached ≥1/≥2/≥3/4). Bf16 only **1.56** (60/33/11/2). Local final **1.56** (60/34/10/2). | Local final minus bf16 only was **0.00 ± 0.13** boxes. Local final minus original was **+0.06 ± 0.15**, with a reported 95% interval of **−0.24 to +0.36**. |
| A5000 timing and index progress | Median `step()` fell from **241 to 106 ms** over 55 replans. The final target index advanced from 36% to 55% of the sequence on average. | The speed gain gives a budget for a second calculation. Advancing the index by itself did not establish a gain in boxes. |

The `±` values above and below are paired standard errors. The notes report one example of an index error: scene `100057` placed box two at step 693, while the target stayed at the lowering phase. They also report that in `100038` an aggressive tracker reached subgoal 14 without placing box one. These failures call for different interventions. [Run notes] · [First idea file] · [Course benchmark]

| Previous work | Recorded result | Consequence for Round 2 |
|---|---|---|
| Pursuit and skip-ahead, original ideas 1/2/30 | Terminal and path versions lost 1.00 and 0.80 boxes per episode against the original on early screens of six and five scenes. | Keep the ordinary sequential index while testing object identity or a new score. |
| Interpolated path cost, idea 5 | At `tau=0.5`, **−0.69 ± 0.33** against bf16 only on 13 scenes. | Use a measured endpoint or physical relation. A hand can take several valid paths to it. |
| Warm start, idea 7 | With the original tracker, **−0.19 ± 0.26** against bf16 only on 16 scenes. The `tau=0.5` variant also had a negative early estimate. | Test which observed outcome supports a plan before reusing it. |
| 24-call stall release and abstention, ideas 3/13 | Both **−0.08 ± 0.08** on 12 scenes against bf16 only. Triggers arrived around steps 975–1225. | A visible missed lift or wrong pickup supplies an earlier trigger. |
| Four dropout rollouts and 64 candidates, ideas 14/6 | **−0.44 ± 0.25** on 18 scenes and **−0.25 ± 0.37** on 12 scenes, respectively, against bf16 only. | Spend extra calls on a specific question about an action. These small screens leave uncertainty about effect size. |
| Whitening and speed work, ideas 17/25 | Whitening was checked offline on 41 traces. CUDA graphs and MPS were benchmarked. Native bf16 was retained. | Use bf16 on the A5000 for both arms. Retain the float32 sampler fallback on hardware without native bf16. |

The original list also proposed two-chunk lookahead (22), tree search (41), RGB stage tracking (46), and a final-stage protection mode (48). They are useful comparison arms. Each experiment below specifies a different decision, measurement, or action. [First idea file] · [Run notes]

## What cell 7 can see and change

- The single editable unit is the **`GoalPlanningAgent` class in cell 7** of `assignment/planning/colab.ipynb`. Put helper methods and switches inside it. Cell 8 constructs it with `candidates=16, generate=16, execute=1, tau=0.25`. Consume new keywords before calling the parent constructor. Support the no-argument `reset()` that the parent calls during construction.
- `reset(goal_frames, init_frame, seed=None)` receives this scene's images and seeds the sampler. Clear all episode memory here. Keep `self.index` a valid integer for the video. `step(frame)` receives a `uint8` image of shape `(128,256,3)`. It returns a `float32` array of shape `(5,25)` in `[-1,1]`. The first three channels command XYZ wrist target increments. The other 22 command finger joint target increments. A raw zero increment holds the previous target. It does not open the hand.
- The frozen Diffusion Forcing policy uses the current standardized 192-value image latent. It draws 16 candidate chunks. Each chunk contains 16 blocks of five actions, or eight seconds. The frozen world model uses up to three actual frame latents and the intervening issued actions. It predicts **latents**. Pixel calculations below use actual live or goal images. The model supplies no future RGB, hand contacts, box identity, or camera depth.
- The baseline advances the target while `||z − g[m]||² < tau × ||g[m] − g[m−1]||²`. At `m=0`, the previous goal is the initial frame. It picks the candidate with the smallest terminal squared distance to the active goal. Only its first five actions reach the simulator before the next observation. A full episode lasts 1,795 control steps.
- The reported number is the **peak size of the largest connected, hands-free, table-supported stack rooted in the goal zone**. A pickup, two separate boxes in the zone, or a farther target index can all leave it unchanged. The task has a separate held-success condition. The old report also counted 152 of 373 course episodes that reached at least two boxes and ended below their peak. Protecting a partly built stack matters because the next box needs a support.

The training track supplied noisy teacher trajectories, world-model pretraining, policy training, and predictor finetuning on sampled policy chunks. The policy, world model, goal file, and evaluation loop are fixed in this track. An agent may use the supplied RGB, its own issued actions, and episode-local memory. The seed initializes random sampling. All box identities, visibility flags, and phase estimates below come from images. Only the evaluator supplies true placements for scoring. [Notebook] · [Training pipeline] · [Policy code] · [World-model code] · [Tower score] · [Episode loop]

### Two inexpensive checks before a long screen

**Check the visual signal.** The front and side cameras are fixed to the simulated world. The hand can occlude the object in either image. I split the 110 frames of `docs/planning_demo.gif` into the two live views and resized each to 128 × 128. A simple hue mask found a green component of at least 25 pixels in **110/110 frames in each view**. It found exactly two blue components of at least seven pixels in **107/110 front frames and 109/110 side frames**. The first frame contains two blue boxes. This establishes visibility in one compressed clip. It does not measure identity accuracy, contact, or performance on validation scenes. [Demo clip] · [Camera definitions]

To reproduce the check, use the upper half of the GIF, bilinear resize, and eight-connected components. I used HSV hue in degrees, saturation above 0.43 and value above 0.23. Green was `136 < H < 173` and blue was `192 < H < 228`. For a planner, calibrate color from `init_frame`, keep a separate track for each instance, and carry an “unknown” flag through occlusion. Use the last clear base location when the green zone is covered. Review crops from several development scenes before using a track to select actions.

**Check what the model's published accuracy measures.** The final predictor selected the closest **encoded endpoint** for 76.0% of all 6,208 released validation states and 77.6% of an earlier 1,152-state subset. Both targets are simulator-rendered frames encoded with the same model. These numbers say how often it can reproduce a latent-distance ranking. They leave open whether that ranking selects a box placement or a useful first action. [Training pipeline] · [Finetuning metric]

### Shared implementation details

`u` below is a candidate converted to raw actions by `clip(blocks * a_std + a_mean, -1, 1)`, then reshaped to `(N,16,5,25)`. With edited actions, convert the issued block back to model units using `(u.reshape(125) − a_mean) / a_std` before appending it to `self.executed`. Preserve the alignment between each observation and the block that followed it. Initialize added memory and random generators in `reset()`.

For the visual experiments, track the pale hand, each colored box, and the goal zone in each view. Associate duplicate colors by position and motion across consecutive frames. Infer a possible grasp when a visible off-zone box moves with the hand. Infer a possible release when the hand departs and the box stays on the support. Require agreement across two visible views for the first trial. If the views disagree or a component splits, use the control action and record the reason. These flags are estimates.

Several tests project a proposed wrist motion into the cameras. Within the current episode, compare the sum of **issued raw wrist increments** in each recent block with the next observed hand displacement. With at least six visible, varied motions, use a regularized least-squares matrix `J` from three wrist commands to four image coordinates `(front_x, front_y, side_x, side_y)`. Reject an estimate that fails to predict the latest held-back visible displacement or lacks motion in the required direction. Apply it over one block, near the observed workspace. Actual joint targets can saturate, and fingers can move the hand centroid. This estimate uses the two existing images and needs no camera poses or extra checkpoint. [Visual servo control][P1]

## Experiments 49–53 · First comparisons

### 49. Compare the first block under several possible futures

The winner of an eight-second rollout can owe its score to a finger motion seven seconds from now. At each replan, keep four distinct first blocks from the ordinary candidates. Include the baseline winner and choose three others with different first-block wrist or finger motion. Predict one block for each. From each predicted latent, draw **three new 15-block continuations** with the frozen policy. Use the same three noise draws for every first block. Roll forward with the proper frame and action history. Give the first block the **mean** of its three terminal costs. Execute the winning original first block. Maximum modeled depth stays at 16. Old ideas 22/41 branch after a full chunk. [PEGASUS][P2] · [Diffusion Forcing][P3] · [Commit While Futures Agree][P16]

**Implementation.** Reuse `pred[i,0]` from the ordinary rollouts. Call the policy with `(pred[i,0] − z_mean) / z_std`. Draw a noise panel of shape `(3,15,125)` once. Reuse that panel for each of the four contexts, either with an explicit-noise sampler or by restoring a private generator's state before each group. One ordinary batch of 12 independent draws would give each prefix different noise. Start each continuation rollout at the history ending in `pred[i,0]`. Include that prefix's action among the context actions. With three tails, the added batch predicts `4 × 3 × 15 = 180` latent transitions per replan. The original call predicts `16 × 16 = 256`. Those counts exclude policy sampling time.

**First test.** Compare one continuation and three on the same four first blocks. Record how often the original first block loses and how often the ranking repeats with a fresh noise panel. Track `P(≥2 | ≥1)`, peak boxes and median replan time. The extra policy batch is the main cost. If rank order changes on every independent panel, the estimator needs more evidence before live use.

### 50. Center the carried box over its actual support in both views

Trigger when a visible box moves with the hand toward an existing stack. For each camera, measure the horizontal offset from the **bottom-center of the carried box** to the **top-center of its support**. Divide by the apparent support width. Among the four lowest-cost policy chunks, project the first block's wrist motion using `J`. Select the block that minimizes the larger **absolute** predicted offset across the two views. Use the normal terminal cost for ties. Issue the selected sample intact, including its finger commands. Small offsets in one view can hide a large error in the other. [Visual servo control][P1] · [ReKep][P4] · [Stable placement planning][P5]

**First test.** Enable the rule only for a second or later placement. Compare it with the same shortlist ranked by ordinary cost. Measure offset at the next real frame, the fraction of apparent releases that add a scoring box, and peak boxes. If the real offset moves the wrong way, fix the projection before testing a larger weight. The rule adds image processing and no model rollouts.

### 51. Change the visual target when the hand has already picked another valid box

Find the moving reference box from changes between adjacent goal images. Compare it with the live box that rose with the hand. When they differ, form one alternate **goal image** by exchanging those two instances' hues in both goal views. Preserve brightness, shading, position, and the original hand pixels. Encode the alternate goal once. Use the same alternate target for every candidate during that carry. Resume scoring the original target when the box is released. Start with clearly separated, similar-size boxes of different colors. Ambiguous duplicate-color matches use the control target. This changes the scored destination during the conflict. Old idea 46 used colors to check stage progress. [Entity abstraction][P6] · [MimicGen][P7]

**First test.** Inspect `100038` for the trigger and the end of the carry. Compare the alternate target with a sham that reapplies each original color. A second control lets the policy act alone as soon as the same RGB conflict is detected. That comparison asks whether the new goal adds anything beyond getting out of the policy's way. Report all scenes and hand-checked conflicts. If the mismatch is mainly spatial layout or box size, recoloring may have little effect. Added cost is one cached goal encoding per new assignment.

### 52. Test the action representation at the clipping boundary

Cell 7 predicts from a sampled, standardized chunk. It clips the raw commands for execution. The on-policy collector also executed clipped commands, but stored the **original sampled chunk** as the predictor input. The predictor could have learned this mapping. Measure the fraction and magnitude of clipped wrist and finger entries. On identical candidates, compare terminal predictions from the original input and from the clipped raw chunk mapped back to standardized units. Change only the representation used for scoring. Issue the same bounded actions for a given candidate. [Notebook] · [On-policy collector] · [Finetuning metric]

**First test.** On released validation shards or paired trace frames, record changed winners and true endpoint ranking regret where true future latents exist. Compare saturated and unsaturated candidates separately. Advance to a live screen if clipping occurs often enough and the second representation improves that check. A single representation costs the same number of rollouts in deployment. The two-way diagnostic costs an extra rollout batch.

### 53. Remember the approach that just produced an empty grasp

When the hand closes, rises, and leaves a visible box on the table, record that instance, its image position, and the direction of the last approach. On the next attempt at the same box, compare the ordinary top four candidates. Prefer a different approach direction if one is available at similar terminal cost. Use four direction bins relative to the box in the two views. Expire the entry after a successful lift, a clear change in box position, or eight calls. Use only this episode's evidence. A timeout cannot tell which failed motion to avoid. [Robustness via Retrying][P8]

**First test.** Compare a remembered failed direction with a randomly assigned direction on the same trigger. Measure repeat approaches, second-attempt lifts, and time from first to second placement. If nearly all candidates share one direction, the trace identifies a proposal problem. This rule adds no model calls.

## Experiments 54–59 · What the world model rewards

### 54. Correct a persistent prediction error using the last issued block

Save the one-block prediction for the action actually sent. At the next frame, compute the signed error `e = z_observed − z_predicted`. Keep an exponential average for the current grasp or carry, starting with equal weight on the new and previous estimate. Add half this average to each candidate's **first** predicted latent and propagate the corrected latent through the remaining rollout. Cap the correction norm at half the median recent observed one-block motion. Clear it on a large direction change or a new phase. Old idea 12 used the size of an error to trigger recovery. Here its direction changes the forecast. [Offset-free MPC][P9]

**First test.** On real traces, ask whether the previous error predicts the direction of the next error on held-back calls. Compare signed correction with zero and sign-shuffled corrections. Run closed loop only where the signed version improves forecasts. It needs the same model depth and a short history.

### 55. Ask what action the frozen policy would take at the target

Two states can have similar image latents with different next actions, such as reaching for a box and withdrawing after release. At reset, query the policy at each real subgoal using two fixed noise draws and `generate=1`. Save each first block. For the four best predicted endpoints, repeat those two queries with exactly the same noise. Compare the resulting net wrist motion and finger increments with those at the goal. Use this difference to break terminal-cost ties. Standardize wrist and finger groups separately and give each group equal weight. This uses the policy as a state comparison without rolling past block 16. [Policy similarity embeddings][P10]

**First test.** Check whether this score separates pairs of real reference images with similar hand poses but different next actions. Compare with using independent noise at each state. Then measure whether it helps the third-box rate. Cost is eight one-token policy samples per selected replan, batched. A predicted endpoint can be outside the policy's training distribution.

### 56. Reduce the influence of hand-only motion in the terminal distance

At reset, find pairs of real subgoals whose visible boxes occupy the same places but whose hand positions differ. Form their latent difference vectors. If at least three reliable pairs exist, extract the first one or two orthonormal directions `U` by singular value decomposition. During a carry, score a terminal error `d` with `||d||² − 0.5 ||Uᵀd||²`. Keep the ordinary tracker. This reduces the weight of a measured hand-motion direction in this scene. The earlier whitening test scaled all coordinates by marginal variance. [First idea file] · [Entity abstraction][P6]

**First test.** Compare with equally many random orthonormal directions and with raw distance. On hand-checked frames, ask whether a correct box layout with a different hand pose now ranks above an incorrect layout. Skip the adjustment if selection of pairs fails. Almost all added work occurs at reset.

### 57. Apply a demonstrated motion to the point where this attempt actually started

On entry to goal `m`, store the real latent `z_entry`. For this stage, score endpoints against `g[m] + 0.5 × (z_entry − g[m−1])`. At `m=0`, use the initial-frame latent for `g[m−1]`. Cap the added offset at one quarter of the norm of the reference transition. Clear it when `m` changes. This asks for the local change seen in the demonstration despite a small persistent difference in the existing tower. It uses one terminal target and avoids the time-aligned cost from old idea 5. The use of a difference in latent space is a hypothesis about this encoder. [MimicGen][P7]

**First test.** Compare the anchored goal with the original endpoint on scenes where a box was placed slightly off the reference center. Log endpoint changes, placement times, and real tower growth. A badly timed entry captures the wrong hand pose, so start with stages that have an unambiguous visible release. No additional model call is needed.

### 58. Ask whether the fingers contributed to a predicted pickup

For the top four candidates at a likely grasp, make a counterfactual copy. Keep the wrist motion and set the finger channels to **raw zero increments** for the first two blocks. Map the copy back to model units. At block four, compare the original and modified predictions against the next real goal that visibly shows the box lifted. Skip the test when the coarse reference sequence omits this phase. Among the original candidates, prefer one whose finger action improves that distance. Execute its original first block. The counterfactual serves only as a model query and can be outside the predictor's training distribution. [Generative predictive control][P11]

**First test.** Compare full and finger-held predictions with a wrist-held counterfactual on the same candidates. Check whether the finger-effect score distinguishes visible lifts from empty-hand motions on development traces. If moving the hand dominates every comparison, drop this score. It costs four additional short rollouts per trigger.

### 59. Reject rankings that depend on a one-level change in camera brightness

Keep the original 16 policy samples. For the four lowest-cost chunks, encode a second copy of the real frame history and goal with a small consistent exposure change, such as `+1` on the uint8 scale in one camera. Re-evaluate the same actions. For each view condition, subtract the best candidate cost in that condition and divide by the cost spread plus a small positive floor. Pick the candidate with the smallest **worst** relative cost across the two conditions. Keep both encoders in evaluation mode. The earlier dropout trial sampled network masks. Here the changed quantity is an observed input with a known perturbation. [Run notes]

**First test.** Measure rank reversals on identical frames and action batches. Use a repeat of the original encoding as the sham. Run live only if the sensitivity predicts poor choices and the robust criterion improves them in replay. Cost is one extra four-candidate rollout plus the re-encodes. Excessive invariance could wash out a small box.

## Experiments 60–66 · Pick, carry, release, and recover

### 60. Make a small lift to verify that the box came with the hand

Trigger after an apparent close over an off-zone box when the grasp is uncertain. For one block, hold finger targets and command a small upward wrist motion with zero sideways travel. A raw `+0.10` on wrist Z for five steps requests about 7.5 mm of target movement before actuator limits. Compare the next image with the previous one. A visible box that rises with the wrist supports continuing the carry. A box left behind triggers a fresh approach to that instance. Use at most one probe per approach. [Camera and action definitions][Camera definitions] · [Interactive perception][P12]

**First test.** Use a same-duration hold as the control. Count empty carries avoided, real lifts lost to the probe, and `P(≥2 | ≥1)`. Start from the natural upward candidates if they produce enough displacement to see. If a 128-pixel image cannot resolve the motion, mark the outcome unknown. No extra model call is required.

### 61. Brake sideways motion when a carried box starts to slide

During a confirmed carry, track box displacement relative to the hand. If the box drifts downward relative to the hand across two calls in both views, prefer a top-four candidate whose first block has little lateral wrist motion. Maintain the current finger targets for one block if the selected motion would open the hand. Resume normal selection at the next frame. The event should occur before the box separates fully. It can occur while the target index still advances. [Interactive perception][P12] · [Vision and touch regrasp study][P13]

**First test.** Compare the relative-motion trigger with applying the same brake at matched random carry calls. Inspect a sample of alerts for hand occlusion and camera perspective. Track recovered carries, dropped boxes, and added time. An RGB slip cue is indirect. This trial uses no force input and adds no world-model rollout.

### 62. Permit release only when the box appears seated

Near a placement, look for a small lower-face to upper-face gap and horizontal overlap in both views. Estimate the apparent seated gap from boxes already stacked in clear goal images. Normalize by the local box height. For one block at most per release, suppress opening if the support gap is still large or the box is still moving sideways. Preserve the candidate's wrist command. Infer opening from a previously observed hand-widening command pattern. Until that pattern and the reference gap are available, use the control. The signs of the 22 joint channels can differ. [ReKep][P4] · [Stable placement planning][P5]

**First test.** Compare the image trigger with a fixed one-block opening delay on the same candidate release events. Count early drops, boxes that the hand fails to release, and releases that add a box. This changes finger timing only. Experiment 50 changes wrist selection. It adds no model calls.

### 63. Keep lateral transport clear of the existing tower

When a box is carried sideways, form a protected region around the existing tower in each view. Expand its outline by half the carried box width. Use `J` to project the swept path of the carried box over the five commands in the first block. Filter a top-four candidate if it crosses the tower before the box bottom clears the visible top. Allow a centered descent once lateral motion is small. If all four are filtered, issue the ordinary winner. [ReKep][P4] · [Stable placement planning][P5]

**First test.** Log tower motion immediately after the loaded hand passes it. Compare with a region of the same size shifted away from the tower as an offline sham. Measure whether fewer two-box towers lose a box before the next placement. A wide exclusion region can prevent the hand from reaching the target. This rule uses only short image projections.

### 64. Choose the departure direction after a release

When the box stays still as the fingers open, save its top outline. From the policy shortlist, prefer a first block that moves the hand up and away from the new top before it sweeps toward the next pickup. Project the hand mask over the next five commands. Use the minimum projected gap from the top after separation as the tie-breaker. Apply this for at most two calls after each of the first three apparent placements. This addresses an empty hand striking the tower on departure. Old idea 48 dealt with finishing at the final goal. [Robustness via Retrying][P8] · [Stable placement planning][P5]

**First test.** Compare with the same upward preference applied one block later. Track displacements immediately after release and the rate of reaching the next box. If a tall tower blocks the next approach, measure the time cost. This idea can help peak score only if it preserves a support needed later. It adds no model rollout.

### 65. Move an empty hand out of the way to see the next box

Trigger when a previously visible loose box disappears under the projected hand in both views and the hand was empty. Shift the observed hand mask by each shortlisted first-block displacement. Choose the motion that uncovers the largest fraction of the box's last visible footprint. Require the movement to stay clear of the tower. Make one such move, take a real observation, then replan. A fixed camera can acquire new evidence because the hand itself causes the occlusion. [Interactive perception][P12]

**First test.** Compare one uncovering move with one stationary block. Count reacquired tracks at the next frame and the time to the next pickup. Disable the rule when the box might already be in the hand. A higher visibility rate alone does not justify the time cost. No new model calls are needed.

### 66. Reuse finger closure from a successful grasp in this episode

After an image-confirmed lift, save the one or two issued finger blocks just before the lift and the box's apparent width near the hand. On a later attempt at a similar-size box, form two candidates that pair the cached finger blocks with fresh policy wrist motion. Keep the remaining actions from the fresh samples. Match closure onset to an observed approach from a similar direction. Re-score the modified sequences. Replace two of the 16 proposals to hold candidate count fixed. The memory expires at episode end. [MimicGen][P7] · [Vision and touch regrasp study][P13]

**First test.** Compare real successful closure with a duration-matched finger sequence from another visible phase. Measure later-box lifts and placements separately. If the cache fires too rarely, report coverage. Similar apparent width can hide a different height or friction. Log that distinction in the video review. This test uses at most the baseline rollout count.

## Experiments 67–72 · Alternative plans and action timing

### 67. Let the next real frame choose among compatible earlier plans

Save up to four previous candidates with first blocks close to the issued block in **raw command space**, together with their predicted next and terminal latents. On the next call, compare their one-block predictions with the observed latent. Take the remaining 15 blocks from the closest compatible candidate. Append one policy block conditioned on its standardized predicted terminal latent. Re-score all 16 blocks from the current real context. Replace one fresh proposal. Reject a saved candidate if its first block differed substantially from the one actually issued. Old idea 7 kept the previous winner by default. This experiment can recover an unselected sibling whose prediction matches reality. [DREAM-Chunk][P14]

**First test.** Compare observation-based choice with picking a random compatible sibling. Record usable siblings per call, their forecast errors, and peak boxes. If the compatible set is usually empty, stop early. Preserve one-block execution. The extra cost is a short tail sample and memory for four plans.

### 68. Hold proposal noise constant while the observed state changes

Draw 16 random initial noise tensors from the episode generator. For two successive replans, decode those same tensors from the **newly observed policy latent** each time. Refresh half the tensors on the second call and the remainder on the third. All actions are generated afresh and any candidate can win. A grasp approach may then evolve smoothly as the box moves, without requiring the old action sequence to remain correct. Old ideas 7, 43, and 47 reused an action plan, an offline-selected noise, or a partly denoised trajectory. [Diffusion Forcing][P3] · [PEGASUS][P2]

**First test.** Reproduce the policy's ten Euler updates inside a class helper with explicit initial noise. Check it against `policy.sample` with the same tensor and precision. Then compare persistent and wholly fresh noise panels, with equal policy and world-model calls. Record unproductive changes in approach direction and successful second pickups. Persistent bad samples can reduce exploration.

### 69. Penalize wrist reversals inside the block the simulator actually receives

For each top-four candidate, let `v[t]` be its five raw wrist increments. Calculate `R = sum_t ||v[t]||₁ − ||sum_t v[t]||₁`. A large value means the wrist moves one way and back in the same half-second. In free transport, use the smaller `R` to break terminal-cost ties. Leave contact phases alone in the first trial. Old idea 31 compared consecutive plans, and idea 32 averaged actions across samples. This score examines one unmodified block. [Phase-aware chunk execution][P15]

**First test.** Compare reversal cost with a simple total-motion penalty. Match or report net wrist travel so a stationary hand cannot appear to improve the result. Check the next observed box motion, carry duration, and peak boxes. A useful correction can reverse direction, especially near contact. This computation is small and uses no new model calls.

### 70. Choose a base that leaves enough support for a later box

At the start of a tower, identify which loose box each plausible reach approaches. Project the first two wrist blocks while the hand is empty. Estimate candidate boxes' relative top-face widths in both views, using the initial image and local table scale from visible scene features. When two boxes are reachable, prefer a wider base if both views agree on their ordering. At later picks, prefer a box whose apparent footprint fits the exposed support. Use the original terminal cost within the selected group. The four objects have square footprints of 50, 55, or 60 mm, so the visual differences are small. [Box geometry] · [Stable placement planning][P5]

**First test.** Compare the width preference with a nearest-reachable-box rule on the same candidate groups. Manually check width order on development images and record how often two groups exist. Report ≥3 and 4 alongside time to box one. If perspective or merged outlines dominate the width estimate, retire this idea. It adds geometry checks and no policy rollouts.

### 71. Choose a reachable next pickup when the episode deadline approaches

Use only elapsed calls and visually observed pickup and placement events. After one completed cycle, estimate the time each remaining pick would require. Use empty-hand travel to the box, loaded travel to the tower, observed image motion per block, and the measured close-and-release overhead. When the time left to step 1,795 is shorter than a typical full cycle, prefer a plausible policy reach to the box with the shortest predicted completion time. Use the ordinary winner if the tracks or timings are unreliable. This changes which box to pick. Old idea 30 moved the goal index with the reference clock.

**First test.** Record predicted and actual completion times before using the estimate for control. Compare deadline-only selection with shortest-trip selection throughout the episode. Inspect late transitions from two to three and three to four, and report how many episodes actually trigger. A shorter trip can still involve a more difficult grasp. No extra model calls are needed.

### 72. Test the timing of finger closure against the first lift

At the start of a lift, make two variants of a top policy block. Move the finger-closing sequence one of the five control steps earlier or later relative to the wrist. Keep each finger's total signed increment and wrist sequence unchanged. For example, if the first finger row is near zero, move rows 2–5 to slots 1–4 and move the near-zero row to slot 5. Use the analogous shift only when the opposite boundary is quiet. Keep a raw, unmodified candidate in the batch. Re-score from the same observed state. This changes contact timing by 0.1 s inside a block while retaining the policy's grasp pattern. [Phase-aware chunk execution][P15]

**First test.** Compare early, original, and late closure with equal candidate count. Log whether the world model distinguishes them and whether the box rises with the hand on the next frames. The model embeds all 125 values in one token, so it may miss this timing difference. If offline predictions tie, compare the two bounded timing variants directly on paired development episodes. Use the true issued block for the next model context.

## How to run the next round

**Reproduce the control.** The local process used `nbconvert` from the repository root. It ran the policy-only and planning episodes on `100038` and saved the executed notebook and both videos. Use a new output directory for each variant. The notes found occasional trajectory differences when another job shared the GPU. Record the device, precision, sampler seed, and whether the GPU was shared. The notebook's printed planner label is hard-coded in cell 8, so name the output directory after the actual class settings. [Notebook] · [Run notes]

```bash
mkdir -p runs/r2-49-first-block
MUJOCO_GL=egl .venv/bin/jupyter nbconvert --to notebook --execute \
  assignment/planning/colab.ipynb \
  --output-dir "$PWD/runs/r2-49-first-block" --output colab.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

**Use the local cell-7 harness for paired runs.** On the workstation described in the notes, the first command below runs the saved control. The second runs the edited notebook cell on the same 32 scenes. Both use the documented default sampler seed 0. Keep a unique result name for each config because the harness resumes existing JSONL files. In a clean checkout these harness files are absent. The stock `assignment.training.planning.evaluate` imports `GoalPlanningAgent` from `agent.py`, so invoking it directly benchmarks the separate Python class. [Run notes] · [Evaluator import]

```bash
cd runs/dev
../../.venv/bin/python harness/nbeval.py --source cells/final_A.py \
  --first 0 --scenes 32 --gpus 0,1 --workers-per-gpu 1 --out results/r2_control.jsonl
../../.venv/bin/python harness/nbeval.py --source notebook \
  --first 0 --scenes 32 --gpus 0,1 --workers-per-gpu 1 --out results/r2_49.jsonl
../../.venv/bin/python harness/compare.py results/r2_control.jsonl results/r2_49.jsonl
```

| Round | What to run | Decision |
|---|---|---|
| **1 · Diagnose** | Use existing traces and videos from the 32 development scenes. Review attempts around the first placement and the next pickup. Mark correct pickup, alternate pickup, empty lift, off-center approach, early release, tower strike, and unknown. Allow several labels per attempt. | If failed grasps dominate, start 53 or 60. If loaded hands reach the tower but placement fails, start 50 or 62. If reference conflict is frequent, prioritize 51. Run the clipping audit from 52 here. |
| **2 · Screen** | Run 49, 50, and the most frequent failure-specific intervention individually on all 32 paired scenes. Log trigger counts and changed actions. For a geometry rule, score the same candidate shortlist in the control. For 49, compare elapsed time with an ordinary planner given the same compute budget. | Promote a candidate when the physical intermediate outcome and peak boxes move in the expected direction. At this sample size, treat a small mean change as provisional. An agent that never triggers needs a different test. |
| **3 · Test one combination** | If 49 and a visual rule each show a gain, run control, 49 alone, the visual rule alone, and both on the same scenes and seeds. For pickup and release rules, check whether one prevents the other from firing. | Keep the combination only if it helps beyond the better single change. Compare median and 90th-percentile `step()` time with the 106 ms local control. |
| **4 · Confirm once** | Fix the intervention, weights, and endpoints. The original 68-scene set has already informed this round. For a fresh-scene test, use a previously untouched portion of the course's 300-scene validation set through a cell-7-capable evaluation harness if available. With the supplied 100-scene file alone, use new sampler seeds such as 2 and 3 and report this as a new-seed test on known scenes. | Compare with the local final and retain bf16-only and the original planner as references. Report the paired difference and 95% interval. Keep the submission notebook's models, goals, and loop fixed. |

**Score each experiment on the task.** The primary result is mean peak boxes. Also report ≥1/≥2/≥3/4, `P(≥2 | ≥1)`, `P(≥3 | ≥2)`, and time to the kth placement. Report how many episodes reach each stage when summarizing placement times. For a conditional rule, give its trigger count, number of affected episodes, false triggers from video review, and the all-episode result. `self.trace` can hold the estimated phase, chosen candidate, reason for an override, and forecast before each action. Review evaluator labels only after the episode. One row of the notebook alone gives no population estimate.

**Budget for small effects.** The reported 68-scene comparison with bf16 only had a paired SE of 0.13. That implies a paired standard deviation near `0.13 × sqrt(68) ≈ 1.07` boxes. Under similar noise, a two-sided 5% test has roughly 80% power for a true +0.20-box effect with about **225 independent scene pairs**. At +0.10 the estimate is about 900. These are planning estimates. Multiple sampler seeds from one scene share its difficulty. Average paired differences within each scene and bootstrap scenes for the interval. The course's 300 scenes × seeds 0/2/3 provide a useful reporting scale. [Run notes] · [Course benchmark]

## Literature and the specific ideas it supports

The papers below establish individual methods in other settings. The proposed thresholds, image rules, and transfers to this frozen checkpoint need the paired tests above. The geometry experiments use only the supplied images. The original idea file flagged handcrafted color cues as a possible competition-rule question. The published planning instructions authorize RGB inputs and leave detailed competition rules open. The first-block test, clipping audit, and policy-comparison test also work with the frozen encoder alone. [Course benchmark] · [First idea file]

| Paper | Transfer to this planner |
|---|---|
| P1 · Chaumette and Hutchinson, *Visual Servo Control, Part I* (2006) | Image errors can drive a feedback controller through a local image-motion map. Experiment 50 uses this to choose among policy wrist moves. [Paper][P1] |
| P2 · Ng and Jordan, *PEGASUS* (2000) | Evaluating choices with shared random scenarios reduces one source of comparison noise. Experiment 49 shares future policy noise across competing first blocks. [Paper][P2] |
| P3 · Chen et al., *Diffusion Forcing* (NeurIPS 2024) | Causal sequence generation supports shorter action prefixes. The repository sampler exposes both a generation horizon and a noise generator. These make 49 and 68 implementable. [Paper][P3] |
| P4 · Huang et al., *ReKep* (CoRL 2024) | Relational keypoints give costs for different manipulation phases. Experiments 50, 62, and 63 use visible box and support features in two images. ReKep itself reasons over 3D keypoints. [Paper][P4] |
| P5 · Nadeau and Kelly, *Stable Object Placement Planning from Contact Point Robustness* (IEEE T-RO 2025) | Stable placement depends on the available support contacts. The proposed image-overlap and width rules approximate only part of that geometry. [Paper][P5] |
| P6 · Veerapaneni et al., *Entity Abstraction in Visual Model-Based Reinforcement Learning* (CoRL 2019) | OP3 uses object-level information and observed motion for visual planning. Experiments 51 and 56 recover limited instance information from images without training OP3. [Paper][P6] |
| P7 · Mandlekar et al., *MimicGen* (CoRL 2023) | Demonstration segments can be adapted to other object configurations. Experiments 51, 57, and 66 test goal or action transfer at inference time. [Paper][P7] |
| P8 · Ebert et al., *Robustness via Retrying* (CoRL 2018) | Maintaining visual correspondence lets a controller retry an object after an error. Experiment 53 adds an episode-local record of the failed approach. [Paper][P8] |
| P9 · Pannocchia and Rawlings, *Disturbance Models for Offset-Free Model-Predictive Control* (2003) | An estimated disturbance can compensate for persistent model error. Experiment 54 tests whether such an error persists in the learned latent over a grasp or carry. [Paper][P9] |
| P10 · Agarwal et al., *Contrastive Behavioural Similarity Embeddings* (ICLR 2021) | Actions available in two states provide information about their similarity. Experiment 55 tests a sample-based proxy using the existing policy. [Paper][P10] |
| P11 · Qi et al., *Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling* (IEEE RA-L) | GPC uses predicted consequences to rank candidate actions from a frozen policy. Experiment 58 adds a paired intervention to ask what effect the fingers had. [Paper][P11] |
| P12 · Bohg et al., *Interactive Perception* (IEEE T-RO 2017) | An action can generate a missing observation. Experiments 60 and 65 spend one block to reveal attachment or position. [Paper][P12] |
| P13 · Calandra et al., *More Than a Feeling* (IEEE RA-L 2018) | The paper uses vision and touch to adjust a grasp. The proposed RGB-only correction has less evidence and needs a measured false-trigger rate. [Paper][P13] |
| P14 · Chen et al., *DREAM-Chunk* (2026 preprint) | The method matches observed outcomes with predictions from several candidate chunks. Experiment 67 adds a first-action compatibility check before reusing a candidate. [Paper][P14] |
| P15 · Nie et al., *PACE* (2026 preprint) | Changes in predicted movement speed identify transitions between manipulation phases. Experiments 69 and 72 examine the timing of individual commands within a five-step block. [Paper][P15] |
| P16 · Li et al., *Commit While Futures Agree* (2026 preprint) | The authors compare the consequences of actions under shared noise to guide commitment. Experiment 49 keeps the execution length fixed and compares several continuations of each first move. [Paper][P16] |

[Commit comparison]: https://github.com/Paskul/pascalteam-cis6280-final-project/compare/9540172640099ac764e3eeca52d09265dbb018fb...79d1893b9ceb787e3739b45aa99c4d5e0f4bf834
[Notebook]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/planning/colab.ipynb
[Run notes]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/notesForRunning.md
[First idea file]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/ideasToTest.md
[Course benchmark]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/planning/README.md
[Training pipeline]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/README.md
[Policy code]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/training/policy.py
[World-model code]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/training/jepa.py
[Tower score]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/data/signals.py#L137-L172
[Episode loop]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/data/task.py#L183-L210
[Demo clip]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/docs/planning_demo.gif
[Camera definitions]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/cube_pick/sharpa.py#L40-L83
[On-policy collector]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/data/collect_onpolicy.py#L118-L147
[Finetuning metric]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/training/finetune_onpolicy.py#L85-L114
[Evaluator import]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/assignment/training/planning/evaluate.py#L62-L75
[Box geometry]: https://github.com/Paskul/pascalteam-cis6280-final-project/blob/79d1893b9ceb787e3739b45aa99c4d5e0f4bf834/docs/SHARPA_VARIED_BOXES.md
[P1]: https://doi.org/10.1109/MRA.2006.250573
[P2]: https://aima.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Ng%2BJordan%3A2000.pdf
[P3]: https://arxiv.org/abs/2407.01392
[P4]: https://proceedings.mlr.press/v270/huang25g.html
[P5]: https://philna.de/projects/stable-placement-planning/
[P6]: https://proceedings.mlr.press/v100/veerapaneni20a.html
[P7]: https://proceedings.mlr.press/v229/mandlekar23a.html
[P8]: https://proceedings.mlr.press/v87/ebert18a.html
[P9]: https://doi.org/10.1002/aic.690490213
[P10]: https://research.google/pubs/contrastive-behavioural-similarity-embeddings-for-generalization-in-reinforcement-learning/
[P11]: https://computationalrobotics.seas.harvard.edu/GPC/
[P12]: https://arxiv.org/abs/1604.03670
[P13]: https://publications.ri.cmu.edu/more-than-a-feeling-learning-to-grasp-and-regrasp-using-vision-and-touch
[P14]: https://arxiv.org/abs/2606.18589
[P15]: https://arxiv.org/abs/2606.00537
[P16]: https://arxiv.org/abs/2610.07949
