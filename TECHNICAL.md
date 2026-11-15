# Ball possession from tracking: evidence + switch-cost segmentation

Technical document for More than a Hack 2027, Phase 1. Team AID.

## 1. Approach

The ranking metric, `spellF1`, scores whole possessions matched by temporal IoU. One flickering frame
splits a possession in two, and a short false `D` does the same. So the method has two stages:

1. **Per-frame evidence.** For each frame, a score for each state `H`, `A` and `D`. Each score is
   built from physical facts in the tracking plus weak priors from Smart Tagging (`solution/evidence.py`).
2. **Segmentation.** A 3-state Viterbi path maximises the total evidence minus a **switch cost** for
   every change of state (`solution/segment.py`). A change needs sustained evidence, so one noisy
   frame can't split a possession. A missing ball contributes no H/A evidence, so the current
   possession is carried through occlusions. That matches what the scorer does with `X`.

A light clean-up follows:
- `D` gaps shorter than 1 s inside one team's spell are closed.
- `D` gaps shorter than 1 s between two different teams are split at the midpoint.
- H/A spells shorter than 0.5 s are absorbed by the state before them.

We chose this over a learned model because no labels from this game may be used. We had no
labelled data from other games in the same format. A transparent model with physical parameters
can't overfit the 17 public possessions.

## 2. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD (players + ball) | yes | main signal: distances, ball speed, ball position vs lines, player speeds |
| ATD metadata XML | yes | which track slot belongs to which team (first team = home) |
| Smart Tagging | yes, weakly | team-tagged phases, set pieces, kick-offs, goals (see 3.3) |
| homography | in the video experiment only | maps video detections to pitch coordinates and places the crops (section 5b) |
| tactical video | tried, dropped from the submission | ball detector to fill ATD gaps (section 5b) |
| broadcast video | no | not synchronised to tactical |
| broadcast ATD | no | not synchronised to tactical; mapping it back is a project on its own |

## 3. Evidence and possession rules

All distances are in metres (normalised coordinates × 105.3 × 68.0). All parameters are in
`solution/params.py`.

### 3.1 Control (H vs A)
- Take the nearest home and the nearest away player to the visible ball.
- Weight: `exp(-max(d_nearest - 1 m, 0) / 1.5 m)`. A ball within a stride counts fully. A pass in flight
  far from everyone counts for almost nothing.
- Sign and strength: `tanh((d_away - d_home) / 1.5 m)`. A clear 1.5 m advantage counts strongly. A
  shoulder-to-shoulder duel counts for almost nothing.
- A ball faster than 12 m/s is struck or in flight, so its weight drops to 25%. A ball out of bounds gets
  zero weight, because a thrower standing out of play does not control a live ball.

### 3.2 Dead ball (D)
- Ball visible beyond a line by more than 0.5 m: strong `D`.
- Ball lost by the tracker within 1.5 m of a line, or reappearing within 1.5 m of a line: mild `D` over the
  whole missing run (it probably left the pitch).
- Visible, in bounds and moving faster than 1 m/s: against `D`.
- **Tempo** is the mean player speed over a ±1 s window. It is calibrated per half on that half's own
  20th and 60th percentiles, which needs no labels. Slow play pushes towards `D`, fast play away from it.
- Ball speed is the net displacement of a smoothed ball position over ~0.5 s, so tracking jitter
  does not read as movement.
- Ball standing still for 2 s or more while the game is slow: `D` (set-piece setup).
- **Kick-off formation**: ball within 2 m of the centre spot, with each team entirely in its own half:
  strong `D`.

### 3.3 Smart Tagging (weak prior)
- **Team-tagged phases** add a small bonus (0.2 per frame) to one side. Overlapping phases are
  combined into one net vote per frame, so they cannot stack into a strong signal. For each code, the data
  decides whether `Team` names the side in possession or the defending side. The rule is agreement with
  the proximity evidence inside that code's intervals, which needs no labels. The run prints the
  decision for each code.
- **Set pieces**: the first half of the tagged window leans `D`.
- **Kick Off (Start)**: graded frames of a half before its tagged kick-off are `D`. This covers the
  second half, which is graded from 110,808 but kicks off later. The kick-off frame is read from the
  tag, not hard-coded.
- **Goal → Kick Off (After Goal)**: the gap between the two tags is `D`, if they are under 3 min apart.
- `N/A` team tags (all throw-ins) give no side.

### 3.4 Where the lines are drawn
- **Loose ball / pass in flight**: possession stays with the last team that controlled it until the
  other team controls it for long enough to pay the switch cost (~0.4 s of clear control).
- **Challenges**: equal distances give almost no evidence, so the current possession continues until one
  side clearly wins the ball.
- **Restarts**: dead from the moment the ball goes out (or the set-piece setup is detected) until the
  ball moves again. The restarting team then takes possession from the first control evidence.
- **Goalkeeper holding the ball**: possession, not `D`, unless the game around it is slow and the ball
  is stationary for more than 2 s.

## 4. Training data

**No labels from the provided game, and no learned weights.**

For calibration only, we used **Metrica Sports' public sample data**
(github.com/metrica-sports/sample-data, Sample_Game_1 and Sample_Game_2). These are two other full
matches with tracking and human-coded event data (Metrica's annotators): 1,745 and 1,935 events.
Section 5c describes how we used them. That check did not change any parameter.

## 5. Validation without labels

- `tests/make_synthetic.py` writes a synthetic bundle in the exact provided formats (EPTS raw
  text + metadata XML + SportsCode XML), with a known ground truth. It tests parsing, the frame
  alignment (`video_frame = frame_id - 1`), the halves and the file format end to end. It is not a
  benchmark, because the generator and the model share assumptions.
- The run prints label-free diagnostics: the share of H/A/D, the number and mean length of
  possessions per half, the Smart Tagging sign decisions, and how often each cue fires on graded
  frames. A cue firing on ~0% or ~100% of frames is a bug, not a signal. These checks catch broken
  wiring. They are not targets to tune towards.
- The output is byte-identical across runs. There is no randomness, and the parse cache does not
  change the result.

### Single pipeline check on real data

Run once on the delivered data, with no parameter changed afterwards:
- Ball visible on 59.3% of graded frames, which matches DATA.md, so the frame alignment is correct.
- The Smart Tagging signs were resolved without labels. Every "defending in ..." code and
  "defensive transition" came out as naming the defending side; every attacking code came out as
  naming the side in possession. This is what the code names say, so it confirms the team mapping.
- Public 5-minute window, used as a correctness check: macroF1 0.68, spellF1 0.42, and 21
  predicted possessions against 17 true. The published floor `baseline_smooth50` scores 0.62 / 0.33
  on the same window. With 17 possessions this window cannot rank anything; it only shows the
  labels mean what we think.

### 5b. Video ball detection: tried and dropped

`solution/video_ball.py` is optional and is not used by the submission.
- **Detector:** a public pretrained detector, Ultralytics `yolo11m.pt` (COCO class "sports ball"), with no
  fine-tuning.
- **Frames:** it ran on all 64,643 graded frames where the ATD has no ball, plus every 10th frame where it
  does (9,422 frames, for checking).
- **Crops:** it looks at a 640x640 full-resolution crop centred on the expected ball position. That
  position is interpolated from the ATD ball, or taken from the nearest players' centroid in long gaps,
  and projected into the image with the homography. Detections are mapped back to the pitch with the
  inverse homography.
- **Threshold, chosen without labels** (`solution/video_check.py`): the lowest confidence at which video
  detections agree with the ATD ball within 5 m on at least 85% of frames where both exist. That is
  conf >= 0.20, at 85.8% agreement and a median distance of 0.85 m. At that threshold it fills the ball
  on 13,379 of the 64,643 gap frames (20.7%).
- **Why it was dropped:**
  1. *Reproducibility.* The submission must be regenerated exactly. FP16 GPU inference thresholded at
     0.20 cannot be guaranteed byte-identical on other hardware, while the tracking-only path is pure
     numpy and reproduces exactly.
  2. *A side effect.* Filling ball gaps also broke up the "ball lost at a line" dead-ball cue
     (`edge_missing` fell from 18.7% to 11.7% of graded frames), and that interaction is untested.
  3. *No evidence it helps.* On the public window the two variants differ by about one possession
     (spellF1 0.42 vs 0.40), which is noise. Choosing on that window is not allowed, so it did not decide
     anything.

  Run it with `python -m solution.video_ball`, then `python -m solution.run --video`. The dependencies
  are in `requirements-video.txt`.

### 5c. Calibration on other matches (Metrica sample data)

`solution/calibrate.py` reproduces this.
- **Ground truth from events, by our own mechanical rule.**
  - Possession is the team of the latest PASS, RECOVERY, SET PIECE, SHOT or BALL LOST event, so a
    loose ball stays with the last team, as CHALLENGE.md specifies.
  - Dead ball runs from a stoppage trigger (ball out, foul, goal or shot out, offside or hand ball)
    to the next SET PIECE, and covers the time before each kick-off.
  - This gives 378 and 353 possessions.
- **Tracking degraded to look like the ATD.** Ball visible on 59% of frames (missing in runs of mean
  1.6 s), 0.15 m player jitter and 0.3 m ball jitter, and players dropping out in runs. Seeded.
- **Grid:** 486 combinations of `SWITCH_HA`, `SWITCH_D`, `MIN_SPELL_S`, `CONTROL_GAP_SCALE_M`,
  `K_TEMPO` and `K_EDGE_MISSING`. We chose on one game, scored on the other, then swapped.
- **Rule fixed before running:** replace the current values only if the mean held-out spellF1 gain
  exceeds 0.04.
- **Result:**
  - Current values: spellF1 0.503 and 0.512.
  - Best alternative (the same one in both folds: `SWITCH_HA` 20, `SWITCH_D` 8,
    `CONTROL_GAP_SCALE_M` 0.75, `K_EDGE_MISSING` 0.8): 0.534 and 0.542 held out, a gain of +0.030.
  - That is below the margin, so the values were kept.
  - The check does confirm that the hand-set priors are close to a good region on independent,
    differently-labelled matches.

## 6. Limitations and future work

- **Long ball occlusions in crowded areas** (boxes, aerial duels). With no ball, there's no control
  evidence, and a turnover inside an occlusion is found late or missed. Ball detection on the
  tactical video, using the homography, would fill these gaps.
- **Identity switches and duplicate tracks** add noise to nearest-player distances. Team membership
  per slot is trusted as delivered.
- **Throw-ins carry no team tag.** The restarting team is inferred only from who controls the ball
  next.
- **Parameters are priors, not fitted.** With labelled tracking from other matches, the weights and
  switch costs could be learned (for example a CRF or HMM with duration modelling) instead of set by
  hand.
- **Fouls and offsides inside the pitch** have no "ball out" signal. They rely on tempo and the
  stationary-ball cue, which are weaker.

## 7. Reproduction

```bash
pip install -r requirements-solution.txt      # numpy, pandas -- Python 3.11
# unpack the data bundle into ./data (or set MTAH_DATA_DIR)
python -m solution.run --team <team_name>     # writes submissions/<team_name>.csv and validates it
python evaluation/run_evaluation.py           # optional: score against the public 5 minutes
```

The first run parses the ATD (about a minute) and caches the arrays in `data/cache/`. Later runs
take seconds and produce the same file.
