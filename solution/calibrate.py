"""Calibrate the segmentation parameters on OTHER games: Metrica's public sample data.

    python -m solution.calibrate            # needs data/external/ (see download() below)

Why: the rules forbid tuning on the provided game (its public 5 minutes included). Labels on
other games are allowed. Metrica publishes two full matches of tracking + human-coded event data
(github.com/metrica-sports/sample-data, Sample_Game_1 and _2). From those events we derive a
possession ground truth mechanically, degrade the clean tracking so it looks like the automatic
ATD of the challenge, run our pipeline, and grid-search the few coarse parameters that the ranking
metric is most sensitive to. Two-fold: choose on one game, report on the other.

Ground truth from events (our derivation, not the organisers'):
  * possession = team of the most recent PASS / RECOVERY / SET PIECE / SHOT / BALL LOST event
    (a loose ball stays with the last team, as CHALLENGE.md specifies);
  * dead = from a stoppage trigger (BALL OUT, FAULT RECEIVED, foul CHALLENGE, goal or shot out,
    offside / hand ball) to the start of the next SET PIECE; and before each period's kick-off.

Degradation (seeded): ball removed on ~41% of frames in runs (the ATD rate), position jitter,
player dropout in runs. Everything printed is aggregate.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import DATA_DIR  # noqa: E402
import possession_metrics as pm  # noqa: E402

from . import evidence, features, segment  # noqa: E402
from . import params as P  # noqa: E402

EXTERNAL = os.path.join(DATA_DIR, "external")
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
GAMES = (1, 2)
PITCH = (105.0, 68.0)
FPS = 25
SEED = 0

TARGET_BALL_VISIBLE = 0.593   # ATD ball visible on 59.3% of graded frames (DATA.md)
BALL_MISSING_RUN_S = 1.6
PLAYER_JITTER_M = 0.15
BALL_JITTER_M = 0.3
PLAYER_MISSING_SHARE = 0.07

GRID = {
    "SWITCH_HA": (5.0, 10.0, 20.0),
    "SWITCH_D": (8.0, 15.0, 30.0),
    "MIN_SPELL_S": (0.5, 1.0),
    "CONTROL_GAP_SCALE_M": (0.75, 1.5, 3.0),
    "K_TEMPO": (0.0, 0.3, 0.6),
    "K_EDGE_MISSING": (0.0, 0.4, 0.8),
}
EVIDENCE_KEYS = ("CONTROL_GAP_SCALE_M", "K_TEMPO", "K_EDGE_MISSING")
ADOPT_MARGIN = 0.04   # held-out spellF1 gain needed to replace the current values


def download():
    """Fetch the two sample games into data/external/ (public, ~160 MB)."""
    import urllib.request
    os.makedirs(EXTERNAL, exist_ok=True)
    for g in GAMES:
        for kind in ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team"):
            name = "Sample_Game_{}_{}.csv".format(g, kind)
            path = os.path.join(EXTERNAL, name)
            if not os.path.isfile(path):
                urllib.request.urlretrieve("{}/Sample_Game_{}/{}".format(BASE_URL, g, name), path)


def read_tracking(path):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(raw.columns)
    period = raw["Period"].to_numpy()
    frame = raw["Frame"].to_numpy()
    players = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            players.append(raw.iloc[:, [i, i + 1]].to_numpy(dtype=np.float64))
    b = cols.index("Ball")
    ball = raw.iloc[:, [b, b + 1]].to_numpy(dtype=np.float64)
    return period, frame, np.stack(players, axis=1), ball


def ground_truth(events, n, period):
    """Per-frame 'H'/'A'/'D' (and 'X' outside periods) derived from the event data."""
    team_code = {"Home": "H", "Away": "A"}
    ev = events.sort_values(["Start Frame", "End Frame"]).reset_index(drop=True)
    sub = ev["Subtype"].fillna("").str.upper()
    typ = ev["Type"].str.upper()
    start = ev["Start Frame"].to_numpy() - 1
    end = np.maximum(ev["End Frame"].to_numpy() - 1, start)

    labels = np.full(n, "X", dtype="<U1")
    owner = np.full(n, "", dtype="<U1")
    owning = typ.isin(["PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"]).to_numpy()
    marks = sorted((int(start[i]), team_code[ev["Team"][i]]) for i in np.flatnonzero(owning))
    for (f, t), nxt in zip(marks, marks[1:] + [(n, "")]):
        owner[f:nxt[0]] = t

    trigger = ((typ == "BALL OUT") | (typ == "FAULT RECEIVED")
               | ((typ == "CHALLENGE") & sub.str.contains("FAULT"))
               | ((typ == "SHOT") & (sub.str.contains("GOAL") | sub.str.contains("OUT")))
               | ((typ == "BALL LOST") & sub.str.contains("OFFSIDE|HAND BALL|END HALF"))).to_numpy()
    dead = np.zeros(n, dtype=bool)
    set_pieces = np.flatnonzero((typ == "SET PIECE").to_numpy())
    for k in set_pieces:
        before = [i for i in np.flatnonzero(trigger) if start[i] <= start[k] and i < k]
        if before and start[k] - end[before[-1]] < 120 * FPS:
            dead[end[before[-1]]:start[k]] = True

    for p in np.unique(period):
        frames = np.flatnonzero(period == p)
        first, last = frames[0], frames[-1] + 1
        kick = [start[k] for k in set_pieces
                if "KICK OFF" in sub[k] and first <= start[k] < last]
        if kick:
            dead[first:min(kick)] = True
        labels[first:last] = np.where(dead[first:last], "D", owner[first:last])
    labels[(labels == "")] = "D"
    return labels


def degrade(rng, players, ball):
    n = len(ball)
    already = np.mean(~np.isnan(ball[:, 0]))
    share = float(np.clip(1 - TARGET_BALL_VISIBLE / max(already, 1e-6), 0.0, 0.9))
    players = players + rng.normal(0, PLAYER_JITTER_M, players.shape) / np.array(PITCH)
    ball = ball + rng.normal(0, BALL_JITTER_M, ball.shape) / np.array(PITCH)
    mean_missing = BALL_MISSING_RUN_S * FPS
    mean_visible = mean_missing * (1 - share) / max(share, 1e-6)
    t, visible = 0, True
    while t < n:
        run = int(rng.geometric(1.0 / (mean_visible if visible else mean_missing)))
        if not visible:
            ball[t:t + run] = np.nan
        t += run
        visible = not visible
    for s in range(players.shape[1]):
        t = 0
        while t < n:
            t += int(rng.geometric(1.0 / (60 * FPS)))
            run = int(rng.geometric(1.0 / (60 * FPS * PLAYER_MISSING_SHARE)))
            players[t:t + run, s] = np.nan
            t += run
    return players, ball


def prepare(game, rng):
    folder = EXTERNAL
    period, _, home, ball = read_tracking(os.path.join(
        folder, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game)))
    _, _, away, _ = read_tracking(os.path.join(
        folder, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)))
    events = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game)))
    n = len(ball)
    gt = ground_truth(events, n, period)

    pos = np.concatenate([home, away], axis=1)
    # The ATD exports a smoothed speed per slot; emulate it from the clean positions (0.2 s span)
    # before jitter is added, so tempo is not drowned in per-frame noise.
    metres = pos * np.array(PITCH)
    speed = np.full(pos.shape[:2], np.nan)
    speed[5:] = np.hypot(*(metres[5:] - metres[:-5]).transpose(2, 0, 1)) * FPS / 5
    pos, ball = degrade(rng, pos, ball)
    speed[np.isnan(pos[:, :, 0])] = np.nan
    players = np.concatenate([pos, speed[:, :, None]], axis=2)
    team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    blocks = []
    for p in np.unique(period):
        frames = np.flatnonzero(period == p)
        blocks.append((int(frames[0]), int(frames[-1]) + 1))
    feat = features.compute(players.astype(np.float32), ball.astype(np.float32), team, PITCH, FPS)
    return feat, blocks, gt


def score(gt, pred):
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(g)
    spell = pm.spell_scores(g, pm.fill_abstentions(p, runs), runs)
    cm = pm.scored_confusion(pm.confusion(g, p))
    return spell["mean_f1"], pm.macro_f1(cm), spell["n_gt"], spell["n_pred"]


def main():
    download()
    rng = np.random.default_rng(SEED)
    data = {}
    for g in GAMES:
        feat, blocks, gt = prepare(g, rng)
        data[g] = (feat, blocks, gt)
        spells = pm.possession_spells(pm.label_codes(gt), pm.valid_runs(pm.label_codes(gt)))
        print("game {}: {} event-derived possessions, D {:.1f}%, ball visible {:.1f}%".format(
            g, len(spells), 100 * np.mean(gt[gt != "X"] == "D"),
            100 * np.mean(feat["visible"])))

    original = {k: getattr(P, k) for k in GRID}
    table = []
    seg_keys = [k for k in GRID if k not in EVIDENCE_KEYS]
    for ev_combo in itertools.product(*(GRID[k] for k in EVIDENCE_KEYS)):
        for k, v in zip(EVIDENCE_KEYS, ev_combo):
            setattr(P, k, v)
        built = {g: evidence.build(data[g][0], [], data[g][1], FPS)[0] for g in GAMES}
        for seg_combo in itertools.product(*(GRID[k] for k in seg_keys)):
            for k, v in zip(seg_keys, seg_combo):
                setattr(P, k, v)
            values = dict(zip(EVIDENCE_KEYS, ev_combo), **dict(zip(seg_keys, seg_combo)))
            row = [values[k] for k in GRID]
            for g in GAMES:
                _, blocks, gt = data[g]
                row.extend(score(gt, segment.label(built[g], blocks, len(gt))))
            table.append(row)
    for k, v in original.items():
        setattr(P, k, v)

    cols = list(GRID) + ["g1_spell", "g1_macro", "g1_ngt", "g1_npred",
                         "g2_spell", "g2_macro", "g2_ngt", "g2_npred"]
    df = pd.DataFrame(table, columns=cols)
    df["mean_spell"] = (df["g1_spell"] + df["g2_spell"]) / 2
    current = df[np.logical_and.reduce([np.isclose(df[k], original[k]) for k in GRID])]
    pd.set_option("display.width", 200)
    print("\ncurrent params:")
    print(current.to_string(index=False, float_format="%.3f"))
    print("\ntop 10 by mean spellF1:")
    print(df.sort_values("mean_spell", ascending=False).head(10).to_string(
        index=False, float_format="%.3f"))
    gains = []
    for train, test in ((1, 2), (2, 1)):
        best = df.sort_values("g{}_spell".format(train), ascending=False).iloc[0]
        held = float(best["g{}_spell".format(test)])
        base = float(current["g{}_spell".format(test)].iloc[0])
        gains.append(held - base)
        print("\nchoose on game {} -> {}: game {} spellF1 {:.3f} (current {:.3f})".format(
            train, {k: float(best[k]) for k in GRID}, test, held, base))
    gain = float(np.mean(gains))
    print("\nmean held-out gain {:+.3f} -> {}".format(
        gain, "ADOPT" if gain > ADOPT_MARGIN
        else "keep current values (below the {:.2f} margin)".format(ADOPT_MARGIN)))
    df.to_csv(os.path.join(EXTERNAL, "calibration_grid.csv"), index=False)


if __name__ == "__main__":
    main()
