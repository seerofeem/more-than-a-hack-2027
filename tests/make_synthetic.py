"""Write a synthetic data bundle in the provided formats, plus its ground truth.

It exists to test the pipeline's plumbing -- parsing, frame alignment, halves, file format -- end
to end without the gated data. It is NOT a benchmark: the generator and the model share
assumptions, so a high score here proves the wiring, not the method.

    python tests/make_synthetic.py OUT_DIR
    python -m solution.run --data-dir OUT_DIR --team synthetic --no-copy
    python evaluation/run_evaluation.py --gt OUT_DIR/synthetic_gt.csv --results submissions
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAMES, GAME, asset_paths, half_video_blocks, source_frames  # noqa: E402

N_SLOTS = 46
FPS = 25


def simulate(rng, n_frames, blocks, pitch):
    length, width = pitch
    players = np.full((n_frames, N_SLOTS, 3), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    gt = np.full(n_frames, "X", dtype="<U1")
    events = []

    base = np.zeros((22, 2))
    base[:11, 0] = rng.uniform(10, 50, 11)
    base[11:22, 0] = rng.uniform(55, 95, 11)
    base[:, 1] = rng.uniform(5, 63, 22)

    for half, (start, stop) in enumerate(blocks, 1):
        t = start
        pos = base.copy()
        # Second half: graded before kick-off, like the real one.
        if half == 2:
            pre = 60 * FPS
            gt[t:t + pre] = "D"
            for k in range(pre):
                players[t + k, :22, :2] = np.stack([np.where(np.arange(22) < 11, 30, 75),
                                                    base[:, 1]], axis=1) + rng.normal(0, .3, (22, 2))
                players[t + k, :22, 2] = 0.5
                ball[t + k] = (length / 2, width / 2)
            events.append(("SET PIECES", "N/A", "Kick Off (Start)", t + pre - 50, t + pre + 75))
            t += pre
        team = 0
        while t < stop:
            dur = min(int(rng.exponential(12 * FPS)) + 2 * FPS, stop - t)
            carrier = rng.integers(0, 11) + 11 * team
            for k in range(dur):
                f = t + k
                pos += rng.normal(0, 0.08, pos.shape)
                pos[:, 0] = np.clip(pos[:, 0], 1, length - 1)
                pos[:, 1] = np.clip(pos[:, 1], 1, width - 1)
                if k % 50 == 49:
                    carrier = rng.integers(0, 11) + 11 * team
                players[f, :22, :2] = pos
                players[f, :22, 2] = rng.uniform(1.5, 3.5, 22)
                ball[f] = pos[carrier] + rng.normal(0, 0.3, 2)
                gt[f] = "H" if team == 0 else "A"
            events.append(("BUILD UP", "Barcelona" if team == 0 else "Atletic Club", "",
                           t, t + dur))
            t += dur
            if t >= stop:
                break
            if rng.random() < 0.35:  # ball goes out
                dead = min(int(rng.uniform(5, 20) * FPS), stop - t)
                for k in range(dead):
                    f = t + k
                    players[f, :22, :2] = pos
                    players[f, :22, 2] = rng.uniform(0.2, 1.0, 22)
                    ball[f] = (-1.5, pos[carrier, 1]) if k < dead // 2 else (np.nan, np.nan)
                    gt[f] = "D"
                events.append(("SET PIECES", "N/A", "Throw-in", t + dead - 50, t + dead + 50))
                t += dead
            team = 1 - team
    # Tracker loses the ball in chunks, ~40% of frames.
    f = 0
    while f < n_frames:
        f += int(rng.exponential(60))
        gap = int(rng.exponential(40))
        ball[f:f + gap] = np.nan
        f += gap
    return players, ball, gt, events


def write_meta(path):
    teams = ('<Team id="T1"><Name>FC Barcelona</Name></Team>'
             '<Team id="T2"><Name>Athletic Club</Name></Team>')
    slots = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                    '<ShirtNumber>{0}</ShirtNumber></Player>'
                    .format(i + 1, "T1" if (i < 11 or 22 <= i < 34) else "T2")
                    for i in range(N_SLOTS))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?><Main><Metadata><Teams>{}</Teams>'
                     '<Players>{}</Players></Metadata></Main>'.format(teams, slots))


def write_raw(path, players, ball):
    norm = np.array(GAMES[GAME]["pitch"])
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for f in range(len(ball)):
            groups = []
            for s in range(N_SLOTS):
                x, y, v = players[f, s]
                groups.append("NaN,NaN,NaN" if np.isnan(x) else
                              "{:.5f},{:.5f},{:.2f}".format(x / norm[0], y / norm[1], v))
            bx, by = ball[f]
            b = "NaN,NaN,NaN" if np.isnan(bx) else "{:.5f},{:.5f},0.00".format(
                bx / norm[0], by / norm[1])
            handle.write("{}:{}:{}\n".format(f + 1, ";".join(groups), b))


def write_tags(path, events):
    rows = []
    for i, (code, team, typ, start, stop) in enumerate(events):
        labels = '<label><group>Team</group><text>{}</text></label>'.format(team)
        if typ:
            labels += '<label><group>Type</group><text>{}</text></label>'.format(typ)
        rows.append('<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>{}'
                    '</instance>'.format(i, start / FPS, stop / FPS, code, labels))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?><file><ALL_INSTANCES>{}'
                     '</ALL_INSTANCES></file>'.format("".join(rows)))


def main(out_dir):
    rng = np.random.default_rng(0)
    n = source_frames("tactical")
    blocks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    players, ball, gt, events = simulate(rng, n, blocks, GAMES[GAME]["pitch"])
    paths = asset_paths(data_dir=out_dir)
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    write_meta(paths["atd_meta"])
    write_raw(paths["atd_raw"], players, ball)
    write_tags(paths["smart_tagging"], events)
    with open(os.path.join(out_dir, "synthetic_gt.csv"), "w", encoding="utf-8",
              newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, g) for i, g in enumerate(gt))
    print("synthetic bundle in {}".format(out_dir))


if __name__ == "__main__":
    main(sys.argv[1])
