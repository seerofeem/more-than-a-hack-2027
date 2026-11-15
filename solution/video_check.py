"""Label-free check of the video ball detector against the automatic tracking.

    python -m solution.video_check

On the validation frames (ATD ball present) it reports, per confidence bin, how often the detector
finds a ball and how far that ball is from the ATD ball. This is how the fusion confidence threshold
(params.VIDEO_MIN_CONF) is chosen: no ground truth is involved, only two automatic systems agreeing.
Prints aggregates only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

from . import atd, video_ball  # noqa: E402


def main():
    meta, raw = require(["atd_meta", "atd_raw"], GAME)
    n = source_frames("tactical")
    _, ball, _ = atd.load(meta, raw, n, cache_dir=os.path.join(DATA_DIR, "cache"))
    with np.load(video_ball.cache_path()) as z:
        vx, vy, conf, processed = z["x"], z["y"], z["conf"], z["processed"]
    length, width = GAMES[GAME]["pitch"]

    atd_seen = ~np.isnan(ball[:, 0])
    check = processed & atd_seen
    fill = processed & ~atd_seen
    print("validation frames (ATD ball present): {}".format(int(check.sum())))
    print("fill frames (ATD ball missing):       {}".format(int(fill.sum())))
    dist = np.hypot((vx - ball[:, 0]) * length, (vy - ball[:, 1]) * width)

    print("\nconf>=   det_rate  within2m  within5m  median_m   fill_frames_gained")
    for thr in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        det = check & (conf >= thr)
        d = dist[det]
        gained = int((fill & (conf >= thr)).sum())
        print("{:>6.2f}   {:>7.1%}  {:>8.1%}  {:>8.1%}  {:>8.2f}   {:>8} ({:.1%} of fill)".format(
            thr, det.sum() / max(check.sum(), 1),
            np.mean(d < 2) if len(d) else 0, np.mean(d < 5) if len(d) else 0,
            np.median(d) if len(d) else float("nan"), gained, gained / max(fill.sum(), 1)))


if __name__ == "__main__":
    main()
