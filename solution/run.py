"""Regenerate the submission from the provided data.

    python -m solution.run --team <team_name>

Writes submissions/<team_name>.csv (and a copy in evaluation/results/ so run_evaluation.py picks
it up), then runs the official validator on the exact file written.
"""

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

from . import atd, evidence, features, segment, tagging  # noqa: E402
from . import params as P  # noqa: E402


def graded_blocks(game=GAME):
    """Half-open [start, stop) video-frame spans of the two halves (published in games.py)."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def predict(data_dir=None, game=GAME, verbose=True):
    meta, raw, tags_path = require(["atd_meta", "atd_raw", "smart_tagging"], game, data_dir)
    n_frames = source_frames("tactical", game)
    fps = GAMES[game]["fps"]
    blocks = graded_blocks(game)

    players, ball, team = atd.load(meta, raw, n_frames,
                                   cache_dir=os.path.join(os.path.dirname(os.path.dirname(meta)),
                                                          "cache"),
                                   verbose=verbose)
    feat = features.compute(players, ball, team, GAMES[game]["pitch"], fps)
    events = tagging.load(tags_path, fps)
    scores, diag = evidence.build(feat, events, blocks, fps)
    labels = segment.label(scores, blocks, n_frames)
    if verbose:
        report(labels, feat, blocks, diag, events, fps)
    return labels


def report(labels, feat, blocks, diag, events, fps):
    """Label-free diagnostics only: counts and shares, never anything about a ground truth."""
    graded = np.zeros(len(labels), dtype=bool)
    for start, stop in blocks:
        graded[start:stop] = True
    print("smart tagging: {} instances".format(len(events)))
    print("ball visible on {:.1f}% of graded frames".format(100 * feat["visible"][graded].mean()))
    print("tag code -> Team names the side {}".format(
        ", ".join("{}={}".format(c, "in possession" if s > 0 else "defending")
                  for c, s in sorted(diag["tag_code_signs"].items()))))
    for half, (start, stop) in enumerate(blocks, 1):
        window = labels[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        spells = [(bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)
                  if window[bounds[i]] in ("H", "A")]
        print("half {}: H {:.1f}%  A {:.1f}%  D {:.1f}%  possessions {}  mean {:.1f} s".format(
            half, 100 * np.mean(window == "H"), 100 * np.mean(window == "A"),
            100 * np.mean(window == "D"), len(spells),
            (np.mean(spells) / fps) if spells else 0.0))


def write(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def validate(path, n_frames):
    import possession_metrics as pm
    reason = pm.validate_submission(pm.read_label_csv(path), n_frames)
    if reason:
        raise SystemExit("INVALID submission {}: {}".format(path, reason))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="submission",
                        help="team name; the file is written as <team>.csv")
    parser.add_argument("--data-dir", default=None,
                        help="data bundle root (default: ./data or $MTAH_DATA_DIR)")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "submissions"))
    parser.add_argument("--no-copy", action="store_true",
                        help="do not copy the file into evaluation/results/")
    args = parser.parse_args(argv)

    labels = predict(args.data_dir)
    path = write(os.path.join(args.out_dir, "{}.csv".format(args.team)), labels)
    validate(path, source_frames("tactical"))
    print("wrote and validated {}".format(os.path.relpath(path, ROOT)))
    if not args.no_copy:
        copy = os.path.join(ROOT, "evaluation", "results", os.path.basename(path))
        shutil.copyfile(path, copy)
        print("copied to {} -- score with: python evaluation/run_evaluation.py"
              .format(os.path.relpath(copy, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
