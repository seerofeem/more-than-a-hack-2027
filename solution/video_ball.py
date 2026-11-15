"""Optional: detect the ball in the tactical video where the ATD has lost it.

    python -m solution.video_ball            # resumable; checkpoints into data/cache/

A public pretrained detector (Ultralytics YOLO, COCO class 32 "sports ball"; no fine-tuning, no
labels from this game) runs on:
  * every graded frame where the ATD ball is missing -- the frames we want to fill, and
  * every VALIDATION_STRIDE-th graded frame where the ATD ball is present -- a label-free check:
    the detector is compared against the automatic tracking, never against a ground truth.

Speed: instead of the whole 1920x1080 frame, the detector sees one CROP x CROP window at full
resolution, centred where the ball is expected:
  * ATD ball gap shorter than NEAR_S on either side: linear interpolation between the last and next
    ATD ball positions;
  * longer gaps: the centroid of the players nearest to that interpolated point (a ball the tracker
    lost for a while is usually inside a crowd).
For validation frames the centre is the ATD ball NEAR_S earlier, not the ball itself, so the check
is not flattered by a crop centred on the answer.

Each detection's pixel centre is mapped to pitch coordinates with the provided per-frame homography.
Detections far outside the pitch are dropped; the most confident remaining one is kept. Output:
data/cache/video_ball_<model>.npz (x, y normalised, conf, processed). All of it stays in data/
(gitignored); frame-level positions are never published.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

from . import atd  # noqa: E402

MODEL = "yolo11m.pt"
CROP = 640
MIN_CONF = 0.05          # keep weak detections; the fusion step decides what to trust
SPORTS_BALL = 32
PITCH_MARGIN = 0.03      # normalised units beyond the lines still accepted (a ball just out)
VALIDATION_STRIDE = 10
NEAR_S = 0.5
CROWD = 6                # players averaged for the long-gap centre
BATCH = 32
CHECKPOINT_EVERY = 8000


def cache_path(model=MODEL, data_dir=None):
    return os.path.join(data_dir or DATA_DIR, "cache",
                        "video_ball_{}.npz".format(os.path.splitext(model)[0]))


def frames_to_process(ball, n_frames):
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    seen = ~np.isnan(ball[:, 0])
    check = np.zeros(n_frames, dtype=bool)
    check[np.flatnonzero(graded & seen)[::VALIDATION_STRIDE]] = True
    return (graded & ~seen) | check, check


def expected_centres(ball, players, check, fps):
    """Expected ball position (normalised pitch) for every frame."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    near = int(NEAR_S * fps)

    centre = np.full((n, 2), np.nan)
    has_prev, has_next = prev >= 0, nxt < n
    p = np.where(has_prev, prev, 0)
    q = np.where(has_next, nxt, 0)
    span = np.maximum(q - p, 1)
    w = np.clip((idx - p) / span, 0, 1)[:, None]
    both = has_prev & has_next
    centre[both] = ((1 - w) * ball[p] + w * ball[q])[both]
    centre[has_prev & ~has_next] = ball[p][has_prev & ~has_next]
    centre[~has_prev & has_next] = ball[q][~has_prev & has_next]

    # Validation frames: pretend the ball is unknown, use where it was NEAR_S earlier.
    back = np.clip(idx - near, 0, n - 1)
    use_back = check & seen[back]
    centre[use_back] = ball[back][use_back]

    # Long gaps: centroid of the players nearest the interpolated point.
    gap = np.minimum(np.where(has_prev, idx - p, n), np.where(has_next, q - idx, n))
    far = ~seen & (gap > near)
    for t in np.flatnonzero(far | np.isnan(centre[:, 0])):
        pts = players[t, :, :2]
        pts = pts[~np.isnan(pts[:, 0])]
        if not len(pts):
            continue
        ref = centre[t] if not np.isnan(centre[t, 0]) else pts.mean(axis=0)
        order = np.argsort(np.hypot(pts[:, 0] - ref[0], (pts[:, 1] - ref[1]) * 0.65))
        centre[t] = pts[order[:CROWD]].mean(axis=0)
    return centre


def project(h, x, y):
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--limit", type=int, default=0, help="stop after N processed frames (test)")
    args = parser.parse_args(argv)

    import cv2
    import torch
    from ultralytics import YOLO
    import load_homography

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    video, meta, raw = require(["video", "atd_meta", "atd_raw"], GAME, args.data_dir)
    n = source_frames("tactical")
    fps = GAMES[GAME]["fps"]
    players, ball, _ = atd.load(meta, raw, n,
                                cache_dir=os.path.join(args.data_dir or DATA_DIR, "cache"))
    todo, check = frames_to_process(ball, n)
    centre = expected_centres(ball.astype(np.float64), players, check, fps)
    matrices, _, valid = load_homography.load(data_dir=args.data_dir)

    path = cache_path(args.model, args.data_dir)
    out_x = np.full(n, np.nan, dtype=np.float32)
    out_y = np.full(n, np.nan, dtype=np.float32)
    out_c = np.zeros(n, dtype=np.float32)
    processed = np.zeros(n, dtype=bool)
    if os.path.isfile(path) and not args.limit:
        with np.load(path) as z:
            if z.get("crop", np.array(0)) == CROP:
                out_x, out_y, out_c, processed = z["x"], z["y"], z["conf"], z["processed"]
                print("resuming: {} frames already done".format(int(processed.sum())))

    def save():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, x=np.round(out_x, 5), y=np.round(out_y, 5), conf=np.round(out_c, 4),
                 processed=processed, crop=np.array(CROP))

    model = YOLO(args.model)
    device = 0 if torch.cuda.is_available() else "cpu"
    remaining = todo & ~processed
    total = int(remaining.sum()) if not args.limit else min(args.limit, int(remaining.sum()))
    print("detecting on {} frames, {}x{} crops, {} on {}".format(total, CROP, CROP, args.model,
                                                                 device), flush=True)

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise SystemExit("cannot open {}".format(video))
    started, done, last_saved = time.time(), 0, 0
    batch = []   # (frame index, crop image, x0, y0)

    def flush():
        results = model.predict([b[1] for b in batch], imgsz=CROP, conf=MIN_CONF,
                                classes=[SPORTS_BALL], device=device, half=device != "cpu",
                                verbose=False)
        for (index, _, x0, y0), result in zip(batch, results):
            processed[index] = True
            if result.boxes is None or not len(result.boxes):
                continue
            h_inv = np.linalg.inv(matrices[index])
            best = -1.0
            for (u, v, _, _), c in zip(result.boxes.xywh.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy()):
                x, y = project(h_inv, float(u) + x0, float(v) + y0)
                inside = (-PITCH_MARGIN <= x <= 1 + PITCH_MARGIN and
                          -PITCH_MARGIN <= y <= 1 + PITCH_MARGIN)
                if inside and c > best:
                    best = float(c)
                    out_x[index], out_y[index], out_c[index] = x, y, c
        batch.clear()

    last_needed = int(np.flatnonzero(remaining)[-1]) if remaining.any() else -1
    for index in range(last_needed + 1):
        if not remaining[index]:
            if not capture.grab():
                break
            continue
        ok, image = capture.read()
        if not ok:
            break
        if not valid[index] or np.isnan(centre[index, 0]):
            processed[index] = True
            continue
        u, v = project(matrices[index], centre[index, 0], centre[index, 1])
        height, width = image.shape[:2]
        x0 = int(np.clip(round(u) - CROP // 2, 0, width - CROP)) if np.isfinite(u) else 0
        y0 = int(np.clip(round(v) - CROP // 2, 0, height - CROP)) if np.isfinite(v) else 0
        batch.append((index, image[y0:y0 + CROP, x0:x0 + CROP], x0, y0))
        if len(batch) == BATCH:
            flush()
            done += BATCH
            if done - last_saved >= CHECKPOINT_EVERY:
                save()
                last_saved = done
                rate = done / (time.time() - started)
                print("  {}/{}  {:.0f} fps  eta {:.0f} min".format(
                    done, total, rate, (total - done) / max(rate, 1e-6) / 60), flush=True)
        if args.limit and done >= args.limit:
            break
    if batch:
        flush()
    capture.release()
    save()
    print("saved {}  ({} processed, {} with a detection)".format(
        path, int(processed.sum()), int((out_c > 0).sum())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
