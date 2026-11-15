"""Per-frame physical features from the tracking, all in metres and seconds.

Nothing here knows about labels. Every feature is a fact about the geometry of one frame or a
short window around it.
"""

import numpy as np

from . import params as P

HOME, AWAY = 0, 1


def rolling_nanmean(values, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN. NaN where the window is all NaN."""
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def run_lengths(mask):
    """For each frame, the length of the run of True it sits in (0 where False)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(len(mask), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = stop - start
    return out


def compute(players, ball, team, pitch, fps=P.FPS):
    """dict of per-frame arrays. players[T, S, 3] and ball[T, 2] are normalised 0-1."""
    length, width = pitch
    n = len(ball)

    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    bx = ball[:, 0].astype(np.float64) * length
    by = ball[:, 1].astype(np.float64) * width
    visible = ~(np.isnan(bx) | np.isnan(by))
    present = ~(np.isnan(px) | np.isnan(py))

    # Nearest player of each side to the ball.
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(present & visible[:, None], dist, np.inf)
    d_home = np.where(team[None, :] == HOME, dist, np.inf).min(axis=1)
    d_away = np.where(team[None, :] == AWAY, dist, np.inf).min(axis=1)

    # Ball speed as net displacement of a smoothed position over ~0.5 s. Frame-to-frame
    # differences would be dominated by tracking jitter and never read "stationary".
    half_span = max(1, int(round(0.25 * fps)))
    sx = rolling_nanmean(np.where(visible, bx, np.nan), 2)
    sy = rolling_nanmean(np.where(visible, by, np.nan), 2)
    ball_speed = np.full(n, np.nan)
    if n > 2 * half_span:
        moved = np.hypot(sx[2 * half_span:] - sx[:-2 * half_span],
                         sy[2 * half_span:] - sy[:-2 * half_span])
        ball_speed[half_span:n - half_span] = moved * fps / (2 * half_span)
    ball_speed[~visible] = np.nan

    out_of_bounds = visible & ((bx < -P.OUT_MARGIN_M) | (bx > length + P.OUT_MARGIN_M) |
                               (by < -P.OUT_MARGIN_M) | (by > width + P.OUT_MARGIN_M))
    near_edge = visible & ((bx < P.EDGE_MARGIN_M) | (bx > length - P.EDGE_MARGIN_M) |
                           (by < P.EDGE_MARGIN_M) | (by > width - P.EDGE_MARGIN_M))

    # Ball lost while at a line: mark the whole missing run that follows.
    edge_missing = np.zeros(n, dtype=bool)
    edges = np.diff(np.concatenate(([0], (~visible).astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        before = start - 1
        after = stop
        left_at_line = before >= 0 and near_edge[before]
        back_at_line = after < n and near_edge[after]
        if left_at_line or back_at_line:
            edge_missing[start:stop] = True

    # Tempo: mean speed of tracked players. Prefer the exported speed; derive it where missing.
    speed = players[:, :, 2].astype(np.float64)
    derived = np.full_like(px, np.nan)
    derived[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * fps
    speed = np.where(np.isnan(speed), derived, speed)
    masked = np.where(present, np.nan_to_num(speed, nan=0.0), 0.0)
    n_present = present.sum(axis=1)
    per_frame = np.where(n_present > 0, masked.sum(axis=1) / np.maximum(n_present, 1), np.nan)
    tempo = rolling_nanmean(per_frame, int(P.TEMPO_WINDOW_S * fps / 2))

    stationary = visible & (np.nan_to_num(ball_speed, nan=np.inf) < P.STATIONARY_MS)
    long_stationary = run_lengths(stationary) >= int(P.STATIONARY_S * fps)

    # Kick-off formation: ball on the spot, each team entirely in one half, opposite halves.
    at_centre = visible & (np.hypot(bx - length / 2, by - width / 2) < P.CENTRE_RADIUS_M)
    left = present & (px < length / 2)
    home_n = (present & (team[None, :] == HOME)).sum(axis=1)
    away_n = (present & (team[None, :] == AWAY)).sum(axis=1)
    home_left = (left & (team[None, :] == HOME)).sum(axis=1) / np.maximum(home_n, 1)
    away_left = (left & (team[None, :] == AWAY)).sum(axis=1) / np.maximum(away_n, 1)
    share = P.FORMATION_SHARE
    split = ((home_left >= share) & (away_left <= 1 - share)) | \
            ((home_left <= 1 - share) & (away_left >= share))
    formation = at_centre & split & (home_n >= 7) & (away_n >= 7)

    return {
        "visible": visible,
        "d_home": d_home,
        "d_away": d_away,
        "ball_speed": ball_speed,
        "out_of_bounds": out_of_bounds,
        "edge_missing": edge_missing,
        "tempo": tempo,
        "long_stationary": long_stationary,
        "formation": formation,
    }
