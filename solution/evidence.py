"""Per-frame scores for the three states H, A, D.

A score is additive evidence in arbitrary but consistent units: the segmenter sums it over time
and pays a switch cost to change state. Positive means "this frame looks like that state".
"""

import numpy as np

from . import params as P

H, A, D = 0, 1, 2


def control_term(feat):
    """Signed per-frame control evidence: + home closer to a playable ball, - away closer."""
    d_home, d_away = feat["d_home"], feat["d_away"]
    nearest = np.minimum(d_home, d_away)
    finite = np.isfinite(nearest)
    weight = np.where(finite, np.exp(-np.maximum(nearest - P.CONTROL_RADIUS_M, 0.0)
                                     / P.CONTROL_DECAY_M), 0.0)
    with np.errstate(invalid="ignore"):
        gap = np.tanh((d_away - d_home) / P.CONTROL_GAP_SCALE_M)  # inf - finite -> +-1
    gap = np.where(np.isnan(gap), 0.0, gap)
    flying = np.nan_to_num(feat["ball_speed"], nan=0.0) > P.BALL_FLIGHT_SPEED_MS
    weight = np.where(flying, weight * P.FLIGHT_WEIGHT, weight)
    weight = np.where(feat["out_of_bounds"], 0.0, weight)  # a thrower out of play is not control
    return P.K_CONTROL * weight * gap


def tempo_term(tempo, blocks):
    """+1 slow (dead-ish) .. -1 fast (live), calibrated on each half's own quantiles."""
    out = np.zeros(len(tempo))
    for start, stop in blocks:
        window = tempo[start:stop]
        ok = ~np.isnan(window)
        if ok.sum() < 100:
            continue
        slow, fast = np.quantile(window[ok], [P.TEMPO_SLOW_Q, P.TEMPO_FAST_Q])
        spread = max(fast - slow, 1e-6)
        mid = 0.5 * (slow + fast)
        out[start:stop] = np.where(ok, np.clip(2.0 * (mid - window) / spread, -1.0, 1.0), 0.0)
    return out


def team_code_signs(events, control):
    """For each team-tagged phase code: +1 if Team names the side in possession, -1 if it names
    the defending side. Decided by agreement with the (label-free) proximity evidence."""
    agreement = {}
    for ev in events:
        if ev["side"] is None or ev["code"] in ("set pieces", "goals"):
            continue
        piece = control[ev["start"]:ev["stop"]]
        if not len(piece):
            continue
        vote = float(piece.mean()) * (1.0 if ev["side"] == 0 else -1.0)
        agreement[ev["code"]] = agreement.get(ev["code"], 0.0) + vote
    return {code: (1 if total >= 0 else -1) for code, total in agreement.items()}


def block_of(frame, blocks):
    for start, stop in blocks:
        if start <= frame < stop:
            return start, stop
    return None


def build(feat, events, blocks, fps=P.FPS):
    """(scores[T, 3], diagnostics dict)."""
    n = len(feat["visible"])
    scores = np.zeros((n, 3))

    control = control_term(feat)
    scores[:, H] += control
    scores[:, A] -= control

    live = feat["visible"] & ~feat["out_of_bounds"] & \
        (np.nan_to_num(feat["ball_speed"], nan=0.0) > P.BALL_MOVING_MS)
    scores[:, D] -= P.K_ALIVE * live
    scores[:, D] += P.K_OUT * feat["out_of_bounds"]
    scores[:, D] += P.K_EDGE_MISSING * feat["edge_missing"]

    tempo = tempo_term(feat["tempo"], blocks)
    scores[:, D] += P.K_TEMPO * tempo
    scores[:, D] += P.K_STATIONARY * (feat["long_stationary"] & (tempo > 0))
    scores[:, D] += P.K_FORMATION * feat["formation"]

    signs = team_code_signs(events, control)
    goals = sorted(ev["stop"] for ev in events if ev["code"] == "goals")
    for ev in events:
        start, stop = max(ev["start"], 0), min(ev["stop"], n)
        if start >= stop:
            continue
        if ev["code"] in signs and ev["side"] is not None:
            side = ev["side"] if signs[ev["code"]] > 0 else 1 - ev["side"]
            scores[start:stop, side] += P.K_TAG_TEAM
        if ev["code"] == "set pieces":
            scores[start:(start + stop) // 2 + 1, D] += P.K_TAG_SETPIECE
            if ev["type"].startswith("kick off"):
                block = block_of(start, blocks)
                if ev["type"] == "kick off (start)" and block:
                    # Graded frames before a half's tagged kick-off are pre-match: dead ball.
                    scores[block[0]:start, D] += P.K_TAG_KICKOFF
                elif ev["type"] == "kick off (after goal)":
                    before = [g for g in goals if g <= start
                              and start - g <= P.GOAL_TO_KICKOFF_MAX_S * fps]
                    if before:
                        scores[before[-1]:start, D] += P.K_TAG_GOAL

    return scores, {"tag_code_signs": signs}
