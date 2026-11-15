"""Turn per-frame scores into possessions: Viterbi with switch costs, then spell clean-up.

The ranking metric scores whole possessions, so the segmenter's job is to say "same as before"
unless the evidence for a change is sustained. A switch cost does exactly that: changing state
must be paid for by accumulated evidence, so one noisy frame cannot split a possession.
"""

import numpy as np

from . import params as P

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def switch_costs():
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = P.SWITCH_HA
    cost[H, D] = cost[A, D] = cost[D, H] = cost[D, A] = P.SWITCH_D
    return cost


def viterbi(scores, cost):
    """Best state path maximising sum(scores[t, s_t]) - sum(cost[s_{t-1}, s_t])."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, 3), dtype=np.int8)
    value = scores[0].copy()
    for t in range(1, n):
        cand = value[:, None] - cost          # [prev, next]
        best = cand.argmax(axis=0)
        back[t] = best
        value = cand[best, np.arange(3)] + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(value.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(path):
    edges = np.flatnonzero(path[1:] != path[:-1]) + 1
    bounds = np.concatenate(([0], edges, [len(path)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(path[bounds[i]]))
            for i in range(len(bounds) - 1)]


def clean(path, fps=P.FPS):
    """Close short dead gaps inside a possession and absorb blink-length possessions."""
    path = path.copy()
    min_dead = int(round(P.MIN_DEAD_S * fps))
    min_spell = int(round(P.MIN_SPELL_S * fps))
    for _ in range(10):
        changed = False
        runs = _runs(path)
        for i, (start, stop, state) in enumerate(runs):
            if state != D or stop - start >= min_dead or i == 0 or i == len(runs) - 1:
                continue
            before, after = runs[i - 1][2], runs[i + 1][2]
            if before == after:
                path[start:stop] = before
            else:
                middle = (start + stop) // 2
                path[start:middle] = before
                path[middle:stop] = after
            changed = True
        runs = _runs(path)
        for i, (start, stop, state) in enumerate(runs):
            if state == D or stop - start >= min_spell or len(runs) == 1:
                continue
            path[start:stop] = runs[i - 1][2] if i > 0 else runs[i + 1][2]
            changed = True
        if not changed:
            break
    return path


def label(scores, blocks, n_frames):
    """'H'/'A'/'D' inside graded blocks, 'X' everywhere else."""
    out = np.full(n_frames, "X", dtype="<U1")
    cost = switch_costs()
    for start, stop in blocks:
        path = clean(viterbi(scores[start:stop], cost))
        out[start:stop] = LETTERS[path]
    return out
