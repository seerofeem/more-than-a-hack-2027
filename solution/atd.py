"""Stream the raw FIFA/EPTS ATD into numpy arrays.

Raw line format (same as starter/baseline.py): "<frame_id>:<slot>;<slot>;...:<ball>", each slot
"x,y,speed" with NaN when not observed; the ball group starts "x,y". frame_id counts from 1, so
video_frame = frame_id - 1.

The parse takes a minute or two, so the arrays are cached next to the data (data/cache/, which is
gitignored with the rest of data/). The cache key includes the raw file's size and mtime, so a new
export is re-parsed automatically.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

HOME, AWAY = 0, 1


def team_of_each_slot(meta_path):
    """int8 array over track slots: 0 home, 1 away. First Team in the XML is home (EPTS)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse(raw_path, n_frames, n_slots):
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    n_records = 0
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            index = int(parts[0]) - 1
            if not 0 <= index < n_frames:
                continue
            n_records += 1
            values = parts[1].replace(";", ",").split(",")
            if len(values) == n_slots * 3:
                players[index] = np.array(values, dtype=np.float32).reshape(n_slots, 3)
            else:  # irregular line: fall back to group by group
                for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                    fields = group.split(",")
                    for k in range(min(3, len(fields))):
                        players[index, slot, k] = float(fields[k])
            fields = parts[2].split(",")
            if len(fields) >= 2 and fields[0][:1] not in ("N", "n", ""):
                ball[index, 0] = float(fields[0])
                ball[index, 1] = float(fields[1])
    return players, ball, n_records


def load(meta_path, raw_path, n_frames, cache_dir=None, verbose=True):
    """(players[T, S, 3] normalised x, y and speed m/s, ball[T, 2] normalised, team[S])."""
    team = team_of_each_slot(meta_path)
    stat = os.stat(raw_path)
    cache_dir = cache_dir or os.path.join(DATA_DIR, "cache")
    key = "{}_{}_{}_{}.npz".format(os.path.basename(raw_path), stat.st_size,
                                   int(stat.st_mtime), n_frames)
    cache = os.path.join(cache_dir, key)
    if os.path.isfile(cache):
        with np.load(cache) as z:
            return z["players"], z["ball"], team

    if verbose:
        print("parsing {} (first run only, cached afterwards) ...".format(
            os.path.basename(raw_path)))
    players, ball, n_records = _parse(raw_path, n_frames, len(team))
    if n_records != n_frames:
        raise ValueError("ATD has {} records for {} video frames -- not the reference export"
                         .format(n_records, n_frames))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, players=players, ball=ball)
    return players, ball, team
