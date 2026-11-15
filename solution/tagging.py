"""Smart Tagging intervals in video frames. SportsCode XML, standard library only.

<start>/<end> are seconds from the first tactical video frame (verified by the organisers, see
starter/load_smart_tagging.py), so round(seconds * fps) is a video frame.
"""

import xml.etree.ElementTree as ET

HOME_NAMES = ("barcelona",)
AWAY_NAMES = ("atletic club", "athletic club")


def side_of(team_text):
    """0 home, 1 away, None for N/A or anything unrecognised."""
    value = (team_text or "").strip().lower()
    if value in HOME_NAMES:
        return 0
    if value in AWAY_NAMES:
        return 1
    return None


def load(path, fps):
    """[{code, type, side, start, stop}], code/type lower-cased, stop exclusive, in file order."""
    out = []
    for instance in ET.parse(path).iter("instance"):
        start = int(round(float(instance.findtext("start")) * fps))
        stop = int(round(float(instance.findtext("end")) * fps))
        # Tag groups are title case in the export ("Team"); match case-insensitively.
        tags = {(label.findtext("group") or "").strip().lower(): label.findtext("text")
                for label in instance.findall("label")}
        out.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "type": (tags.get("type") or "").strip().lower(),
            "side": side_of(tags.get("team")),
            "start": start,
            "stop": max(stop, start + 1),
        })
    return out
