"""Ball-possession labelling for More than a Hack 2027, Phase 1.

Pipeline (see TECHNICAL.md for the reasoning):

    atd.py       stream the raw ATD into numpy arrays (cached under data/cache/)
    tagging.py   read the Smart Tagging XML into frame intervals
    features.py  per-frame physical evidence: who is near the ball, is the ball out, tempo, ...
    evidence.py  turn features + tags into per-frame scores for H / A / D
    segment.py   Viterbi segmentation with switch costs, then spell clean-up
    run.py       the command that regenerates the submission

Everything is deterministic: no random numbers, no learned weights, no labels from the game.
"""
