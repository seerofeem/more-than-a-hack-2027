"""Every tunable number in the pipeline, in one place, with where it comes from.

None of these was fitted to labels of the provided game. They are physical or football priors
(distances in metres, durations in seconds) or self-calibrating quantiles of label-free signals.
The public 5-minute ground truth was used only to check that the pipeline is aligned and that the
labels mean what we think -- never to pick between values below. See TECHNICAL.md.
"""

FPS = 25

# --- proximity / control -----------------------------------------------------------------------
# A player "controls" the ball when it is within playing distance. 1 m is a stride; beyond that
# evidence decays with a 1.5 m scale, so a pass in flight 10 m from everyone carries ~no weight.
CONTROL_RADIUS_M = 1.0
CONTROL_DECAY_M = 1.5
# How decisive the gap between the nearest home and nearest away player is. A 1.5 m advantage
# gives tanh(1) = 0.76 of full weight; a shoulder-to-shoulder duel gives close to nothing.
CONTROL_GAP_SCALE_M = 1.5
# Ball faster than this is struck / in flight: nobody is controlling it on that frame, so the
# proximity term is down-weighted (a player it flies past has not won it).
BALL_FLIGHT_SPEED_MS = 12.0
FLIGHT_WEIGHT = 0.25
# Per-frame score for the team closer to the ball (and minus that for the other).
K_CONTROL = 1.0

# --- dead ball ---------------------------------------------------------------------------------
# The ball is out when it is beyond a line by more than tracking noise.
OUT_MARGIN_M = 0.5
K_OUT = 2.0
# Ball lost by the tracker while at / beyond a line: it most likely left the pitch.
EDGE_MARGIN_M = 1.5
K_EDGE_MISSING = 0.4
# Visible, in-bounds, moving ball: play is live.
BALL_MOVING_MS = 1.0
K_ALIVE = 0.3
# A ball standing still for this long with a slow game around it is a free kick / goal kick setup.
STATIONARY_MS = 0.5
STATIONARY_S = 2.0
K_STATIONARY = 0.6
# Kick-off formation: ball on the centre spot, the two teams in opposite halves.
CENTRE_RADIUS_M = 2.0
FORMATION_SHARE = 0.9
K_FORMATION = 3.0
# Game tempo = mean speed of tracked players, smoothed over this window. Calibrated per half from
# its own quantiles (label-free): slow quantile -> dead-ish, fast quantile -> live.
TEMPO_WINDOW_S = 2.0
TEMPO_SLOW_Q = 0.20
TEMPO_FAST_Q = 0.60
K_TEMPO = 0.3

# --- Smart Tagging -----------------------------------------------------------------------------
# Team-tagged phases are evidence for a side, weakly: they are phases, not possessions. The sign of
# each code (does "Team" name the attacking or the defending side?) is decided from the data by
# agreement with the proximity evidence inside its intervals -- no labels involved.
K_TAG_TEAM = 0.2
# Set pieces: the restart happens inside the tagged window, so its first half leans dead.
K_TAG_SETPIECE = 0.2
# Before a tagged kick-off the ball is dead.
K_TAG_KICKOFF = 3.0
# Goal -> kick-off gap is dead, if the two tags are this close together.
GOAL_TO_KICKOFF_MAX_S = 180.0
K_TAG_GOAL = 2.0

# --- segmentation ------------------------------------------------------------------------------
# Viterbi switch costs, in the same units as the per-frame scores. H<->A needs ~0.4 s of clear
# control by the other side; entering or leaving D needs a little more.
SWITCH_HA = 10.0
SWITCH_D = 15.0
# Clean-up after Viterbi.
MIN_SPELL_S = 0.5          # shorter H/A spells are absorbed by what came before
MIN_DEAD_S = 1.0           # shorter D gaps inside one team's spell are closed
