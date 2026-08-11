"""
Pitch control on a second tracking source: SkillCorner
=========================================
`Metrica_PitchControl.py` and `Metrica_Velocities.py` have only ever been run against
the two Metrica sample matches this course ships. This example runs the same,
unmodified Spearman pitch-control model against a different tracking provider --
`SkillCorner's open broadcast-tracking data <https://github.com/SkillCorner/opendata>`_
(10 matches, 2024/25 Australian A-League) -- as a check that the pipeline generalises
rather than being implicitly tuned to Metrica's own data.

This is real frame-by-frame tracking (so real velocities, computed the normal way),
just from broadcast video rather than a fixed stadium camera rig, which brings its own
data-quality quirks -- see `SkillCorner_IO.py`'s module docstring for the three that
mattered here (coordinate convention, substitutes, and frame dropouts).

The match's tracking file (~90MB) is downloaded on demand into
`../data/SkillCorner/` (gitignored) the first time this runs.
"""

import numpy as np
import matplotlib.pyplot as plt

import SkillCorner_IO as scio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_Viz as mviz

DATA_DIR = "../data/SkillCorner"
MATCH_ID = 2017461  # Melbourne Victory vs Auckland FC -- see matches.json for the other 9

##############################################################################
# Download and load
# ----------------------------
# `frame_windows` keeps this example fast (a couple of seconds of tracking data from
# each half, so both a first-half and second-half frame are available -- calc_player_
# velocities needs to see both halves to split its smoothing at the correct half-time
# boundary). Pass `frame_windows=None` to `read_match_data` to load the full match.

scio.download_match(MATCH_ID, DATA_DIR)
match_json = scio.read_match_json(DATA_DIR, MATCH_ID)
periods = match_json["match_periods"]
frame_windows = [
    (periods[0]["start_frame"] + 200, periods[0]["start_frame"] + 700),
    (periods[1]["start_frame"] + 200, periods[1]["start_frame"] + 700),
]

tracking_home, tracking_away, match_meta = scio.read_match_data(DATA_DIR, MATCH_ID, frame_windows=frame_windows)
print(f"{match_meta['home_team']} (home) vs {match_meta['away_team']} (away)")
print(f"Loaded {len(tracking_home)} frames, field_dimen={match_meta['field_dimen']}, "
      f"GK_numbers={match_meta['GK_numbers']}")

##############################################################################
# Velocities
# ----------------------------
# Exactly Metrica_Velocities.calc_player_velocities, unmodified.

tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

##############################################################################
# Pitch control at one frame
# ----------------------------
# SkillCorner's dynamic_events.csv doesn't share Metrica's RawEventsData.csv schema (see
# SkillCorner_IO's module docstring), so this bypasses generate_pitch_control_for_event's
# event-lookup and instead builds the attacking/defending player lists directly for a
# chosen frame -- using the tracking file's own per-frame `possession` field to decide
# which side is attacking. calculate_pitch_control_at_target itself is unmodified.

params = mpc.default_model_params()
GK_home, GK_away = match_meta["GK_numbers"]

first_half = tracking_home[tracking_home["Period"] == 1]
candidates = first_half[first_half["possession_team"].notna()]
test_frame = candidates.index[len(candidates) // 2]
attacking_side = candidates.loc[test_frame, "possession_team"]
print(f"\nFrame {test_frame}: {attacking_side} in possession")

if attacking_side == "Home":
    attacking_players = mpc.initialise_players(tracking_home.loc[test_frame], "Home", params, GK_home)
    defending_players = mpc.initialise_players(tracking_away.loc[test_frame], "Away", params, GK_away)
else:
    attacking_players = mpc.initialise_players(tracking_away.loc[test_frame], "Away", params, GK_away)
    defending_players = mpc.initialise_players(tracking_home.loc[test_frame], "Home", params, GK_home)

ball_start_pos = np.array([tracking_home.loc[test_frame, "ball_x"], tracking_home.loc[test_frame, "ball_y"]])
print(f"{len(attacking_players)} attacking players, {len(defending_players)} defending players, "
      f"ball at ({ball_start_pos[0]:.1f}, {ball_start_pos[1]:.1f})")

field_dimen = match_meta["field_dimen"]
n_grid_cells_x = 50
n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
xgrid = np.linspace(-field_dimen[0] / 2.0, field_dimen[0] / 2.0, n_grid_cells_x)
ygrid = np.linspace(-field_dimen[1] / 2.0, field_dimen[1] / 2.0, n_grid_cells_y)
PPCFa = np.zeros((len(ygrid), len(xgrid)))
for i in range(len(ygrid)):
    for j in range(len(xgrid)):
        target_position = np.array([xgrid[j], ygrid[i]])
        PPCFa[i, j], _ = mpc.calculate_pitch_control_at_target(
            target_position, attacking_players, defending_players, ball_start_pos, params
        )
checksum = np.nansum(PPCFa) + np.nansum(1 - PPCFa)
print(f"Pitch control surface computed ({PPCFa.size} cells, sum check {checksum:.1f} -- should equal {PPCFa.size})")

##############################################################################
# Plot
# ----------------------------
# Reuses Metrica_Viz.plot_pitch for the pitch itself; the control surface and player
# markers are drawn directly, since plot_pitchcontrol_for_event expects Metrica's event
# schema (see module notes).

fig, ax = mviz.plot_pitch(field_dimen=field_dimen, field_color="white")
ax.imshow(np.flipud(PPCFa), extent=(-field_dimen[0] / 2, field_dimen[0] / 2, -field_dimen[1] / 2, field_dimen[1] / 2),
          cmap="bwr", vmin=0, vmax=1, alpha=0.6, zorder=0)
for p in attacking_players:
    ax.plot(*p.position, "o", color="red", markersize=10, markeredgecolor="black", zorder=3)
for p in defending_players:
    ax.plot(*p.position, "o", color="blue", markersize=10, markeredgecolor="black", zorder=3)
ax.plot(*ball_start_pos, "o", color="black", markersize=8, zorder=4)
ax.set_title(f"{match_meta['home_team']} vs {match_meta['away_team']} -- "
             f"SkillCorner tracking, frame {test_frame}, attacking={attacking_side}")
fig.savefig("SkillCorner_pitch_control.png", dpi=150, bbox_inches="tight")
plt.show()
