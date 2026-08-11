"""
EPV on a second tracking source: SkillCorner
=========================================
`Metrica_EPV.py` -- Friends of Tracking Tutorial 4, EPV combined with pitch control to
score passes and search for the best available target -- has only ever been run
against Metrica's two sample matches. This example runs the same underlying model
(`SkillCorner_EPV.py` adapts the event *lookup* only, see its module docstring) against
a full SkillCorner match: every completed pass gets an EPV-added score, then one pass
gets the full Tutorial-4 treatment -- the entire EEPV surface, the pass actually played,
and the best target that was available instead.

Downloads (on demand, into `../data/SkillCorner/`, gitignored): the match's tracking
file (~90MB, cached after the first run) and its `dynamic_events.csv` pass log.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SkillCorner_IO as scio
import SkillCorner_EPV as scepv
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import Metrica_Viz as mviz

DATA_DIR = "../data/SkillCorner"
MATCH_ID = 2017461  # Melbourne Victory vs Auckland FC -- see matches.json for the other 9

##############################################################################
# Download and load -- the whole first half plus the substitution-free start of the
# second (EPV needs a broad sample of passes, not just a couple of demo frames),
# which takes on the order of 20-30 seconds.
# ----------------------------
# Not frame_windows=None (the whole match): a substitute's tracking columns are NaN
# for their entire time on the bench, and Metrica_Velocities.calc_player_velocities
# smooths each player's column as one continuous stretch per half -- which breaks
# across a stretch that long (see SkillCorner_IO's module docstring, point 4).
# calc_player_velocities also needs *some* second-half data to find the half
# boundary (its `Period.idxmax()`), so this uses all of period 1 plus period 2 up to
# (not including) the first substitution, found directly from match.json's per-player
# playing_time rather than parsing its display-time strings.

scio.download_match(MATCH_ID, DATA_DIR, dynamic_events=True)
match_json = scio.read_match_json(DATA_DIR, MATCH_ID)
period_1, period_2 = match_json["match_periods"][0], match_json["match_periods"][1]
first_sub_frame = min(
    (bp["start_frame"] for p in match_json["players"] for bp in p["playing_time"]["by_period"]
     if bp["name"] == "period_2" and bp["start_frame"] > period_2["start_frame"]),
    default=period_2["end_frame"],
)
frame_windows = [(period_1["start_frame"], period_1["end_frame"]), (period_2["start_frame"], first_sub_frame - 1)]
print(f"Period 2 used up to frame {first_sub_frame - 1} (first substitution at {first_sub_frame})")

tracking_home, tracking_away, match_meta = scio.read_match_data(DATA_DIR, MATCH_ID, frame_windows=frame_windows)
print(f"{match_meta['home_team']} (home) vs {match_meta['away_team']} (away): {len(tracking_home)} frames loaded")

tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

params = mpc.default_model_params()
EPV = mepv.load_EPV_grid("../data/Metrica/EPV_grid.csv")
GK_numbers = match_meta["GK_numbers"]
field_dimen = match_meta["field_dimen"]

##############################################################################
# EPV added, every completed pass
# ----------------------------

passes = scepv.load_pass_events(DATA_DIR, MATCH_ID, match_meta)
print(f"{len(passes)} completed passes in the match log")

rows = []
for _, pass_row in passes.iterrows():
    frame = int(pass_row["frame"])
    if frame not in tracking_home.index or frame not in tracking_away.index:
        continue  # outside the loaded windows (post-substitution, or late in either half) or a dropout gap too long to interpolate
    try:
        EEPV_added, EPV_difference = scepv.calculate_epv_added(
            pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta, field_dimen=field_dimen
        )
    except (AssertionError, ValueError):
        continue  # e.g. a goalkeeper not on the pitch that frame -- see check_offsides
    rows.append({
        "event_id": pass_row["event_id"], "side": pass_row["side"],
        "passer": pass_row["passer"], "receiver": pass_row["receiver"],
        "EEPV_added": EEPV_added, "EPV_difference": EPV_difference,
    })

df = pd.DataFrame(rows)
print(f"Scored {len(df)} passes ({len(passes) - len(df)} skipped -- dropout gaps or missing GK)")
print(df["EEPV_added"].describe())

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["EEPV_added"], bins=40)
ax.set_xlabel("Expected EPV added (pitch-control-weighted)")
ax.set_ylabel("Count")
ax.set_title(f"{match_meta['home_team']} vs {match_meta['away_team']} -- EPV added per completed pass")
fig.tight_layout()
fig.savefig("SkillCorner_EPV_distribution.png", dpi=150)

leaderboard = df.groupby(["side", "passer"]).agg(
    n_passes=("EEPV_added", "size"), mean_EEPV_added=("EEPV_added", "mean"), total_EEPV_added=("EEPV_added", "sum")
).reset_index()
leaderboard = leaderboard[leaderboard["n_passes"] >= 5].sort_values("total_EEPV_added", ascending=False)
print("\nTop passers by total EPV added:")
print(leaderboard.head(10).to_string())
leaderboard.to_csv("SkillCorner_EPV_leaderboard.csv", index=False)

##############################################################################
# One pass, in full: the actual EEPV surface and the best target available
# ----------------------------
# find_max_value_added_target scans the whole pitch-control surface -- expensive
# (~1600 grid cells, each its own pitch-control calculation) -- so this runs it once,
# for the single highest-EEPV_added pass found above, rather than for every pass.

best = df.sort_values("EEPV_added", ascending=False).iloc[0]
pass_row = passes[passes["event_id"] == best["event_id"]].iloc[0]
print(f"\nIllustrating pass: {pass_row['passer']} -> {pass_row['receiver']} "
      f"({pass_row['side']}, frame {pass_row['frame']}), EEPV_added={best['EEPV_added']:.4f}")

maxEPV_added, max_target_location, EEPV_grid = scepv.find_max_value_added_target(
    pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta, field_dimen=field_dimen
)
print(f"Best available target would have added {maxEPV_added:.4f} "
      f"(vs {best['EEPV_added']:.4f} for the pass actually played) at {max_target_location}")

fig, ax = mviz.plot_pitch(field_dimen=field_dimen, field_color="white")
ax.imshow(np.flipud(EEPV_grid), extent=(-field_dimen[0] / 2, field_dimen[0] / 2, -field_dimen[1] / 2, field_dimen[1] / 2),
          cmap="Reds", alpha=0.7, zorder=0)
start = pass_row["start_pos"]
target = pass_row["target_pos"]
ax.plot(*start, "o", color="black", markersize=8, zorder=4, label="Pass start")
ax.plot(*target, "o", color="blue", markersize=10, markeredgecolor="black", zorder=4, label="Actual target (reception)")
ax.plot(*max_target_location, "*", color="gold", markersize=20, markeredgecolor="black", zorder=4, label="Best available target")
ax.annotate("", xy=target, xytext=start, arrowprops=dict(arrowstyle="->", color="blue", lw=2))
ax.annotate("", xy=max_target_location, xytext=start, arrowprops=dict(arrowstyle="->", color="goldenrod", lw=2))
ax.legend(loc="upper left", fontsize=9)
ax.set_title(f"{pass_row['passer']} -> {pass_row['receiver']}: EEPV surface, actual vs. best target")
fig.savefig("SkillCorner_EPV_best_target.png", dpi=150, bbox_inches="tight")
plt.show()
