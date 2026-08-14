# -*- coding: utf-8 -*-
"""
All 10 matches, and checking our numbers against SkillCorner's own
=========================================
Two things in one script: (1) run the pitch-control/EPV pipeline
(`SkillCorner_EPV.score_match`) across every match in SkillCorner's open dataset, not
just the one semi-final `plot_SkillCornerEPV.py` covers, and (2) compare our
independently-computed numbers against SkillCorner's own precomputed per-pass metrics
for the exact same passes -- a genuine validation check, since the two are built by
completely different methods (our physics-based pitch-control model vs. their trained
classifier).

The comparison is against the right quantities, not just anything with a similar name:
`player_targeted_xpass_completion` (SkillCorner's own pass-success probability) is the
counterpart of our `Patt_target` (pitch control's probability at the reception point) --
both are "how likely was this pass to succeed." `player_targeted_xthreat` (a danger
score for the destination alone, not a delta) is the counterpart of our `EPV_target`,
not our `EEPV_added` (which is a start-to-target delta and can be negative -- comparing
it against xthreat, which never is, would be comparing different things). Verified
directly against the data before writing this script (see the plan/commit message).

Downloads ~9 more ~90MB tracking files beyond the one `plot_SkillCornerEPV.py` already
caches (into `../data/SkillCorner/`, gitignored) -- this is a long first run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import SkillCorner_IO as scio
import SkillCorner_EPV as scepv
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv

DATA_DIR = "../data/SkillCorner"

##############################################################################
# Score every match
# ----------------------------

matches = scio.download_matches_json(DATA_DIR)
match_ids = [m["id"] for m in matches]
print(f"{len(match_ids)} matches: {match_ids}")

params = mpc.default_model_params()
EPV = mepv.load_EPV_grid("../data/Metrica/EPV_grid.csv")

match_dfs = []
minutes_played = {}  # player_id -> {'name': str, 'minutes': float}

for match_id in match_ids:
    df, tracking_home, tracking_away, match_meta = scepv.score_match(DATA_DIR, match_id, params=params, EPV=EPV)
    match_dfs.append(df)
    print(f"Match {match_id}: {match_meta['home_team']} vs {match_meta['away_team']} -- {len(df)} passes scored")

    match_json = scio.read_match_json(DATA_DIR, match_id)
    for p in match_json["players"]:
        if p["playing_time"]["total"] is None:
            continue  # an unused substitute -- never took the pitch, no minutes to add
        entry = minutes_played.setdefault(p["id"], {"name": p["short_name"], "minutes": 0.0})
        entry["minutes"] += p["playing_time"]["total"]["minutes_played"]

all_passes = pd.concat(match_dfs, ignore_index=True)
all_passes.to_csv("SkillCorner_AllMatches_passes.csv", index=False)
print(f"\nTotal: {len(all_passes)} passes scored across {len(match_ids)} matches")

##############################################################################
# Cross-match leaderboard
# ----------------------------

player_totals = all_passes.groupby(["passer_id", "passer"]).agg(
    n_passes=("EEPV_added", "size"), total_EEPV_added=("EEPV_added", "sum")
).reset_index()
player_totals["minutes"] = player_totals["passer_id"].map(lambda pid: minutes_played.get(pid, {}).get("minutes", np.nan))
player_totals = player_totals[player_totals["n_passes"] >= 20]
player_totals["EEPV_added_per90"] = 90.0 * player_totals["total_EEPV_added"] / player_totals["minutes"]
player_totals = player_totals.sort_values("EEPV_added_per90", ascending=False)
print("\nTop 10 by EPV added per 90 (min. 20 passes across the dataset):")
print(player_totals.head(10).to_string())
player_totals.to_csv("SkillCorner_AllMatches_leaderboard.csv", index=False)

##############################################################################
# Comparing our numbers against SkillCorner's own
# ----------------------------

compare = all_passes.dropna(subset=["player_targeted_xthreat", "player_targeted_xpass_completion"])
n_invalid = (compare["Patt_target"] > 1.0).sum()
if n_invalid:
    print(f"Excluding {n_invalid} pass(es) with Patt_target > 1.0 (a numerically pathological "
          f"integration case in calculate_pitch_control_at_target -- rare, {n_invalid}/{len(compare)} here)")
    compare = compare[compare["Patt_target"] <= 1.0]
print(f"\n{len(compare)} of {len(all_passes)} passes have both our numbers and SkillCorner's own")

# Both Pearson (sensitive to the heavy right-skew both distributions have -- most
# passes are safe/low-value, a few are very risky/dangerous) and Spearman (rank-based,
# not fooled by that skew) are worth reporting -- if they disagree a lot, that itself
# says something about whether any agreement is a real per-pass relationship or just
# both metrics sharing the same skewed shape.
r_pass, _ = stats.pearsonr(compare["Patt_target"], compare["player_targeted_xpass_completion"])
rho_pass, _ = stats.spearmanr(compare["Patt_target"], compare["player_targeted_xpass_completion"])
r_threat, _ = stats.pearsonr(compare["EPV_target"], compare["player_targeted_xthreat"])
rho_threat, _ = stats.spearmanr(compare["EPV_target"], compare["player_targeted_xthreat"])
print(f"Patt_target vs. player_targeted_xpass_completion: Pearson r = {r_pass:.3f}, Spearman rho = {rho_pass:.3f}")
print(f"EPV_target vs. player_targeted_xthreat: Pearson r = {r_threat:.3f}, Spearman rho = {rho_threat:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(compare["player_targeted_xpass_completion"], compare["Patt_target"], alpha=0.15, s=10)
axes[0].set_xlabel("SkillCorner xpass_completion")
axes[0].set_ylabel("Our Patt_target (pitch control)")
axes[0].set_title(f"Pass success probability (Pearson r={r_pass:.2f}, Spearman rho={rho_pass:.2f})", fontsize=10)

axes[1].scatter(compare["player_targeted_xthreat"], compare["EPV_target"], alpha=0.15, s=10)
axes[1].set_xlabel("SkillCorner xthreat")
axes[1].set_ylabel("Our EPV_target")
axes[1].set_title(f"Value of the destination (Pearson r={r_threat:.2f}, Spearman rho={rho_threat:.2f})", fontsize=10)
fig.tight_layout()
fig.savefig("SkillCorner_AllMatches_comparison.png", dpi=150)

# Correlation pooled across all 10 matches can look consistent even if there's no real
# relationship, if e.g. both metrics just happen to share the same skew -- checking
# per-match tells a different, more honest story if the sign/strength isn't stable.
print("\nPer-match Spearman rho (checking whether any pooled correlation is a stable, real relationship):")
for mid, g in compare.groupby("match_id"):
    rho_m_pass, _ = stats.spearmanr(g["Patt_target"], g["player_targeted_xpass_completion"])
    rho_m_threat, _ = stats.spearmanr(g["EPV_target"], g["player_targeted_xthreat"])
    print(f"  match {mid} (n={len(g)}): pass rho={rho_m_pass:.3f}, threat rho={rho_m_threat:.3f}")

##############################################################################
# Where do the two methods disagree most?
# ----------------------------
# Bucket by pass distance -- a short, simple pass should be easy for both methods to
# agree on; a long, risky pass is where a physics model and a trained classifier are
# more likely to see the situation differently.

compare = compare.copy()
compare["abs_pass_diff"] = np.abs(compare["Patt_target"] - compare["player_targeted_xpass_completion"])
compare["distance_bucket"] = pd.cut(compare["pass_distance"], bins=[0, 10, 20, 30, 100],
                                     labels=["0-10m", "10-20m", "20-30m", "30m+"])
bucket_summary = compare.groupby("distance_bucket", observed=True)["abs_pass_diff"].agg(["mean", "count"])
print("\nDisagreement on pass success probability, by pass distance:")
print(bucket_summary)

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.bar(bucket_summary.index.astype(str), bucket_summary["mean"])
ax2.set_xlabel("Pass distance")
ax2.set_ylabel("Mean |Patt_target - xpass_completion|")
ax2.set_title("Where our model and SkillCorner's disagree most on pass success")
fig2.tight_layout()
fig2.savefig("SkillCorner_AllMatches_disagreement_by_distance.png", dpi=150)
