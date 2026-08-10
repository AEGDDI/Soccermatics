# -*- coding: utf-8 -*-
"""
H1 on real data: space control at Euro 2024
=========================================
Applies the H1 "space control" freeze-frame metric -- a convex hull of defenders plus
an EPV-weighted attacking-value estimate, the same combination ArMat-Analytics'
`Contextual-Football-Scouting <https://github.com/ArMat-Analytics/Contextual-Football-Scouting>`_
project uses -- to real matches from StatsBomb's free open data
(`competition_id=55, season_id=282` is UEFA Euro 2024, which includes 360 freeze-frame
data). There's no tracking data available for Euro 2024, so this is a real analysis on
real matches, not a validation against ground truth.

Covers the full 51-match tournament. Event/frame data is cached to disk so re-runs
don't re-download from StatsBomb.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Sbopen

import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import StatsBomb_ArMatSpaceControl as sbarm

COMPETITION_ID = 55  # UEFA Euro
SEASON_ID = 282      # 2024
N_MATCHES = 51         # full tournament
CACHE_DIR = '../data/StatsBomb_Euro2024_cache'
FIELD_DIMEN = (105., 68.)

os.makedirs(CACHE_DIR, exist_ok=True)
parser = Sbopen()

##############################################################################
# Pick matches and fetch/cache their event + 360 data
# ----------------------------

df_match = parser.match(competition_id=COMPETITION_ID, season_id=SEASON_ID)
match_ids = df_match['match_id'].head(N_MATCHES).tolist()
print(f"Using {len(match_ids)} matches: {match_ids}")

params = mpc.default_model_params()
EPV = mepv.load_EPV_grid('../data/Metrica/EPV_grid.csv')

rows = []
minutes_played = {}  # (match_id, team_name, player_name) -> minutes

for match_id in match_ids:
    event_cache = f'{CACHE_DIR}/{match_id}_event.pkl'
    frame_cache = f'{CACHE_DIR}/{match_id}_frame.pkl'

    if os.path.exists(event_cache) and os.path.exists(frame_cache):
        df_event = pd.read_pickle(event_cache)
        df_frame = pd.read_pickle(frame_cache)
    else:
        df_event = parser.event(match_id)[0]
        df_frame, _ = parser.frame(match_id)
        df_event.to_pickle(event_cache)
        df_frame.to_pickle(frame_cache)

    print(f"Match {match_id}: {len(df_event)} events, {df_frame['id'].nunique()} with a 360 frame")

    attack_directions = sbarm.infer_attack_directions(df_event, field_dimen=FIELD_DIMEN)

    # minutes played, approximated from each player's own event timestamp span
    passes_and_touches = df_event.dropna(subset=['player_name'])
    for (team, player), grp in passes_and_touches.groupby(['team_name', 'player_name']):
        span_minutes = grp['minute'].max() - grp['minute'].min() + 1
        minutes_played[(match_id, team, player)] = span_minutes

    frame_ids = set(df_frame['id'])
    passes = df_event[(df_event['type_name'] == 'Pass') & df_event['id'].isin(frame_ids)]

    for _, event_row in passes.iterrows():
        frame_for_event = df_frame[df_frame['id'] == event_row['id']]
        result = sbarm.score_event(event_row, frame_for_event, EPV, params, attack_directions, field_dimen=FIELD_DIMEN)
        if result is None:
            continue
        h_area, controlled_value = result
        rows.append({
            'match_id': match_id, 'event_id': event_row['id'],
            'team_name': event_row['team_name'], 'player_name': event_row['player_name'],
            'period': event_row['period'],
            'hull_area': h_area, 'controlled_value': controlled_value,
        })

df = pd.DataFrame(rows)
df.to_csv('Euro2024_space_control.csv', index=False)
print(f"\nScored {len(df)} pass events across {len(match_ids)} matches")
print(df[['hull_area', 'controlled_value']].describe())

##############################################################################
# Distribution
# ----------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['hull_area'], bins=40)
axes[0].set_xlabel('Defensive hull area [m$^2$]')
axes[0].set_ylabel('Count')
axes[0].set_title('Space conceded by the defense, per pass')
axes[1].hist(df['controlled_value'], bins=40)
axes[1].set_xlabel('Attacking controlled value [Patt*EPV, summed]')
axes[1].set_title('Attacking value controlled, per pass')
fig.tight_layout()
fig.savefig('Euro2024_space_control_distribution.png', dpi=150)

##############################################################################
# Per-player leaderboard
# ----------------------------
# Only the passer's identity is known from a StatsBomb 360 freeze frame (every other
# player is an anonymous position dot) -- so this leaderboard reads as "how much space
# was around this player when they played their passes", not a per-player breakdown of
# who occupied that space.

player_summary = df.groupby(['team_name', 'player_name']).agg(
    n_passes=('hull_area', 'size'),
    mean_hull_area=('hull_area', 'mean'),
    mean_controlled_value=('controlled_value', 'mean'),
).reset_index()

player_summary['minutes'] = player_summary.apply(
    lambda r: sum(minutes_played.get((mid, r['team_name'], r['player_name']), 0.) for mid in match_ids), axis=1)
player_summary = player_summary[(player_summary['n_passes'] >= 10) & (player_summary['minutes'] > 0)]
player_summary['controlled_value_per90'] = 90. * player_summary['mean_controlled_value'] * player_summary['n_passes'] / player_summary['minutes']
player_summary = player_summary.sort_values('controlled_value_per90', ascending=False)

print("\nTop 10 by attacking value controlled per 90:")
print(player_summary.head(10).to_string())
player_summary.to_csv('Euro2024_space_control_leaderboard.csv', index=False)

fig2, ax2 = plt.subplots(figsize=(11, 6))
top10 = player_summary.head(10).iloc[::-1]
ax2.barh(top10['player_name'] + ' (' + top10['team_name'] + ')', top10['controlled_value_per90'])
ax2.set_xlabel('Attacking value controlled per 90')
ax2.set_title(f'Euro 2024 space-control leaderboard\n({len(match_ids)} matches, min. 10 scored passes)', fontsize=12)
fig2.tight_layout()
fig2.savefig('Euro2024_space_control_leaderboard.png', dpi=150)
