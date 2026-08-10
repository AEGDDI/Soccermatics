# -*- coding: utf-8 -*-
"""
Pitch control and risky passes
=========================================
In this tutorial we use Laurie Shaw's pitch control model to flag "risky" passes: completed
passes that succeeded even though, at the moment of release, the model gave the defending
team a better-than-even chance of winning the ball at the pass's end location. Code adapted
from Laurie Shaw's (@EightyFivePoint) `Tutorial3_PitchControl.py
<https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Tutorial3_PitchControl.py>`_
provided for `Friends of Tracking <https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w>`_.
"""

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Opening data
# ----------------------------
# We use Metrica sample match 2, the same match used elsewhere in this lesson.

DATADIR = '../data/Metrica'
game_id = 2  # let's look at sample match 2

# read in the event data
events = mio.read_event_data(DATADIR, game_id)

# read in tracking data
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

# Convert positions from metrica units to meters
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

# reverse direction of play in the second half so that home team is always attacking from right->left
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

# Calculate player velocities
# NOTE: the default Savitzky-Golay filter can raise "array must not contain infs or NaNs"
# with newer scipy versions; fall back to a moving-average filter in that case.
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True, filter_='moving average')
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True, filter_='moving average')

##############################################################################
# Pitch control for the passes leading up to a goal
# ----------------------------
# As a sanity check, look at the pitch control surface for the three events leading up to
# the match's second goal, the same example used in Laurie Shaw's tutorial.

shots = events[events['Type'] == 'SHOT']
goals = shots[shots['Subtype'].str.contains('-GOAL')].copy()
print(goals)

mviz.plot_events(events.loc[820:823], color='k', indicators=['Marker', 'Arrow'], annotate=True)

params = mpc.default_model_params()
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]

for event_id in [820, 821, 822]:
    PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
        event_id, events, tracking_home, tracking_away, params, GK_numbers,
        field_dimen=(106., 68.,), n_grid_cells_x=50)
    mviz.plot_pitchcontrol_for_event(event_id, events, tracking_home, tracking_away, PPCF, annotate=True)

##############################################################################
# Pass success probability for every completed home-team pass
# ----------------------------
# For each pass, evaluate the pitch control model at the pass's end location using the
# player positions and velocities at the moment the pass was struck. This gives the
# probability that the attacking team would have won the ball there, ``Patt``.

home_passes = events[(events['Type'].isin(['PASS'])) & (events['Team'] == 'Home')]

pass_success_probability = []

for i, row in home_passes.iterrows():
    pass_start_pos = np.array([row['Start X'], row['Start Y']])
    pass_target_pos = np.array([row['End X'], row['End Y']])
    pass_frame = row['Start Frame']

    attacking_players = mpc.initialise_players(tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0])
    defending_players = mpc.initialise_players(tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1])
    Patt, Pdef = mpc.calculate_pitch_control_at_target(
        pass_target_pos, attacking_players, defending_players, pass_start_pos, params)

    pass_success_probability.append((i, Patt))

fig, ax = plt.subplots()
ax.hist([p[1] for p in pass_success_probability], np.arange(0, 1.1, 0.1))
ax.set_xlabel('Pass success probability')
ax.set_ylabel('Frequency')

##############################################################################
# Identifying risky passes
# ----------------------------
# Sort passes by pitch control probability. Passes below 0.5 succeeded despite the model
# favouring the defending team to win the ball — these are the "risky" passes.

pass_success_probability = sorted(pass_success_probability, key=lambda x: x[1])

risky_passes = events.loc[[p[0] for p in pass_success_probability if p[1] < 0.5]]

mviz.plot_events(risky_passes, color='k', indicators=['Marker', 'Arrow'], annotate=True)

print("Event following a risky (completed) pass")
for p in pass_success_probability[:20]:
    outcome = events.loc[p[0] + 1].Type
    print(p[1], outcome)
