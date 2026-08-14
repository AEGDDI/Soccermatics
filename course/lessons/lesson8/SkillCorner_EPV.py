#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for computing EPV-added and best-target search -- Metrica_EPV.py's
`calculate_epv_added` / `find_max_value_added_target`, Friends of Tracking's Tutorial 4
-- against SkillCorner's pass log instead of Metrica's RawEventsData.csv.

Same model, same primitives (Metrica_PitchControl.calculate_pitch_control_at_target,
the Friends-of-Tracking EPV_grid.csv loaded via Metrica_EPV.load_EPV_grid) as the
Metrica tutorial. Only two things are adapted, both because SkillCorner's event data
doesn't share Metrica's schema:

1. **Event lookup.** `{match_id}_dynamic_events.csv` has no Start X/Y, End X/Y, Start
   Frame or Team columns. `load_pass_events` maps SkillCorner's own columns onto that
   shape: a completed pass is an `event_type == 'player_possession'` row with
   `pass_outcome == 'successful'`. Its start position is the passer's own position at
   `frame_start` (`x_start`/`y_start` -- same convention as Metrica's "Start X/Y", the
   actor's position, not a separate ball position); its *target* is
   `player_targeted_x_reception`/`_y_reception` -- where the receiving player actually
   is at reception -- not `x_end`/`y_end`, which is the *passer's own* position at the
   end of their possession (i.e. release point), not where the ball arrives.

2. **Attack direction.** `calculate_epv_added`/`find_max_value_added_target` call
   `Metrica_IO.find_playing_direction(tracking_home, 'Home')` internally, which infers
   direction from the goalkeeper's position in the tracking DataFrame's *first row* --
   a heuristic that assumes that row is kickoff. That assumption breaks for any
   SkillCorner tracking read that doesn't start exactly at a period boundary (this
   module's own example doesn't, for speed). SkillCorner's `{match_id}_match.json`
   already states attack direction directly (`home_team_side`, surfaced as
   `match_meta['periods'][i]['home_direction']` by `SkillCorner_IO.match_summary`), so
   `attack_direction_for_period` reads it directly instead of re-deriving it.
"""

import os

import numpy as np
import pandas as pd

import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import SkillCorner_IO as scio


def load_pass_events(data_dir, match_id, match_meta):
    """ load_pass_events(data_dir, match_id, match_meta)

    Reads `{match_id}_dynamic_events.csv` (must already be downloaded alongside the
    tracking file, see SkillCorner_IO.download_match -- note dynamic_events.csv is
    NOT LFS-hosted, unlike the tracking file, so a plain raw.githubusercontent.com
    fetch is fine for it) and returns completed passes as one row per pass:
    event_id, frame_start, period, side ('Home'/'Away'), passer/receiver names and ids,
    start_pos/target_pos as (x,y) tuples, and SkillCorner's own
    player_targeted_xthreat/player_targeted_xpass_completion for that same pass (NOT
    the bare xthreat/xpass_completion columns -- those are always empty on pass rows,
    populated for other event types instead; verified directly against the data).

    passer_id/receiver_id (SkillCorner's own player ids, not just names) are carried
    through so a leaderboard built across multiple matches can join on a stable key --
    two different clubs could plausibly share a player's short name, ids can't collide.
    """
    path = os.path.join(data_dir, f"{match_id}_dynamic_events.csv")
    df = pd.read_csv(path, low_memory=False)

    passes = df[(df["event_type"] == "player_possession") & (df["pass_outcome"] == "successful")].copy()
    passes = passes.dropna(subset=["player_targeted_x_reception", "player_targeted_y_reception"])

    home_team = match_meta["home_team"]
    passes["side"] = np.where(passes["team_shortname"] == home_team, "Home", "Away")
    passes["start_pos"] = list(zip(passes["x_start"], passes["y_start"]))
    passes["target_pos"] = list(zip(passes["player_targeted_x_reception"], passes["player_targeted_y_reception"]))

    return passes.rename(columns={
        "frame_start": "frame", "player_name": "passer", "player_targeted_name": "receiver",
        "player_id": "passer_id", "player_targeted_id": "receiver_id",
    })[["event_id", "frame", "period", "side", "passer", "passer_id", "receiver", "receiver_id",
        "start_pos", "target_pos", "player_targeted_xthreat", "player_targeted_xpass_completion"]].reset_index(drop=True)


def attack_direction_for_period(match_meta, period, side):
    """ Returns +1 (left->right) or -1 (right->left) for `side` ('Home'/'Away') in the
    given `period`, read directly from match.json's own `home_team_side` (via
    SkillCorner_IO.match_summary) rather than inferred from tracking data -- see
    module docstring, point 2. """
    home_direction = 1 if match_meta["periods"][period - 1]["home_direction"] == "left_to_right" else -1
    return home_direction if side == "Home" else -home_direction


def _initialise_attacking_defending(pass_row, tracking_home, tracking_away, GK_numbers, params):
    frame = int(pass_row["frame"])
    if pass_row["side"] == "Home":
        attacking = mpc.initialise_players(tracking_home.loc[frame], "Home", params, GK_numbers[0])
        defending = mpc.initialise_players(tracking_away.loc[frame], "Away", params, GK_numbers[1])
    else:
        attacking = mpc.initialise_players(tracking_away.loc[frame], "Away", params, GK_numbers[1])
        defending = mpc.initialise_players(tracking_home.loc[frame], "Home", params, GK_numbers[0])
    return attacking, defending


def calculate_epv_added(pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta,
                         field_dimen=(105., 68.)):
    """ calculate_epv_added(pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta, field_dimen=(105.,68.))

    Exactly Metrica_EPV.calculate_epv_added's computation -- EPV-weighted pitch control
    at the pass's start position vs. its actual target (reception) position, both
    evaluated from the single tracking frame at the moment of release -- against one
    row of `load_pass_events`'s output instead of a Metrica events DataFrame row.

    Returns
    -----------
        EEPV_added: expected EPV added by this pass (pitch-control-weighted)
        EPV_difference: raw EPV difference between target and start (ignoring pitch control)
    """
    pass_start_pos = np.array(pass_row["start_pos"])
    pass_target_pos = np.array(pass_row["target_pos"])
    attack_direction = attack_direction_for_period(match_meta, pass_row["period"], pass_row["side"])

    attacking_players, defending_players = _initialise_attacking_defending(pass_row, tracking_home, tracking_away, GK_numbers, params)
    attacking_players = mpc.check_offsides(attacking_players, defending_players, pass_start_pos, GK_numbers)

    Patt_start, _ = mpc.calculate_pitch_control_at_target(pass_start_pos, attacking_players, defending_players, pass_start_pos, params)
    Patt_target, _ = mpc.calculate_pitch_control_at_target(pass_target_pos, attacking_players, defending_players, pass_start_pos, params)

    EPV_start = mepv.get_EPV_at_location(pass_start_pos, EPV, attack_direction=attack_direction, field_dimen=field_dimen)
    EPV_target = mepv.get_EPV_at_location(pass_target_pos, EPV, attack_direction=attack_direction, field_dimen=field_dimen)

    EEPV_added = Patt_target * EPV_target - Patt_start * EPV_start
    EPV_difference = EPV_target - EPV_start
    return EEPV_added, EPV_difference


def calculate_epv_components(pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta,
                              field_dimen=(105., 68.)):
    """ calculate_epv_components(pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta, field_dimen=(105.,68.))

    Same computation as `calculate_epv_added`, but returns the intermediate values too
    -- needed to compare against SkillCorner's own per-pass metrics, which aren't
    deltas: `player_targeted_xpass_completion` is a probability-of-success estimate for
    the target alone (the counterpart of `Patt_target` here, not `EEPV_added`), and
    `player_targeted_xthreat` is a danger/value score for the target location alone
    (the counterpart of `EPV_target`). `calculate_epv_added` is left as-is (nothing
    depending on its 2-value return breaks); this is an additive sibling.

    Returns
    -----------
        dict with Patt_start, EPV_start, Patt_target, EPV_target, EEPV_added, EPV_difference
    """
    pass_start_pos = np.array(pass_row["start_pos"])
    pass_target_pos = np.array(pass_row["target_pos"])
    attack_direction = attack_direction_for_period(match_meta, pass_row["period"], pass_row["side"])

    attacking_players, defending_players = _initialise_attacking_defending(pass_row, tracking_home, tracking_away, GK_numbers, params)
    attacking_players = mpc.check_offsides(attacking_players, defending_players, pass_start_pos, GK_numbers)

    Patt_start, _ = mpc.calculate_pitch_control_at_target(pass_start_pos, attacking_players, defending_players, pass_start_pos, params)
    Patt_target, _ = mpc.calculate_pitch_control_at_target(pass_target_pos, attacking_players, defending_players, pass_start_pos, params)

    EPV_start = mepv.get_EPV_at_location(pass_start_pos, EPV, attack_direction=attack_direction, field_dimen=field_dimen)
    EPV_target = mepv.get_EPV_at_location(pass_target_pos, EPV, attack_direction=attack_direction, field_dimen=field_dimen)

    return {
        "Patt_start": Patt_start, "EPV_start": EPV_start,
        "Patt_target": Patt_target, "EPV_target": EPV_target,
        "EEPV_added": Patt_target * EPV_target - Patt_start * EPV_start,
        "EPV_difference": EPV_target - EPV_start,
    }


def score_match(data_dir, match_id, params=None, EPV=None):
    """ score_match(data_dir, match_id, params=None, EPV=None)

    Reusable per-match pipeline: downloads (cached) the match's tracking + event data,
    reads the whole match with velocities (`SkillCorner_IO.read_full_match_with_
    velocities`, substitution-safe), scores every completed pass with
    `calculate_epv_components`, and returns one row per successfully-scored pass with
    our components alongside SkillCorner's own `player_targeted_xthreat`/
    `player_targeted_xpass_completion` for the same pass (already joined in by
    `load_pass_events`) -- ready to concatenate across multiple matches.

    `params`/`EPV` can be passed in to reuse across many `score_match` calls rather
    than reloading `EPV_grid.csv` and rebuilding default params for every match;
    defaults are created if omitted.

    Also returns the loaded tracking_home/tracking_away/match_meta, so a caller that
    wants to do something extra with one specific pass afterwards (e.g. illustrate it
    with `find_max_value_added_target`, as `plot_SkillCornerEPV.py` does) doesn't have
    to pay the cost of reading the whole match a second time.

    Returns
    -----------
        results: DataFrame -- match_id, event_id, side, passer, passer_id, receiver,
            receiver_id, pass_distance, Patt_target, EPV_target, EEPV_added,
            EPV_difference, player_targeted_xthreat, player_targeted_xpass_completion
        tracking_home, tracking_away, match_meta: as returned by
            SkillCorner_IO.read_full_match_with_velocities
    """
    if params is None:
        params = mpc.default_model_params()
    if EPV is None:
        EPV = mepv.load_EPV_grid("../data/Metrica/EPV_grid.csv")

    scio.download_match(match_id, data_dir, dynamic_events=True)
    tracking_home, tracking_away, match_meta = scio.read_full_match_with_velocities(data_dir, match_id)
    GK_numbers = match_meta["GK_numbers"]
    field_dimen = match_meta["field_dimen"]

    passes = load_pass_events(data_dir, match_id, match_meta)

    rows = []
    for _, pass_row in passes.iterrows():
        frame = int(pass_row["frame"])
        if frame not in tracking_home.index or frame not in tracking_away.index:
            continue
        try:
            components = calculate_epv_components(
                pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta, field_dimen=field_dimen
            )
        except (AssertionError, ValueError):
            continue
        pass_distance = float(np.linalg.norm(np.array(pass_row["target_pos"]) - np.array(pass_row["start_pos"])))
        rows.append({
            "match_id": match_id, "event_id": pass_row["event_id"], "side": pass_row["side"],
            "passer": pass_row["passer"], "passer_id": pass_row["passer_id"],
            "receiver": pass_row["receiver"], "receiver_id": pass_row["receiver_id"],
            "pass_distance": pass_distance,
            "Patt_target": components["Patt_target"], "EPV_target": components["EPV_target"],
            "EEPV_added": components["EEPV_added"], "EPV_difference": components["EPV_difference"],
            "player_targeted_xthreat": pass_row["player_targeted_xthreat"],
            "player_targeted_xpass_completion": pass_row["player_targeted_xpass_completion"],
        })
    return pd.DataFrame(rows), tracking_home, tracking_away, match_meta


def find_max_value_added_target(pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta,
                                 field_dimen=(105., 68.), n_grid_cells_x=50):
    """ find_max_value_added_target(pass_row, tracking_home, tracking_away, GK_numbers, EPV, params, match_meta, field_dimen=(105.,68.), n_grid_cells_x=50)

    Exactly Metrica_EPV.find_max_value_added_target's computation -- builds the full
    pitch-control surface at the pass frame (same grid loop as
    Metrica_PitchControl.generate_pitch_control_for_event, reused directly here since
    that function itself needs a Metrica-schema `events` DataFrame), multiplies it by
    the (direction-flipped) EPV grid, and returns wherever that product is highest.

    n_grid_cells_x=50 matches the shipped EPV_grid.csv's own shape (32, 50) --
    changing it without a differently-shaped EPV grid would silently misalign the two
    surfaces in the final elementwise multiply.

    Returns
    -----------
        maxEPV_added: the highest EEPV achievable from this frame, minus EEPV at the actual pass start
        max_target_location: (x,y) of that location
        EEPV_grid: the full grid, for plotting
    """
    pass_start_pos = np.array(pass_row["start_pos"])
    attack_direction = attack_direction_for_period(match_meta, pass_row["period"], pass_row["side"])

    attacking_players, defending_players = _initialise_attacking_defending(pass_row, tracking_home, tracking_away, GK_numbers, params)
    attacking_players = mpc.check_offsides(attacking_players, defending_players, pass_start_pos, GK_numbers)

    Patt_start, _ = mpc.calculate_pitch_control_at_target(pass_start_pos, attacking_players, defending_players, pass_start_pos, params)
    EPV_start = mepv.get_EPV_at_location(pass_start_pos, EPV, attack_direction=attack_direction, field_dimen=field_dimen)
    EEPV_start = Patt_start * EPV_start

    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    xgrid = np.linspace(-field_dimen[0] / 2., field_dimen[0] / 2., n_grid_cells_x)
    ygrid = np.linspace(-field_dimen[1] / 2., field_dimen[1] / 2., n_grid_cells_y)
    PPCFa = np.zeros((len(ygrid), len(xgrid)))
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            PPCFa[i, j], _ = mpc.calculate_pitch_control_at_target(target_position, attacking_players, defending_players, pass_start_pos, params)

    EEPV_grid = (np.fliplr(EPV) if attack_direction == -1 else EPV) * PPCFa
    imax = np.unravel_index(EEPV_grid.argmax(), EEPV_grid.shape)
    maxEPV_added = EEPV_grid[imax] - EEPV_start
    max_target_location = (xgrid[imax[1]], ygrid[imax[0]])
    return maxEPV_added, max_target_location, EEPV_grid
