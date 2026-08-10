#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for applying the H1 "space control" freeze-frame metric (convex hull of
defenders + EPV-weighted attacking value), the same combination ArMat-Analytics'
Contextual-Football-Scouting project uses, to real StatsBomb 360 data.

StatsBomb 360 data has no player identity for anyone except the event's own actor, and
no velocity at all -- every player in a freeze frame is just an (x,y) dot. This module
builds a duck-typed player object with the same interface
Metrica_PitchControl.calculate_pitch_control_at_target expects, directly from a
StatsBomb freeze frame, instead of trying to reuse Metrica_PitchControl.player (whose
constructor is hard-coded to read Metrica's tracking-row column layout).
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

import Metrica_PitchControl as mpc
import Metrica_EPV as mepv


def hull_area(points):
    """ hull_area(points)

    Convex hull area (m^2) of a set of (x,y) points -- the ArMat H1 primitive. Returns
    NaN if fewer than 3 points are provided or the hull computation fails (e.g.
    collinear points).
    """
    points = np.asarray(points)
    if points.shape[0] < 3:
        return np.nan
    try:
        return ConvexHull(points).volume  # 'volume' is the 2D area for 2D input
    except Exception:
        return np.nan


def sb_to_meters(x, y, field_dimen=(105., 68.)):
    """ sb_to_meters(x, y, field_dimen=(105.,68.))

    Converts StatsBomb's 120x80-yard, top-left-origin coordinates to a centred,
    metres-based system -- the same convention Metrica_IO.to_metric_coordinates uses,
    so the existing EPV grid (built for that convention) works unmodified.
    """
    x_m = (np.asarray(x, dtype=float) / 120. - 0.5) * field_dimen[0]
    y_m = -1. * (np.asarray(y, dtype=float) / 80. - 0.5) * field_dimen[1]
    return x_m, y_m


class FreezeFramePlayer(object):
    """ FreezeFramePlayer

    Minimal duck-typed replacement for Metrica_PitchControl.player, built directly from
    a StatsBomb freeze-frame row instead of a Metrica tracking row. Velocity is always
    zero -- StatsBomb 360 data has no velocity information at all, which is exactly the
    freeze-frame limitation this whole lesson is about. Implements only the attributes
    and methods calculate_pitch_control_at_target actually touches.
    """

    def __init__(self, position, params, is_gk=False):
        self.position = position
        self.velocity = np.array([0., 0.])
        self.vmax = params['max_player_speed']
        self.reaction_time = params['reaction_time']
        self.tti_sigma = params['tti_sigma']
        self.lambda_att = params['lambda_att']
        self.lambda_def = params['lambda_gk'] if is_gk else params['lambda_def']
        self.PPCF = 0.

    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0.
        r_reaction = self.position + self.velocity * self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final - r_reaction) / self.vmax
        return self.time_to_intercept

    def probability_intercept_ball(self, T):
        return 1. / (1. + np.exp(-np.pi / np.sqrt(3.0) / self.tti_sigma * (T - self.time_to_intercept)))


def players_from_freeze_frame(frame_for_event, params, teammate_value, field_dimen=(105., 68.)):
    """ players_from_freeze_frame(frame_for_event, params, teammate_value, field_dimen=(105.,68.))

    Builds a list of FreezeFramePlayer for one side ('teammate_value' True or False) of
    a single event's 360 freeze frame. Uses the 'keeper' column mplsoccer already
    provides to flag the goalkeeper (StatsBomb 360 data has no player_id/name for
    non-actor players, so identity isn't otherwise available).
    """
    subset = frame_for_event[frame_for_event['teammate'] == teammate_value]
    players = []
    for _, row in subset.iterrows():
        x_m, y_m = sb_to_meters(row['x'], row['y'], field_dimen=field_dimen)
        if np.isnan(x_m) or np.isnan(y_m):
            continue
        is_gk = bool(row['keeper']) if 'keeper' in row and pd.notna(row['keeper']) else False
        players.append(FreezeFramePlayer(np.array([x_m, y_m]), params, is_gk=is_gk))
    return players


def infer_attack_directions(df_event, field_dimen=(105., 68.)):
    """ infer_attack_directions(df_event, field_dimen=(105.,68.))

    Returns {(team_name, period): +1 or -1}, inferred from the mean x-location (after
    centring) of each team's own shots in that period -- StatsBomb data has no
    goalkeeper-tracking equivalent of Metrica_IO.find_playing_direction, so this is an
    approximation. Falls back to the opposite of the team's other-period direction if
    they had no shots in a given period.
    """
    directions = {}
    shots = df_event[df_event['type_name'] == 'Shot']
    for (team, period), grp in shots.groupby(['team_name', 'period']):
        x_m, _ = sb_to_meters(grp['x'].values, np.zeros(len(grp)), field_dimen=field_dimen)
        directions[(team, period)] = 1 if np.nanmean(x_m) > 0 else -1

    teams = [t for t in df_event['team_name'].dropna().unique()]
    periods = sorted(df_event['period'].dropna().unique())
    for team in teams:
        for period in periods:
            if (team, period) in directions:
                continue
            other = [p for p in periods if p != period and (team, p) in directions]
            directions[(team, period)] = -1 * directions[(team, other[0])] if other else 1
    return directions


def score_event(event_row, frame_for_event, EPV, params, attack_directions, field_dimen=(105., 68.)):
    """ score_event(event_row, frame_for_event, EPV, params, attack_directions, field_dimen=(105.,68.))

    H1 metrics for one pass event, from a StatsBomb 360 freeze frame alone:
        hull_area: convex hull area (m^2) of the defending team's visible positions
                   (Metrica_ArMatComparison.hull_area, reused directly)
        controlled_value: sum over attacking-team freeze-frame positions of
                   Patt(position) * EPV(position) -- the freeze-frame-only pitch
                   control + value estimate for the attacking team

    Returns None if there isn't enough freeze-frame data to compute a hull (fewer than
    3 defenders) or no attacking players at all.
    """
    team = event_row['team_name']
    period = event_row['period']
    attack_direction = attack_directions.get((team, period), 1)

    ball_x, ball_y = sb_to_meters(event_row['x'], event_row['y'], field_dimen=field_dimen)
    if np.isnan(ball_x) or np.isnan(ball_y):
        return None
    ball_start_pos = np.array([ball_x, ball_y])

    attacking_players = players_from_freeze_frame(frame_for_event, params, True, field_dimen=field_dimen)
    defending_players = players_from_freeze_frame(frame_for_event, params, False, field_dimen=field_dimen)
    if len(attacking_players) < 1 or len(defending_players) < 3:
        return None

    h_area = hull_area(np.array([p.position for p in defending_players]))
    if np.isnan(h_area):
        return None

    controlled_value = 0.
    for p in attacking_players:
        Patt, _ = mpc.calculate_pitch_control_at_target(p.position, attacking_players, defending_players, ball_start_pos, params)
        epv = mepv.get_EPV_at_location(p.position, EPV, attack_direction=attack_direction, field_dimen=field_dimen)
        controlled_value += Patt * epv

    return h_area, controlled_value
