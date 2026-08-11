#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for reading SkillCorner's open broadcast-tracking data
(https://github.com/SkillCorner/opendata, 10 matches, 2024/25 Australian A-League) into
the same tracking DataFrame shape Metrica_IO.tracking_data() produces -- one row per
frame, columns named "{Home|Away}_{jersey}_x"/"_y" plus "ball_x"/"ball_y", "Period" and
"Time [s]" -- so Metrica_Velocities.calc_player_velocities and Metrica_PitchControl's
Spearman model run against it unmodified. Where StatsBomb_ArMatSpaceControl.py adapts
this lesson's pitch-control code to freeze-frame data with no velocity at all,
SkillCorner is a second *tracking* dataset -- a genuine second ground-truth source,
not an approximation, useful for checking the pipeline isn't overfit to Metrica's
two sample games specifically.

Three things this module has to handle that Metrica_IO.tracking_data() never does,
because Metrica's bundled sample data doesn't have them:

1. SkillCorner's coordinate convention -- meters, pitch-centre origin -- is already
   Metrica_IO.to_metric_coordinates' *target* convention, so no transform is needed.
   (This is luck, not a general property of tracking providers; verify it for any
   other source before assuming the same.)

2. The full squad (subs included) is listed for every match, but a substitute has no
   tracking data outside their own playing-time window -- unlike Metrica's sample
   data, which only ever lists the fixed starting XI. Handled by keeping only the
   columns for players who are actually on the pitch in the requested window
   (see `_squad_on_pitch_columns`).

3. Broadcast tracking has real frame dropouts -- in a spot-check on one match, about
   1 in 4 frames in a given window had NO tracking output at all (broadcast replays
   and cutaways -- not per-player gaps: whenever SkillCorner *does* report a frame,
   every tracked player in it has a valid, already-extrapolated position). Handled
   by reindexing to the full contiguous frame range within each period and linearly
   interpolating gaps up to `interpolate_limit` frames (default 20 = 2s at 10fps);
   anything longer is left as a genuine hole and dropped downstream by whichever
   function needs a complete row (e.g. Metrica_PitchControl.initialise_players,
   via `player.inframe`).

4. **Substitutions and `calc_player_velocities`.** A substitute's columns are NaN for
   their entire time on the bench -- tens of minutes, far past `interpolate_limit` --
   which `read_tracking_data` itself handles fine (it only requires the *ball* to have
   output for a row to survive, not every player ever listed; a substitute's own NaN
   stretch is left as-is, same as any out-of-frame player). But
   `Metrica_Velocities.calc_player_velocities` smooths each player's column as one
   continuous stretch across a whole half, which breaks (the same NaN-intolerant
   Savitzky-Golay issue described in `Metrica_Velocities.py`, but for a stretch far
   longer than that fix's isolated-point interpolation can cover) if fed a column
   that's genuinely NaN for tens of minutes before a substitute enters. Metrica's own
   sample data never hits this -- its tracking files only ever list the fixed starting
   XI. Not fixed here: compute velocities on a frame_windows read that doesn't span a
   substitution (e.g. one period, checked against
   `match_json['players'][].start_time`/`end_time` first), or restrict to players who
   were on the pitch for the entire window before calling calc_player_velocities.

Data source: https://github.com/SkillCorner/opendata (CC BY-NC-SA 4.0). The tracking
file for a single match is ~90MB and Git-LFS-hosted, so it is downloaded on demand into
`data_dir` (gitignored) rather than committed -- see `download_match`.
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
LFS_BASE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"

FRAME_RATE_HZ = 10.0  # SkillCorner tracking is sampled at 10 fps


# ----------------------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------------------

def download_matches_json(data_dir):
    """ download_matches_json(data_dir)

    Downloads the match index (id, teams, date, competition) for all matches in the
    open dataset. Returns it as a list of dicts. Cached to `{data_dir}/matches.json`.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "matches.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{RAW_BASE}/matches.json", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def download_match(match_id, data_dir, force=False, dynamic_events=False):
    """ download_match(match_id, data_dir, force=False, dynamic_events=False)

    Downloads the files this module needs for `match_id` into `data_dir`:
    `{id}_match.json` (lineups, team/player metadata, pitch size, period boundaries)
    and `{id}_tracking_extrapolated.jsonl` (~90MB, Git-LFS-hosted, so it is fetched
    through the LFS media endpoint rather than raw.githubusercontent.com, which would
    otherwise silently return a ~130-byte LFS pointer file instead of the real data).
    Skips re-downloading a file that already exists, unless `force=True`.

    `dynamic_events=True` also downloads `{id}_dynamic_events.csv` (SkillCorner's
    "Game Intelligence" pass/event log, needed by SkillCorner_EPV.load_pass_events --
    a few MB, not LFS-hosted, so a plain fetch).
    """
    os.makedirs(data_dir, exist_ok=True)
    match_json_path = os.path.join(data_dir, f"{match_id}_match.json")
    tracking_path = os.path.join(data_dir, f"{match_id}_tracking_extrapolated.jsonl")

    if force or not os.path.exists(match_json_path):
        urllib.request.urlretrieve(f"{RAW_BASE}/matches/{match_id}/{match_id}_match.json", match_json_path)
    if force or not os.path.exists(tracking_path):
        urllib.request.urlretrieve(
            f"{LFS_BASE}/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl", tracking_path
        )
    if dynamic_events:
        events_path = os.path.join(data_dir, f"{match_id}_dynamic_events.csv")
        if force or not os.path.exists(events_path):
            urllib.request.urlretrieve(f"{RAW_BASE}/matches/{match_id}/{match_id}_dynamic_events.csv", events_path)
        return match_json_path, tracking_path, events_path
    return match_json_path, tracking_path


# ----------------------------------------------------------------------------------
# Match metadata
# ----------------------------------------------------------------------------------

def read_match_json(data_dir, match_id):
    """ read_match_json(data_dir, match_id)

    Loads `{match_id}_match.json` as-is (must already be downloaded, see
    `download_match`). Useful on its own -- e.g. to read period frame boundaries before
    deciding what `frame_windows` to pass to `read_tracking_data` -- as well as being
    used internally by `match_summary`.
    """
    path = os.path.join(data_dir, f"{match_id}_match.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _player_metadata(match_json):
    """ Returns {player_id: {'side': 'Home'|'Away', 'jersey': str, 'is_gk': bool, 'name': str}}.

    player_id here is `players[].id`, which is exactly what the tracking file's
    `player_data[].player_id` refers to (verified directly against a downloaded match --
    not `trackable_object`, which is a different id SkillCorner also carries).
    """
    home_id = match_json["home_team"]["id"]
    meta = {}
    for p in match_json["players"]:
        side = "Home" if p["team_id"] == home_id else "Away"
        meta[p["id"]] = {
            "side": side,
            "jersey": str(p["number"]),
            "is_gk": p["player_role"]["acronym"] == "GK",
            "name": p["short_name"],
        }
    return meta


def gk_numbers(match_json):
    """ Returns (home_gk_jersey, away_gk_jersey) as strings, for Metrica_PitchControl's
    GK_numbers argument. Read directly from `player_role.acronym == 'GK'`, which is more
    direct than Metrica_IO's own GK-by-position heuristic. """
    home_id = match_json["home_team"]["id"]
    home_gk = next(str(p["number"]) for p in match_json["players"]
                   if p["team_id"] == home_id and p["player_role"]["acronym"] == "GK")
    away_gk = next(str(p["number"]) for p in match_json["players"]
                   if p["team_id"] != home_id and p["player_role"]["acronym"] == "GK")
    return home_gk, away_gk


def match_summary(match_json):
    """ Returns a small dict of match-level facts used throughout this module and its
    callers: field_dimen, GK_numbers, team names, and each period's (start_frame,
    end_frame) and attacking direction ('left_to_right' / 'right_to_left') for the
    home team -- the away team always attacks the opposite way. """
    return {
        "field_dimen": (match_json["pitch_length"], match_json["pitch_width"]),
        "GK_numbers": gk_numbers(match_json),
        "home_team": match_json["home_team"]["short_name"],
        "away_team": match_json["away_team"]["short_name"],
        "periods": [
            {
                "period": p["period"],
                "start_frame": p["start_frame"],
                "end_frame": p["end_frame"],
                "home_direction": match_json["home_team_side"][p["period"] - 1],
            }
            for p in match_json["match_periods"]
        ],
    }


# ----------------------------------------------------------------------------------
# Tracking data
# ----------------------------------------------------------------------------------

def _squad_on_pitch_columns(tracking, prefix):
    """ A substitute has no tracking data outside their own playing-time window, so their
    columns are NaN for most of a match-length read -- unlike Metrica's sample data, whose
    tracking files only ever list the fixed starting XI. calc_player_velocities can't
    smooth through a column that's entirely (or mostly) NaN, so restrict to players with
    at least one valid frame in the read window; short remaining gaps are interpolated by
    `read_tracking_data` before this is used, so "at least one valid frame" is a low bar
    that only excludes players who plainly never appeared in the window at all. """
    keep = []
    for c in tracking.columns:
        if c.startswith(prefix) and c.endswith("_x"):
            base = c[:-2]
            if tracking[f"{base}_x"].notna().any():
                keep += [f"{base}_x", f"{base}_y"]
    return keep


def read_tracking_data(data_dir, match_id, frame_windows=None, interpolate_limit=20):
    """ read_tracking_data(data_dir, match_id, frame_windows=None, interpolate_limit=20)

    Reads `{match_id}_tracking_extrapolated.jsonl` (must already be downloaded, see
    `download_match`) and returns (tracking_home, tracking_away) in the same column
    shape Metrica_IO.tracking_data() produces -- ready for
    Metrica_Velocities.calc_player_velocities and Metrica_PitchControl, but NOT yet run
    through either.

    Parameters
    ----------
    frame_windows: list of (start_frame, end_frame) tuples, inclusive, or None to read
        every period in `{match_id}_match.json`'s match_periods. WARNING: see point 4
        in the module docstring -- `None` returns an empty result on any match with a
        substitution, silently. Prefer a single substitution-free window (e.g. one
        period, or up to the first substitution) until that's fixed.
    interpolate_limit: max consecutive missing frames to linearly interpolate across
        (see module docstring, point 3). Gaps longer than this are left as NaN and
        dropped.

    Returns
    -------
    tracking_home, tracking_away: DataFrames indexed by Frame, with "Period",
        "Time [s]", one "{side}_{jersey}_x"/"_y" column pair per player who appears at
        least once in the read window, "ball_x"/"ball_y", and "possession_team"
        ('Home'/'Away'/None) -- useful for picking an attacking side per frame without
        needing an adapted event schema (see `read_match_data`).
    """
    match_json = read_match_json(data_dir, match_id)
    player_meta = _player_metadata(match_json)

    if frame_windows is None:
        frame_windows = [(p["start_frame"], p["end_frame"]) for p in match_json["match_periods"]]

    tracking_path = os.path.join(data_dir, f"{match_id}_tracking_extrapolated.jsonl")
    rows = {}
    with open(tracking_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            fr = rec["frame"]
            if rec.get("period") is None or not any(lo <= fr <= hi for lo, hi in frame_windows):
                continue
            possession_group = rec.get("possession", {}).get("group")
            possession_team = {"home team": "Home", "away team": "Away"}.get(possession_group)
            row = {"Period": rec["period"], "ball_x": rec["ball_data"]["x"], "ball_y": rec["ball_data"]["y"],
                   "possession_team": possession_team}
            for pdata in rec["player_data"]:
                meta = player_meta.get(pdata["player_id"])
                if meta is None:
                    continue
                row[f"{meta['side']}_{meta['jersey']}_x"] = pdata["x"]
                row[f"{meta['side']}_{meta['jersey']}_y"] = pdata["y"]
            rows[fr] = row

    tracking = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    tracking.index.name = "Frame"

    # Reindex to the full contiguous frame range *within each period* (never across the
    # half-time gap) and interpolate short dropouts before anything downstream sees them.
    full_frames = []
    for lo, hi in frame_windows:
        full_frames.extend(range(lo, hi + 1))
    tracking = tracking.reindex(full_frames)
    tracking["Period"] = tracking["Period"].ffill().bfill()
    tracking["Time [s]"] = tracking.index / FRAME_RATE_HZ
    pos_cols = [c for c in tracking.columns if c.endswith("_x") or c.endswith("_y")]
    tracking[pos_cols] = tracking[pos_cols].interpolate(method="linear", limit=interpolate_limit, limit_area="inside")
    # Drop rows with no tracking output at all (point 3), identified via the ball alone --
    # NOT via every player column, which would (silently, previously) empty the whole
    # result on any match with a substitution (point 4): a substitute's columns are
    # genuinely NaN for their entire time on the bench, far past interpolate_limit, and
    # requiring all of them non-null at once means no row ever qualifies. Individual
    # players' NaN is left in place -- Metrica_PitchControl.initialise_players already
    # excludes an out-of-frame player via `player.inframe` rather than erroring.
    tracking = tracking.dropna(subset=["ball_x", "ball_y"])
    if len(tracking) == 0:
        raise ValueError(
            f"No tracking output at all in frame_windows={frame_windows} for match {match_id} -- "
            "check the window falls inside a period's (start_frame, end_frame)."
        )

    extra_cols = ["ball_x", "ball_y", "possession_team"]
    home_cols = ["Period", "Time [s]"] + _squad_on_pitch_columns(tracking, "Home_") + extra_cols
    away_cols = ["Period", "Time [s]"] + _squad_on_pitch_columns(tracking, "Away_") + extra_cols
    return tracking[home_cols].copy(), tracking[away_cols].copy()


def read_match_data(data_dir, match_id, frame_windows=None, interpolate_limit=20):
    """ read_match_data(data_dir, match_id, frame_windows=None, interpolate_limit=20)

    Mirrors Metrica_IO.read_match_data's role: one call to get everything needed to run
    the pitch-control pipeline. Unlike Metrica_IO, there is no third `events` DataFrame
    -- SkillCorner's `{id}_dynamic_events.csv` doesn't share Metrica's RawEventsData.csv
    schema (own event taxonomy, unscaled x/y, see the SkillCorner opendata README), so it
    isn't adapted here. `generate_pitch_control_for_event`-style event lookups need that
    adaptation as a separate step; direct-target pitch control (pick a frame and a target
    position yourself) works with what this function returns.

    Returns
    -------
    tracking_home, tracking_away, match_meta (see `match_summary`)
    """
    match_json = read_match_json(data_dir, match_id)
    tracking_home, tracking_away = read_tracking_data(
        data_dir, match_id, frame_windows=frame_windows, interpolate_limit=interpolate_limit
    )
    return tracking_home, tracking_away, match_summary(match_json)
