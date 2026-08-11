#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:52:19 2020

Module for measuring player velocities, smoothed using a Savitzky-Golay filter, with Metrica tracking data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)

"""
import numpy as np
import scipy.signal as signal

def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home','Away'] ] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()
    
    # index of first frame in second half
    second_half_idx = team.Period.idxmax()
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player+"_x"].diff() / dt
        vy = team[player+"_y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan

        # vx/vy can have NaN at this point from two sources: the very first sample of a
        # .diff() series (nothing precedes it -- guaranteed, every call) and, occasionally,
        # a maxspeed-outlier rejection landing on the last sample before a half-time
        # boundary (seen even in Metrica's own sample data). Modern scipy's savgol_filter
        # (mode='interp') hard-rejects ANY NaN in its input, where older scipy tolerated
        # them; interpolate over these isolated points -- extrapolating from the nearest
        # valid value at the very edges -- before smoothing sees them.
        if smoothing and (vx.isna().any() or vy.isna().any()):
            vx = vx.interpolate(limit_direction='both')
            vy = vy.interpolate(limit_direction='both')

        # A tracking read confined to a single period (e.g. one substitution-safe chunk
        # of a SkillCorner match, see SkillCorner_IO.read_full_match_with_velocities)
        # has a constant Period column, so second_half_idx (= Period.idxmax(), the
        # first occurrence of the max) lands on the very first row -- splitting into a
        # 1-frame "first half" and an (n-1)-frame "second half". savgol_filter rejects
        # a window (window_length=7) longer than its input, so that 1-frame slice
        # errors out. When there's only one period present, just filter the whole
        # chunk in one pass instead of splitting.
        single_period = team['Period'].nunique() <= 1

        if smoothing:
            if filter_=='Savitzky-Golay':
                if single_period:
                    vx.loc[:] = signal.savgol_filter(vx,window_length=window,polyorder=polyorder)
                    vy.loc[:] = signal.savgol_filter(vy,window_length=window,polyorder=polyorder)
                else:
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                    vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                    vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window
                if single_period:
                    vx.loc[:] = np.convolve( vx, ma_window, mode='same' )
                    vy.loc[:] = np.convolve( vy, ma_window, mode='same' )
                else:
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' )
                    vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' )
                    vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' )
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return team

def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    team = team.drop(columns=columns)
    return team
    