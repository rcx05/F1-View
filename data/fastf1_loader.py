import os
import fastf1
import pandas as pd
import numpy as np

# Setup caching
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def load_race(year, round_num):
    """
    Loads historical race data for a given year and round using FastF1.
    Normalises telemetry X/Y for mapping.
    """
    session = fastf1.get_session(year, round_num, 'R')
    session.load(telemetry=True, weather=True, messages=True)

    laps = session.laps
    results = session.results
    weather = session.weather_data
    
    # Process position data and normalise x, y
    all_pos_data = []
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    pos_data_dict = {}
    try:
        session_pos = session.pos_data
        if session_pos:
            # First pass: find global min/max for the circuit
            for driver, pos_df in session_pos.items():
                if 'X' in pos_df.columns and 'Y' in pos_df.columns:
                    cur_x_min, cur_x_max = pos_df['X'].min(), pos_df['X'].max()
                    cur_y_min, cur_y_max = pos_df['Y'].min(), pos_df['Y'].max()
                    
                    if cur_x_min < x_min: x_min = cur_x_min
                    if cur_x_max > x_max: x_max = cur_x_max
                    if cur_y_min < y_min: y_min = cur_y_min
                    if cur_y_max > y_max: y_max = cur_y_max
                    
            # Second pass: augment and store
            for driver, pos_df in session_pos.items():
                if 'X' in pos_df.columns and 'Y' in pos_df.columns:
                    pos_norm = pos_df.copy()
                    
                    if x_max > x_min:
                        pos_norm['x_norm'] = (pos_norm['X'] - x_min) / (x_max - x_min)
                    else:
                        pos_norm['x_norm'] = 0.5
                        
                    if y_max > y_min:
                        pos_norm['y_norm'] = (pos_norm['Y'] - y_min) / (y_max - y_min)
                    else:
                        pos_norm['y_norm'] = 0.5
                        
                    pos_data_dict[driver] = pos_norm
    except Exception as e:
        print(f"Warning: unable to load pos_data: {e}")

    return {
        'laps': laps,
        'results': results,
        'weather': weather,
        'pos_data': pos_data_dict,
        'session_info': session.session_info
    }

def get_schedule(year):
    """Gets the race schedule for a specific year."""
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] != 'testing']
    return schedule

def load_telemetry(year, round_num, driver1, driver2):
    """
    Loads telemetry for the fastest lap of two selected drivers.
    Returns a dictionary mapping driver code to their telemetry lists (Speed, Distance, X, Y).
    """
    session = fastf1.get_session(year, round_num, 'R')
    # Use load with telemetry
    session.load(telemetry=True, weather=False, messages=False)

    laps = session.laps
    res = {}
    
    for d in [driver1, driver2]:
        if not d: continue
        d = str(d).upper()
        d_laps = laps.pick_driver(d)
        if d_laps.empty: continue
            
        fastest = d_laps.pick_fastest()
        if pd.isna(fastest['LapTime']): continue
        
        try:
            tel = fastest.get_telemetry()
        except Exception:
            continue
        
        # Add Lap Time as a reference
        res[d] = {
            'Distance': tel['Distance'].tolist(),
            'Speed': tel['Speed'].tolist(),
            'Throttle': tel['Throttle'].tolist(),
            'Brake': tel['Brake'].tolist(),
            'X': tel['X'].tolist(),
            'Y': tel['Y'].tolist(),
            'LapTime': str(fastest['LapTime'])
        }
        
    return res

def load_lap_replay(year, round_num, lap_number=None):
    """
    Extracts positional tracking data for all drivers. Assumes a specific lap if lap_number 
    is provided. If lap_number is None, extracts the entire race downsampled for performance.
    Also returns the fastest lap's X/Y coordinates as a background track layout.
    """
    session = fastf1.get_session(year, round_num, 'R')
    session.load(telemetry=True, weather=False, messages=False)
    
    try:
        fastest_lap = session.laps.pick_fastest()
        track_tel = fastest_lap.get_telemetry()
        track_df = track_tel[['X', 'Y']].copy()
    except Exception:
        track_df = pd.DataFrame()
    
    drivers = pd.unique(session.laps['Driver'])
    all_data = []
    
    for drv in drivers:
        try:
            drv_laps = session.laps.pick_driver(drv)
            if drv_laps.empty: continue
            
            if lap_number is not None:
                lap = drv_laps[drv_laps['LapNumber'] == lap_number]
                if lap.empty: continue
                lap = lap.iloc[0]
                tel = lap.get_telemetry()
                if tel.empty: continue
                # Relative time from 0 for the single lap
                tel['RelTime'] = tel['Time'] - tel['Time'].min()
                # 2Hz downsample for a single lap
                freq = 2.0
            else:
                tel = drv_laps.get_telemetry()
                if tel.empty: continue
                # Absolute session time to maintain synchronization across all drivers all race
                tel['RelTime'] = tel['Time']
                # 0.5Hz downsample (1 frame every 2 seconds) for full race to avoid animation crashes
                freq = 0.5
                
            tel['Frame'] = (tel['RelTime'].dt.total_seconds() * freq).round() / freq
            
            subset = tel[['Frame', 'X', 'Y', 'Speed']].copy()
            subset['Driver'] = drv
            
            subset = subset.groupby('Frame').first().reset_index()
            all_data.append(subset)
        except Exception:
            pass
            
    if not all_data:
        return pd.DataFrame(), track_df
        
    combined = pd.concat(all_data)
    combined = combined.sort_values(['Frame', 'Driver'])
    return combined, track_df

def load_microsectors(year, round_num, driver1, driver2, num_sectors=25):
    """
    Computes which driver is faster in each mini-sector of their fastest lap.
    Returns a dataframe with X, Y, and the fastest driver abbreviation.
    """
    session = fastf1.get_session(year, round_num, 'Q')
    try:
        session.load(telemetry=True, weather=False, messages=False)
    except Exception:
        session = fastf1.get_session(year, round_num, 'R')
        session.load(telemetry=True, weather=False, messages=False)

    laps = session.laps
    try:
        laps_d1 = laps.pick_driver(str(driver1).upper()).pick_fastest()
        laps_d2 = laps.pick_driver(str(driver2).upper()).pick_fastest()
        tel_d1 = laps_d1.get_telemetry().add_distance()
        tel_d2 = laps_d2.get_telemetry().add_distance()
    except Exception:
        return pd.DataFrame()

    tel_d1['Driver'] = driver1
    tel_d2['Driver'] = driver2

    # Merge telemetry
    telemetry = pd.concat([tel_d1, tel_d2])
    
    # Calculate mini sectors
    total_distance = max(telemetry['Distance'])
    minisector_length = total_distance / num_sectors

    telemetry['Minisector'] = (telemetry['Distance'] / minisector_length).astype(int) + 1

    # Find the fastest driver per mini sector
    average_speed = telemetry.groupby(['Minisector', 'Driver'])['Speed'].mean().reset_index()
    fastest_driver = average_speed.loc[average_speed.groupby('Minisector')['Speed'].idxmax()]
    fastest_driver = fastest_driver[['Minisector', 'Driver']].rename(columns={'Driver': 'Fastest_Driver'})

    # Merge back to telemetry so we can plot it
    telemetry = telemetry.merge(fastest_driver, on='Minisector')

    # We only need one driver's X/Y coordinates to draw the shape of the track
    track_coords = telemetry[telemetry['Driver'] == driver1].copy()
    return track_coords[['X', 'Y', 'Minisector', 'Fastest_Driver']]

