import pandas as pd
import numpy as np

def engineer_race_features(laps_df, results_df, weather_df, pre_race_elos, circuit_id, current_year=None, driver_history_df=None):
    """
    Builds the feature matrix for the Race Win model. Includes past 3-year historical track features.
    """
    if laps_df.empty or results_df.empty:
        return pd.DataFrame()
        
    df = laps_df.copy()
    
    # Merge weather based on closest Time
    if not weather_df.empty and 'Time' in weather_df.columns:
        df = pd.merge_asof(df.sort_values('Time'), 
                          weather_df[['Time', 'TrackTemp']].sort_values('Time'), 
                          on='Time', direction='backward')
    else:
        df['TrackTemp'] = 30.0
        
    # Convert timedelta to seconds
    if 'LapTime' in df.columns:
        df['LapTime_s'] = df['LapTime'].dt.total_seconds()
        
    features = []
    
    for driver_num, group in df.groupby('DriverNumber'):
        driver_info = results_df[results_df['DriverNumber'] == driver_num]
        if driver_info.empty:
            continue
            
        is_winner = 1 if driver_info.iloc[0].get('Position', 0) == 1.0 else 0
        driver_code = driver_info.iloc[0].get('Abbreviation', 'UNK')
        elo = pre_race_elos.get(driver_code, 1500)
        
        # Calculate 3-year track stats
        track_past_3_years_wins = 0
        track_past_3_years_points = 0
        track_past_3_years_avg_pos = 20.0
        
        if driver_history_df is not None and not driver_history_df.empty and current_year is not None:
            mask = (
                (driver_history_df['DriverCode'] == driver_code) & 
                (driver_history_df['Location'] == circuit_id) & 
                (driver_history_df['Year'] >= current_year - 3) & 
                (driver_history_df['Year'] < current_year)
            )
            past_races = driver_history_df[mask]
            if not past_races.empty:
                track_past_3_years_wins = past_races['IsWin'].sum()
                track_past_3_years_points = past_races['Points'].sum()
                track_past_3_years_avg_pos = past_races['Position'].mean()
        
        group = group.sort_values('LapNumber')
        total_laps = group['LapNumber'].max()
        
        pit_count = 0
        laps_on_tire = 0
        
        for idx, row in group.iterrows():
            if pd.notnull(row.get('PitOutTime')):
                pit_count += 1
                laps_on_tire = 0
                
            laps_on_tire += 1
            compound = row.get('Compound', 'UNKNOWN')
            
            # Use '1' as TrackStatus clear, any other value as safety car/VSC
            sc_active = 0
            if 'TrackStatus' in row and row['TrackStatus'] != '1':
                sc_active = 1
                
            feature_row = {
                'driver_code': driver_code, 
                'current_position': row.get('Position', 0),
                'track_past_3_years_wins': track_past_3_years_wins,
                'track_past_3_years_points': track_past_3_years_points,
                'track_past_3_years_avg_pos': track_past_3_years_avg_pos,
                'laps_on_tire': laps_on_tire,
                'tire_compound': compound,
                'pit_stop_count': pit_count,
                'gap_to_leader': 0.0, # Needs cross-driver lookup implementation
                'gap_to_pit_window': 0.0,
                'safety_car_active': sc_active,
                'track_temp': row.get('TrackTemp', 30.0),
                'lap_number_norm': row.get('LapNumber', 1) / (total_laps if total_laps > 0 else 1),
                'driver_elo_rating': elo,
                'fuel_adjusted_pace': row.get('LapTime_s', 0) - ((total_laps - row.get('LapNumber', 1)) * 0.035) 
                                      if total_laps and 'LapTime_s' in row and not pd.isna(row['LapTime_s']) else 0,
                'circuit_id': circuit_id,
                'drs_zones': 2,
                'is_winner': is_winner
            }
            features.append(feature_row)
            
    res_df = pd.DataFrame(features)
    
    # One-hot encode tire compound
    if not res_df.empty:
        compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'UNKNOWN']
        for c in compounds:
            expected_c = c
            # FastF1 uses title case for compounds e.g., 'Soft'
            res_df[f'compound_{c}'] = (res_df['tire_compound'].str.upper() == c).astype(int)
        res_df.drop('tire_compound', axis=1, inplace=True)
        
    return res_df


def engineer_championship_features(season_summary_df):
    """
    Builds the feature matrix for the Championship model.
    """
    if season_summary_df.empty:
        return pd.DataFrame()
    
    # We drop the identifiers to make it purely feature-based for xgboost
    features = season_summary_df.drop(columns=['driver_code', 'year', 'final_points'], errors='ignore')
    return features.apply(pd.to_numeric, errors='coerce').fillna(0)
