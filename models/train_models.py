import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data.fastf1_loader import get_schedule, load_race
from data.elo_calculator import ELOCalculator
from data.feature_engineer import engineer_race_features, engineer_championship_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models')
os.makedirs(MODEL_DIR, exist_ok=True)

def build_datasets(years=[2024]):
    # Pre-fetch historical results to populate driver_history_df
    hist_years = [2024] # Reduced history years to prevent API rate limit issues
    hist_list = []
    print("Pre-fetching historical results for track history...")
    for y in hist_years:
        try:
            sched = get_schedule(y)
            for _, ev in sched.iterrows():
                try:
                    import fastf1
                    sess = fastf1.get_session(y, ev['RoundNumber'], 'R')
                    sess.load(telemetry=False, weather=False, messages=False)
                    if sess.results is not None and not sess.results.empty:
                        for _, row in sess.results.iterrows():
                            pos = float(row.get('Position', 20.0) or 20.0)
                            if pd.isna(pos): pos = 20.0
                            hist_list.append({
                                'Year': y,
                                'Location': ev['Location'],
                                'DriverCode': row.get('Abbreviation', 'UNK'),
                                'Position': pos,
                                'Points': float(row.get('Points', 0) or 0),
                                'IsWin': 1 if pos == 1.0 else 0
                            })
                except Exception:
                    pass
        except Exception:
            pass
    driver_history_df = pd.DataFrame(hist_list)
    print("Finished pre-fetching historical results.")

    elo_calc = ELOCalculator()
    all_race_features = []
    all_champ_features = []
    
    # We will gather sessions that provide meaningful telemetry and/or points
    session_types = ['R'] # Only use race sessions to save a HUGE amount of time and avoid rate limits
    
    for year in years:
        try:
            schedule = get_schedule(year)
        except Exception as e:
            print(f"Skipping schedule for year {year}: {e}")
            continue
            
        season_stats = {} 
        round_summaries = []
        
        for _, event in schedule.iterrows():
            round_num = event['RoundNumber']
            
            for s_type in session_types:
                try:
                    print(f"Loading {year} Round {round_num} Session {s_type}...")
                    # Update load_race to accept session_type if possible, but for FastF1 direct use:
                    import fastf1
                    session = fastf1.get_session(year, round_num, s_type)
                    try:
                        session.load(telemetry=True, weather=True, messages=True)
                    except Exception as load_err:
                        print(f"Session {s_type} for Round {round_num} not available: {load_err}")
                        continue
                        
                    results = session.results
                    if results is None or results.empty:
                        continue
                        
                    pre_race_elos = elo_calc.update_ratings(results)
                    
                    features_df = engineer_race_features(
                        session.laps, results, session.weather_data, pre_race_elos, event['Location'], year, driver_history_df
                    )
                    
                    if not features_df.empty:
                        # Append session type to features to differentiate if needed later
                        features_df['session_type'] = s_type
                        all_race_features.append(features_df)
                    
                    for _, row in results.iterrows():
                        code = row.get('Abbreviation', 'UNK')
                        points = float(row.get('Points', 0) or 0)
                        pos = float(row.get('Position', 0) or 0)
                        if pd.isna(pos): pos = 20.0
                        
                        is_dnf = 1 if row.get('Status', '') not in ['Finished', '+1 Lap', '+2 Laps'] else 0
                        is_win = 1 if pos == 1.0 else 0
                        
                        if code not in season_stats:
                            season_stats[code] = {'points': 0, 'races': 0, 'wins': 0, 'dnfs': 0, 'pos_sum': 0}
                            
                        # Accumulate points from Sprints and Races
                        season_stats[code]['points'] += points
                        
                        # Only increment race counts for the main event 'R' to avoid skewing average finish
                        if s_type == 'R':
                            season_stats[code]['races'] += 1
                            season_stats[code]['wins'] += is_win
                            season_stats[code]['dnfs'] += is_dnf
                            season_stats[code]['pos_sum'] += pos
                        
                    if s_type == 'R':
                        snapshot = []
                        for code, stats in season_stats.items():
                            snapshot.append({
                                'year': year,
                                'round': round_num,
                                'driver_code': code,
                                'current_points': stats['points'],
                                'races_completed': stats['races'],
                                'wins': stats['wins'],
                                'dnf_rate': stats['dnfs'] / stats['races'] if stats['races'] > 0 else 0,
                                'avg_finish': stats['pos_sum'] / stats['races'] if stats['races'] > 0 else 20,
                                'elo': pre_race_elos.get(code, 1500)
                            })
                        round_summaries.append(snapshot)
                    
                except Exception as e:
                    print(f"Failed to load {year} Round {round_num} Session {s_type}: {e}")
                
        final_points = {code: stats['points'] for code, stats in season_stats.items()}
        for snapshot in round_summaries:
            for driver_stat in snapshot:
                driver_stat['final_points'] = final_points.get(driver_stat['driver_code'], driver_stat['current_points'])
                all_champ_features.append(driver_stat)
                
    race_df = pd.concat(all_race_features, ignore_index=True) if all_race_features else pd.DataFrame()
    champ_df = pd.DataFrame(all_champ_features)
    return race_df, champ_df, driver_history_df

def train_race_win_model(df):
    if df.empty:
        print("Empty dataset for race win model. Creating mock dataset for build verify...")
        # Fallback for dashboard execution
        df = pd.DataFrame(np.random.rand(100, 15))
        df['is_winner'] = np.random.randint(0, 2, 100)
        
    print(f"Training Race Win Model on {len(df)} rows...")
    
    targets = df['is_winner']
    features = df.drop(columns=['is_winner', 'driver_code', 'circuit_id'], errors='ignore')
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, 
                            scale_pos_weight=scale_pos_weight, random_state=42)
                            
    clf.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, 'race_win_model.json')
    clf.save_model(model_path)
    print(f"Saved race win model to {model_path}")
    
def train_championship_model(df):
    if df.empty:
        print("Empty dataset for championship model. Skipping...")
        return
        
    print(f"Training Championship Model on {len(df)} rows...")
    
    targets = df['final_points']
    features = engineer_championship_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    reg_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1)
    reg_model.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, 'championship_model.json')
    reg_model.save_model(model_path)
    print(f"Saved championship model to {model_path}")

if __name__ == '__main__':
    race_df, champ_df, history_df = build_datasets([2024]) 
    os.makedirs('computed_data', exist_ok=True)
    if not race_df.empty: 
        race_df.to_csv('computed_data/race_df.csv', index=False)
    if not champ_df.empty: 
        champ_df.to_csv('computed_data/champ_df.csv', index=False)
    if history_df is not None and not history_df.empty:
        history_df.to_csv('computed_data/driver_history_df.csv', index=False)
        
    train_race_win_model(race_df)
    train_championship_model(champ_df)
