import xgboost as xgb
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'race_win_model.json')

class RaceWinModel:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        if os.path.exists(MODEL_PATH):
            self.model.load_model(MODEL_PATH)
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            self.loaded = False
            return
        self.loaded = True
            
    def predict_win_prob(self, features_df):
        """
        Takes a dataframe of engineered features for all drivers currently on track
        and returns win probabilities.
        """
        if features_df.empty or not self.loaded:
            return np.ones(len(features_df)) / len(features_df) if not features_df.empty else []
            
        drop_cols = ['is_winner', 'driver_code', 'circuit_id']
        X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # predict_proba returns [prob_lose, prob_win]
        probs = self.model.predict_proba(X)[:, 1]
        
        # Softmax mechanism to make probabilities sum to 1 over the grid
        e_x = np.exp(probs - np.max(probs))
        return e_x / e_x.sum()
