import xgboost as xgb
import os
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'championship_model.json')

class ChampionshipModel:
    def __init__(self):
        self.model = xgb.XGBRegressor()
        if os.path.exists(MODEL_PATH):
            self.model.load_model(MODEL_PATH)
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            self.loaded = False
            return
        self.loaded = True
            
    def predict_points(self, features_df):
        """
        Takes a dataframe of driver standings features and returns predicted final season points.
        """
        if features_df.empty or not self.loaded:
            return []
        
        X = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        points = self.model.predict(X)
        return np.maximum(points, 0)
