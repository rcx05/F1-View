import pandas as pd
from collections import defaultdict

class ELOCalculator:
    def __init__(self, k_factor=32, initial_elo=1500):
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.driver_elos = defaultdict(lambda: self.initial_elo)
        
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        
    def update_ratings(self, race_results_df):
        """
        Takes a FastF1 results DataFrame for a single race and updates ELOs.
        Returns a dict of driver -> ELO at the *start* of this race.
        """
        if race_results_df is None or race_results_df.empty:
            return {}

        # Store ratings before the race starts for feature engineering
        pre_race_elos = {drv: self.driver_elos[drv] for drv in race_results_df['Abbreviation'] if pd.notnull(drv)}
        
        # We need to drop drivers who didn't start (DNS) or don't have a valid position
        valid_results = race_results_df.dropna(subset=['Position'])
        valid_results = valid_results.sort_values('Position')
        
        driver_codes = valid_results['Abbreviation'].tolist()
        n_drivers = len(driver_codes)
        
        # Calculate rating changes
        changes = defaultdict(float)
        
        for i in range(n_drivers):
            for j in range(i + 1, n_drivers):
                drv_a = driver_codes[i]
                drv_b = driver_codes[j]
                
                # drv_a beat drv_b because of sorting by position
                rating_a = self.driver_elos[drv_a]
                rating_b = self.driver_elos[drv_b]
                
                exp_a = self.expected_score(rating_a, rating_b)
                exp_b = self.expected_score(rating_b, rating_a)
                
                # Actual score: 1 for A (win), 0 for B (loss)
                changes[drv_a] += self.k_factor * (1 - exp_a) / (n_drivers - 1)
                changes[drv_b] += self.k_factor * (0 - exp_b) / (n_drivers - 1)
                
        # Apply changes
        for drv, change in changes.items():
            self.driver_elos[drv] += change
            
        return pre_race_elos
