# F1 Analytics Pipeline

A full-stack Python project combining **FastF1**, two **XGBoost** models, and a **Plotly Dash** dashboard. 

## Features
1. **Historical Race Browser:** Replay any 2024 race lap-by-lap
2. **Live Race Tracker:** Real-time car dots on a track map with live win probability
3. **Championship Predictor:** Predicts end-of-season points based on cumulative performance and ELO.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the models (this will take some time and cache FastF1 data):
```bash
python models/train_models.py
```

3. Launch the dashboard:
```bash
python dashboard/app.py
```
Then navigate to `http://127.0.0.1:8050` in your web browser.
