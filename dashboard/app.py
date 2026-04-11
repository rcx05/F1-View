import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fastf1_loader import load_race, get_schedule, load_telemetry, load_lap_replay, load_microsectors
from data.live_timing import get_latest_snapshot
from models.championship_model import ChampionshipModel
from models.race_win_model import RaceWinModel
from models.commentary_model import generate_commentary, analyze_head_to_head

# Initialize app with FontAwesome
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
])
app.title = "F1 Analytics Dashboard"

# Load Models
champ_model = ChampionshipModel()
race_model = RaceWinModel()

# UI Layout
app.layout = html.Div(className='dashboard-container', children=[
    html.Div([
        html.H1([
            html.I(className="fa-solid fa-flag-checkered", style={'marginRight': '15px'}),
            html.Span("F1"), " AI Analytics Center"
        ], className="header-title"),
        html.P("Advanced telemetry analysis & real-time race predictions", className="header-subtitle")
    ], className="header-section"),
    
    dcc.Tabs(id='main-tabs', value='tab-championship', className='custom-tabs', children=[
        dcc.Tab(label=' AI Prediction Center', value='tab-championship', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label=' Historical Browser', value='tab-historical', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label=' Driver Telemetry', value='tab-telemetry', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label=' 3D Replay', value='tab-replay', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label=' Live Tracker', value='tab-live', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label=' Weather Impact', value='tab-weather', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label=' Micro-Sectors', value='tab-microsector', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label=' Pit Strategy', value='tab-pitstrategy', className='custom-tab', selected_className='custom-tab--selected')
    ]),
    html.Div(id='tabs-content', className="animated-tab-content")
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('main-tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-championship':
        return html.Div([
            html.Div(className="split-panel", children=[
                html.Div([
                    html.H3([html.I(className="fa-solid fa-trophy", style={'color':'#E10600', 'marginRight':'10px'}), "Championship Prediction"]),
                    html.P("Predicting final season points based on historical trajectory and performance factors."),
                    html.Button([html.I(className="fa-solid fa-brain"), " Load Championship AI"], id="btn-predict-champ", n_clicks=0, className="custom-button"),
                    dcc.Loading(id="loading-champ", type="default", color="#E10600", children=[html.Div(id="champ-output", style={'marginTop': '20px'})])
                ], className="content-card"),
                
                html.Div([
                    html.H3([html.I(className="fa-solid fa-stopwatch", style={'color':'#00d2be', 'marginRight':'10px'}), "Next Race Win Predictor"]),
                    html.P("AI simulated probabilities for the next/current active race based on driver form dynamics."),
                    html.Button([html.I(className="fa-solid fa-robot"), " Generate Race Prediction"], id="btn-predict-race", n_clicks=0, className="custom-button"),
                    dcc.Loading(id="loading-race", type="default", color="#00d2be", children=[html.Div(id="race-output", style={'marginTop': '20px'})])
                ], className="content-card")
            ])
        ])
    elif tab == 'tab-historical':
        return html.Div([
            html.H3([html.I(className="fa-solid fa-database", style={'marginRight':'10px'}), "Historical Race Browser"]),
            html.P("Select a past race to view telemetry and lap data."),
            html.Div([
                dcc.Dropdown(id='hist-year', options=[{'label': str(y), 'value': y} for y in range(2023, 2027)], value=2026, className="custom-dropdown"),
                dcc.Input(id='hist-round', type='number', min=1, max=24, value=1, placeholder="Round number", className="custom-input"),
                html.Button([html.I(className="fa-solid fa-cloud-arrow-down"), " Fetch Race Data"], id="btn-fetch-hist", n_clicks=0, className="custom-button")
            ], className="controls-row"),
            dcc.Loading(id="loading-hist", type="default", color="#E10600", children=[html.Div(id="hist-output")]),
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)', 'marginTop': '30px'}),
            html.Div([
                html.H4([html.I(className="fa-solid fa-microphone-lines", style={'color': '#00d2be', 'marginRight': '10px'}), "AI Race Commentary"], style={'marginBottom': '12px'}),
                html.P("Enter your Google Gemini API key and generate a vivid race analysis powered by AI.", style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '0.9em'}),
                html.Div([
                    dcc.Input(id='gemini-api-key', type='password', placeholder='Gemini API Key (AIza...)', className="custom-input",
                              style={'flex': '1', 'minWidth': '300px'}),
                    html.Button([html.I(className="fa-solid fa-wand-magic-sparkles"), " Generate Commentary"], id="btn-commentary", n_clicks=0, className="custom-button",
                                style={'background': 'linear-gradient(135deg, #00d2be, #0091a1)'})
                ], className="controls-row"),
                dcc.Loading(id="loading-commentary", type="dot", color="#00d2be",
                            children=[html.Div(id="commentary-output", style={'marginTop': '16px'})])
            ], className="content-card", style={'marginTop': '20px', 'border': '1px solid rgba(0,210,190,0.25)'})
        ], className="content-card")
    elif tab == 'tab-telemetry':
        return html.Div([
            html.H3([html.I(className="fa-solid fa-chart-line", style={'marginRight':'10px'}), "Telemetry Comparator"]),
            html.P("Compare fastest laps and track positions for two drivers head-to-head."),
            html.Div([
                dcc.Dropdown(id='tel-year', options=[{'label': str(y), 'value': y} for y in range(2023, 2027)], value=2026, className="custom-dropdown"),
                dcc.Input(id='tel-round', type='number', min=1, max=24, value=1, placeholder="Round number", className="custom-input"),
                dcc.Input(id='tel-driver1', type='text', value='VER', placeholder="Driver 1 (VER)", className="custom-input"),
                dcc.Input(id='tel-driver2', type='text', value='NOR', placeholder="Driver 2 (NOR)", className="custom-input"),
                html.Button([html.I(className="fa-solid fa-code-compare"), " Fetch Telemetry"], id="btn-fetch-tel", n_clicks=0, className="custom-button")
            ], className="controls-row"),
            dcc.Loading(id="loading-tel", type="default", color="#E10600", children=[html.Div(id="tel-output")]),
            dcc.Loading(id="loading-h2h", type="dot", color="#E10600", children=[html.Div(id="h2h-output", style={'marginTop': '20px'})])
        ], className="content-card")
    elif tab == 'tab-replay':
        return html.Div([
            html.H3([html.I(className="fa-solid fa-video", style={'marginRight':'10px'}), "Animated Race Replay"]),
            html.P("Watch the entire grid race around the track layout for a given lap."),
            html.Div([
                dcc.Dropdown(id='rep-year', options=[{'label': str(y), 'value': y} for y in range(2023, 2027)], value=2026, className="custom-dropdown"),
                dcc.Input(id='rep-round', type='number', min=1, max=24, value=1, placeholder="Round number", className="custom-input"),
                dcc.Input(id='rep-lap', type='number', min=1, max=80, value=1, placeholder="Lap number", className="custom-input"),
                html.Button([html.I(className="fa-solid fa-play"), " Load Animation"], id="btn-fetch-rep", n_clicks=0, className="custom-button")
            ], className="controls-row"),
            dcc.Loading(id="loading-rep", type="default", color="#00d2be", children=[html.Div(id="rep-output")])
        ], className="content-card")
    elif tab == 'tab-live':
        return html.Div([
            html.H3([html.I(className="fa-solid fa-satellite-dish", style={'color':'#E10600', 'marginRight':'10px'}), "Live Race Tracker"]),
            html.P("Real-time position updates parsing live telemetry endpoints."),
            dcc.Interval(id='live-interval', interval=2000, n_intervals=0),
            html.Div(id='live-output', style={'marginTop': '20px'})
        ], className="content-card")
    elif tab == 'tab-weather':
        return html.Div([
            html.H3([html.I(className="fa-solid fa-cloud-sun-rain", style={'color':'#00d2be', 'marginRight':'10px'}), "Weather Impact Analysis"]),
            html.P("Analyze how track temperature, air temperature, and humidity evolve throughout the race."),
            html.Div([
                dcc.Dropdown(id='weather-year', options=[{'label': str(y), 'value': y} for y in range(2023, 2027)], value=2026, className="custom-dropdown"),
                dcc.Input(id='weather-round', type='number', min=1, max=24, value=1, placeholder="Round number", className="custom-input"),
                html.Button([html.I(className="fa-solid fa-cloud"), " Load Weather Data"], id="btn-fetch-weather", n_clicks=0, className="custom-button")
            ], className="controls-row"),
            dcc.Loading(id="loading-weather", type="default", color="#00d2be", children=[html.Div(id="weather-output")])
        ], className="content-card")
    elif tab == 'tab-microsector':
        return html.Div([
            html.H3([html.I(className="fa-solid fa-map-location-dot", style={'color':'#E10600', 'marginRight':'10px'}), "Micro-Sector Dominance"]),
            html.P("Compare two drivers to see who was faster in every mini-sector of the track."),
            html.Div([
                dcc.Dropdown(id='ms-year', options=[{'label': str(y), 'value': y} for y in range(2023, 2027)], value=2026, className="custom-dropdown"),
                dcc.Input(id='ms-round', type='number', min=1, max=24, value=1, placeholder="Round number", className="custom-input"),
                dcc.Input(id='ms-driver1', type='text', value='VER', placeholder="Driver 1 (VER)", className="custom-input"),
                dcc.Input(id='ms-driver2', type='text', value='NOR', placeholder="Driver 2 (NOR)", className="custom-input"),
                html.Button([html.I(className="fa-solid fa-draw-polygon"), " Generate Map"], id="btn-fetch-ms", n_clicks=0, className="custom-button")
            ], className="controls-row"),
            dcc.Loading(id="loading-ms", type="default", color="#E10600", children=[html.Div(id="ms-output")])
        ], className="content-card")
    elif tab == 'tab-pitstrategy':
        return html.Div([
            html.H3([html.I(className="fa-solid fa-stopwatch-20", style={'color':'#FFE500', 'marginRight':'10px'}), "Pit Strategy & Tire Degradation"]),
            html.P("Analyze lap times over the race distance, color-coded by tire compound to visualize degradation and stint lengths."),
            html.Div([
                dcc.Dropdown(id='pit-year', options=[{'label': str(y), 'value': y} for y in range(2023, 2027)], value=2026, className="custom-dropdown"),
                dcc.Input(id='pit-round', type='number', min=1, max=24, value=1, placeholder="Round number", className="custom-input"),
                dcc.Input(id='pit-driver', type='text', placeholder="Driver (e.g. VER) or leave blank for all", className="custom-input", style={'minWidth':'250px'}),
                html.Button([html.I(className="fa-solid fa-chart-area"), " Analyze Tires"], id="btn-fetch-pit", n_clicks=0, className="custom-button")
            ], className="controls-row"),
            dcc.Loading(id="loading-pit", type="default", color="#FFE500", children=[html.Div(id="pit-output")])
        ], className="content-card")

@app.callback(
    Output("champ-output", "children"),
    Input("btn-predict-champ", "n_clicks")
)
def update_champ_predictions(n_clicks):
    if n_clicks == 0: return html.P("")
    if not getattr(champ_model, 'loaded', False):
        return html.Div("Championship Model mapping missing. Re-run train_models.py.", style={'color': 'red'})

    try:
        champ_df = pd.read_csv('computed_data/champ_df.csv')
        latest_year = champ_df['year'].max()
        latest_round = champ_df[champ_df['year'] == latest_year]['round'].max()
        latest_standings = champ_df[(champ_df['year'] == latest_year) & (champ_df['round'] == latest_round)].copy()
        
        predicted_points = champ_model.predict_points(latest_standings[['round', 'current_points', 'races_completed', 'wins', 'dnf_rate', 'avg_finish', 'elo']])
        latest_standings['projected_final_points'] = predicted_points
        latest_standings = latest_standings.sort_values(by='projected_final_points', ascending=False).head(15)
        
        fig = px.bar(latest_standings, x='driver_code', y='projected_final_points', 
                     hover_data=['current_points', 'wins', 'elo'],
                     title=f"AI Projected Final Points (Year {latest_year}, Post-Round {latest_round})",
                     labels={'driver_code': 'Driver', 'projected_final_points': 'Projected'},
                     color='projected_final_points', color_continuous_scale='Reds')
        
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), margin=dict(l=40, r=40, t=60, b=40))
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error executing AI projection: {e}", style={'color': 'red'})

@app.callback(
    Output("race-output", "children"),
    Input("btn-predict-race", "n_clicks")
)
def update_race_predictions(n_clicks):
    if n_clicks == 0: return html.P("")
    if not getattr(race_model, 'loaded', False):
        return html.Div("Race Win Model missing. Re-run train_models.py.", style={'color': 'red'})
    
    try:
        race_df = pd.read_csv('computed_data/race_df.csv')
        champ_df = pd.read_csv('computed_data/champ_df.csv')
        latest_year = champ_df['year'].max()
        latest_round = champ_df[champ_df['year'] == latest_year]['round'].max()
        
        history_df = pd.read_csv('computed_data/driver_history_df.csv')
        sched = get_schedule(latest_year)
        next_round = latest_round + 1
        upcoming = sched[sched['RoundNumber'] == next_round]
        if upcoming.empty:
            next_round = latest_round
            upcoming = sched[sched['RoundNumber'] == latest_round]
        location = upcoming.iloc[0]['Location']
        
        # Take drivers from latest round
        latest_race_feat = race_df.drop_duplicates(subset=['driver_code'], keep='last').copy()
        
        for idx, row in latest_race_feat.iterrows():
            dcode = row['driver_code']
            mask = (history_df['DriverCode'] == dcode) & (history_df['Location'] == location) & (history_df['Year'] >= latest_year - 3) & (history_df['Year'] < latest_year)
            past_races = history_df[mask]
            if not past_races.empty:
                latest_race_feat.at[idx, 'track_past_3_years_wins'] = past_races['IsWin'].sum()
                latest_race_feat.at[idx, 'track_past_3_years_points'] = past_races['Points'].sum()
                latest_race_feat.at[idx, 'track_past_3_years_avg_pos'] = past_races['Position'].mean()
            else:
                latest_race_feat.at[idx, 'track_past_3_years_wins'] = 0
                latest_race_feat.at[idx, 'track_past_3_years_points'] = 0
                latest_race_feat.at[idx, 'track_past_3_years_avg_pos'] = 20.0
                
            latest_race_feat.at[idx, 'current_position'] = 0
        
        if 'grid_position' in latest_race_feat.columns:
            latest_race_feat.drop(columns=['grid_position'], inplace=True)
            
        probs = race_model.predict_win_prob(latest_race_feat)
        latest_race_feat['win_prob'] = probs * 100
        
        # Sort and take top 10
        top_contenders = latest_race_feat.sort_values(by='win_prob', ascending=False)
        top_contenders = top_contenders.drop_duplicates(subset=['driver_code']).head(10)
        
        # Generate Text Insight
        top_dr = top_contenders.iloc[0]['driver_code']
        top_pr = top_contenders.iloc[0]['win_prob']
        sec_dr = top_contenders.iloc[1]['driver_code']
        
        insight_text = f"Based on form and 3-year historical track data at {location} (Round {next_round}), " \
                       f"the autonomous neural engine strongly favors {top_dr} with a {top_pr:.1f}% probabilty of outright victory, " \
                       f"facing primary challenge from {sec_dr}."
        
        fig = px.pie(top_contenders, values='win_prob', names='driver_code', hole=0.7,
                     title=f"Race Win Probabilities", color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), annotations=[dict(text='Grid AI', x=0.5, y=0.5, font_size=20, showarrow=False)])
        
        return html.Div([
            html.Div([
                html.I(className="fa-solid fa-bolt", style={'color': '#00d2be', 'marginRight': '10px'}),
                html.Span(insight_text, style={'fontStyle': 'italic', 'color': '#00d2be'})
            ], className="ai-insight-box"),
            dcc.Graph(figure=fig)
        ])
    except Exception as e:
        return html.Div(f"Error computing race probabilities: {e}", style={'color': 'red'})


@app.callback(
    Output("hist-output", "children"),
    Input("btn-fetch-hist", "n_clicks"),
    State("hist-year", "value"), State("hist-round", "value")
)
def fetch_historical_race(n_clicks, year, round_num):
    if n_clicks == 0: return html.P("")
    try:
        race_data = load_race(year, round_num)
        laps = race_data.get('laps')
        if laps is None or laps.empty: return html.Div("No lap data available.", style={'color': 'red'})
        
        results = race_data.get('results', pd.DataFrame())
        res_df = pd.DataFrame()
        if not results.empty:
            display_cols = ['Position', 'Abbreviation', 'TeamName', 'Status', 'Points']
            avail_cols = [c for c in display_cols if c in results.columns]
            res_df = results[avail_cols].copy()
            for c in res_df.columns: res_df[c] = res_df[c].astype(str)
                
        results_html = html.Div([
            dash_table.DataTable(
                data=res_df.to_dict('records') if not res_df.empty else [],
                columns=[{'name': i, 'id': i} for i in res_df.columns],
                style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                style_cell={'backgroundColor': 'rgba(25,25,35,0.7)', 'color': 'white', 'border': '1px solid rgba(255,255,255,0.1)', 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Outfit'},
                style_header={'backgroundColor': 'rgba(225, 6, 0, 0.4)', 'fontWeight': 'bold', 'color': 'white'}
            )
        ])

        # Track Map extraction from position data
        pos_data = race_data.get('pos_data', {})
        track_map_html = html.Div()
        if pos_data:
            driver_code = res_df['Abbreviation'].iloc[0] if not res_df.empty else list(pos_data.keys())[0]
            if driver_code in pos_data:
                driver_pos = pos_data[driver_code]
                map_fig = px.line(driver_pos, x='X', y='Y', title=f"Track Layout (via {driver_code} Telemetry)", color_discrete_sequence=['#00d2be'])
                map_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Outfit", color="#f0f0f5"), hovermode=False)
                map_fig.update_xaxes(visible=False)
                map_fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
                track_map_html = dcc.Graph(figure=map_fig)

        laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
        fig = px.line(laps, x='LapNumber', y='LapTime_s', color='Driver', 
                      title=f"Telemetry Breakdown", labels={'LapNumber': 'Lap', 'LapTime_s': 'Lap Time (s)'})
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), hovermode="x unified",
                          hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=13, font_family="Outfit", font_color="white"))
        
        return html.Div([
            html.Div([
                html.Div([html.H4("Official Standings"), results_html], style={'flex':'1'}),
                html.Div([html.H4("Circuit Overview"), track_map_html], style={'flex':'1'})
            ], style={'display':'flex', 'gap':'30px', 'flexWrap':'wrap'}),
            html.H4("Lap Timings"), 
            dcc.Graph(figure=fig)
        ])
    except Exception as e:
        return html.Div(f"Error loading race: {e}", style={'color': 'red'})

@app.callback(
    Output("tel-output", "children"),
    Output("h2h-output", "children"),
    Input("btn-fetch-tel", "n_clicks"),
    State("tel-year", "value"), State("tel-round", "value"),
    State("tel-driver1", "value"), State("tel-driver2", "value")
)
def fetch_telemetry_data(n_clicks, year, round_num, d1, d2):
    if n_clicks == 0: return html.P(""), html.P("")
    try:
        data = load_telemetry(year, round_num, d1, d2)
        if not data: return html.Div("No telemetry data found.", style={'color': 'red'}), html.P("")
        
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Speed vs Distance", "Track Map (X, Y)"), column_widths=[0.6, 0.4])
        colors = ['#00d2be', '#E10600']
        
        for idx, (driver, dic) in enumerate(data.items()):
            color = colors[idx % len(colors)]
            name_label = f"{driver} ({dic['LapTime']})"
            fig.add_trace(go.Scatter(x=dic['Distance'], y=dic['Speed'], name=name_label, mode='lines', line=dict(color=color, width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=dic['X'], y=dic['Y'], name=f"{driver} Map", mode='lines', line=dict(color=color, width=3), showlegend=False), row=1, col=2)
            
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), hovermode="x unified", height=500,
                          hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=13, font_family="Outfit", font_color="white"))
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        fig.update_xaxes(title_text="Distance (m)", row=1, col=1)
        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        tel_graph = dcc.Graph(figure=fig)

        # Build H2H scorecard if we have two drivers
        h2h_card = html.P("")
        if len(data) == 2:
            drivers = list(data.keys())
            d1_code, d2_code = drivers[0], drivers[1]
            sc = analyze_head_to_head(d1_code, data[d1_code], d2_code, data[d2_code])
            h2h_card = _build_h2h_card(sc, colors[0], colors[1])

        return tel_graph, h2h_card
    except Exception as e:
        return html.Div(f"Error fetching telemetry: {e}", style={'color': 'red'}), html.P("")


def _build_h2h_card(sc, color1, color2):
    """Renders the H2H scorecard as Dash HTML."""
    d1 = sc['driver1']
    d2 = sc['driver2']
    winner = sc['overall_winner']
    winner_color = color1 if winner == d1 else (color2 if winner == d2 else '#ffffff')
    winner_icon = '🏆'

    def cell_style(winner_code, this_code, c):
        if winner_code == this_code:
            return {'color': c, 'fontWeight': 'bold', 'fontSize': '1em'}
        return {'color': 'rgba(255,255,255,0.5)', 'fontSize': '0.95em'}

    sector_rows = []
    for seg in sc.get('sectors', []):
        seg_winner = seg['winner']
        sector_rows.append(
            html.Tr([
                html.Td(seg['d1_speed'], style=cell_style(seg_winner, d1, color1)),
                html.Td(seg['label'], style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.6)', 'fontSize': '0.85em', 'padding': '6px 16px'}),
                html.Td(seg['d2_speed'], style={**cell_style(seg_winner, d2, color2), 'textAlign': 'right'}),
            ], style={'borderBottom': '1px solid rgba(255,255,255,0.06)'})
        )

    metric_rows = []
    for m in sc.get('metrics', []):
        mw = m['winner']
        metric_rows.append(
            html.Tr([
                html.Td(m['d1_val'], style=cell_style(mw, d1, color1)),
                html.Td(m['label'], style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.6)', 'fontSize': '0.85em', 'padding': '6px 16px'}),
                html.Td(m['d2_val'], style={**cell_style(mw, d2, color2), 'textAlign': 'right'}),
            ], style={'borderBottom': '1px solid rgba(255,255,255,0.06)'})
        )

    return html.Div([
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
        html.H4([
            html.I(className="fa-solid fa-ranking-star", style={'color': '#E10600', 'marginRight': '10px'}),
            "AI Head-to-Head Scorecard"
        ], style={'marginBottom': '16px'}),
        # Driver header
        html.Div([
            html.Span(d1, style={'color': color1, 'fontWeight': '800', 'fontSize': '1.4em', 'letterSpacing': '2px'}),
            html.Span(" vs ", style={'color': 'rgba(255,255,255,0.4)', 'margin': '0 16px', 'fontSize': '1.1em'}),
            html.Span(d2, style={'color': color2, 'fontWeight': '800', 'fontSize': '1.4em', 'letterSpacing': '2px'}),
        ], style={'textAlign': 'center', 'marginBottom': '8px'}),
        html.Div([
            html.Span(f"{sc['d1_laptime']}", style={'color': color1, 'fontSize': '0.9em', 'marginRight': '30px'}),
            html.Span("Fastest Lap", style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '0.85em'}),
            html.Span(f"{sc['d2_laptime']}", style={'color': color2, 'fontSize': '0.9em', 'marginLeft': '30px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        # Sector table
        html.H5("Sector-by-Sector (Avg Speed)", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '0.8em', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '8px'}),
        html.Table(sector_rows, style={'width': '100%', 'marginBottom': '20px'}),

        # Metrics table
        html.H5("Performance Metrics", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '0.8em', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '8px'}),
        html.Table(metric_rows, style={'width': '100%', 'marginBottom': '24px'}),

        # Overall verdict
        html.Div([
            html.Span(f"{winner_icon} ", style={'fontSize': '1.6em'}),
            html.Span(winner if winner != 'Tied' else 'DEAD HEAT', style={'color': winner_color, 'fontWeight': '900', 'fontSize': '1.3em', 'letterSpacing': '3px'}),
            html.Br(),
            html.Span(sc['margin_text'], style={'color': 'rgba(255,255,255,0.55)', 'fontSize': '0.9em'})
        ], style={
            'textAlign': 'center', 'padding': '20px',
            'background': f'linear-gradient(135deg, rgba(0,0,0,0.3), rgba(0,0,0,0.1))',
            'border': f'1px solid {winner_color}44',
            'borderRadius': '12px'
        })
    ], style={
        'background': 'rgba(255,255,255,0.03)',
        'border': '1px solid rgba(255,255,255,0.08)',
        'borderRadius': '14px',
        'padding': '24px',
        'marginTop': '20px'
    })


@app.callback(
    Output("rep-output", "children"),
    Input("btn-fetch-rep", "n_clicks"),
    State("rep-year", "value"), State("rep-round", "value"), State("rep-lap", "value")
)
def fetch_replay_data(n_clicks, year, round_num, lap_num):
    if n_clicks == 0: return html.P("")
    try:
        df, track_df = load_lap_replay(year, round_num, lap_num)
        if df is None or df.empty: return html.Div("No telemetry found for this session/lap.", style={'color': 'red'})
        
        x_min, x_max = df['X'].min() - 500, df['X'].max() + 500
        y_min, y_max = df['Y'].min() - 500, df['Y'].max() + 500
        
        fig = px.scatter(df, x="X", y="Y", animation_frame="Frame", animation_group="Driver",
                         color="Driver", hover_name="Driver", text="Driver", 
                         range_x=[x_min, x_max], range_y=[y_min, y_max], title=f"Animated Replay - Lap {lap_num}")
                         
        if not track_df.empty:
            track_trace = dict(type='scatter', x=track_df['X'], y=track_df['Y'], mode='lines', 
                               line=dict(color='rgba(255,255,255,0.15)', width=4), hoverinfo='skip', showlegend=False)
            fig.add_trace(go.Scatter(**track_trace))
            for frame in fig.frames:
                frame.data = frame.data + (go.Scatter(**track_trace),)
        
        fig.update_traces(marker=dict(size=14, line=dict(width=2, color='rgba(255,255,255,0.7)')), textfont=dict(color='white', size=11), textposition='top center', selector=dict(mode='markers+text'))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), height=750, xaxis=dict(visible=False), yaxis=dict(visible=False),
                          updatemenus=[dict(type="buttons", buttons=[dict(label="► Play", method="animate", args=[None, {"frame": {"duration": 150, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}])], showactive=False, x=0.05, y=0, font=dict(color="black"))])
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error fetching replay: {e}", style={'color': 'red'})


@app.callback(
    Output("commentary-output", "children"),
    Input("btn-commentary", "n_clicks"),
    State("hist-year", "value"),
    State("hist-round", "value"),
    State("gemini-api-key", "value"),
    prevent_initial_call=True
)
def generate_race_commentary(n_clicks, year, round_num, api_key):
    if not n_clicks: return html.P("")
    try:
        race_data = load_race(year, round_num)
        laps = race_data.get('laps', pd.DataFrame())
        results = race_data.get('results', pd.DataFrame())
        weather = race_data.get('weather', pd.DataFrame())

        commentary_text = generate_commentary(results, laps, weather, year, round_num, api_key or '')

        # Render as styled card
        paragraphs = [p.strip() for p in commentary_text.split('\n') if p.strip()]
        para_elements = [html.P(p, style={'lineHeight': '1.85', 'marginBottom': '16px', 'color': 'rgba(255,255,255,0.88)'}) for p in paragraphs]

        return html.Div([
            html.Div([
                html.I(className="fa-solid fa-microphone-lines", style={'color': '#00d2be', 'marginRight': '10px', 'fontSize': '1.1em'}),
                html.Span(f"AI Commentary — {year} Round {round_num}", style={'color': '#00d2be', 'fontWeight': '600', 'fontSize': '1em'})
            ], style={'marginBottom': '16px', 'paddingBottom': '12px', 'borderBottom': '1px solid rgba(0,210,190,0.2)'}),
            *para_elements
        ], style={
            'background': 'linear-gradient(135deg, rgba(0,210,190,0.06), rgba(0,0,0,0.2))',
            'border': '1px solid rgba(0,210,190,0.2)',
            'borderRadius': '12px',
            'padding': '24px',
            'fontFamily': 'Outfit, sans-serif',
            'fontSize': '0.97em'
        })
    except Exception as e:
        return html.Div(f"Commentary error: {e}", style={'color': 'red'})


@app.callback(
    Output("live-output", "children"),
    Input("live-interval", "n_intervals")
)
def update_live_tracker(n):
    snapshot = get_latest_snapshot()
    if not snapshot or not snapshot.get('drivers'):
        return html.Div("Awaiting active session uplink (Ensure data/live_timing.py is active)...", style={'color': 'orange'})
        
    drivers = snapshot['drivers']
    df = pd.DataFrame([{'Driver': k, 'Position': v.get('Position', 20)} for k, v in drivers.items()]).dropna(subset=['Position'])
    if df.empty: return html.Div("No driver position data yet.")
    
    df['Position'] = pd.to_numeric(df['Position'])
    df = df.sort_values('Position')
    
    fig = px.bar(df, x='Position', y='Driver', orientation='h', title="Live Grid State", color='Position', color_continuous_scale='Reds_r')
    fig.update_layout(yaxis={'categoryorder':'total descending'}, xaxis={'autorange': 'reversed', 'title': 'Position'},
                      template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Outfit", color="#f0f0f5"), showlegend=False)
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run(debug=True)

@app.callback(
    Output("weather-output", "children"),
    Input("btn-fetch-weather", "n_clicks"),
    State("weather-year", "value"), State("weather-round", "value")
)
def fetch_weather_data(n_clicks, year, round_num):
    if n_clicks == 0: return html.P("")
    try:
        race_data = load_race(year, round_num)
        weather = race_data.get('weather')
        if weather is None or weather.empty: 
            return html.Div("No weather data available for this race.", style={'color': 'red'})
            
        weather['Time_m'] = weather['Time'].dt.total_seconds() / 60.0
        
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Temperature (°C)", "Humidity & Wind"))
        
        fig.add_trace(go.Scatter(x=weather['Time_m'], y=weather['TrackTemp'], name="Track Temp", line=dict(color='#E10600', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=weather['Time_m'], y=weather['AirTemp'], name="Air Temp", line=dict(color='#00d2be', width=2)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=weather['Time_m'], y=weather['Humidity'], name="Humidity (%)", line=dict(color='#3498db', width=2)), row=2, col=1)
        if 'WindSpeed' in weather.columns:
            fig.add_trace(go.Scatter(x=weather['Time_m'], y=weather['WindSpeed'], name="Wind Speed (m/s)", line=dict(color='#f39c12', width=2, dash='dot')), row=2, col=1)

        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), hovermode="x unified", height=600,
                          hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=13, font_family="Outfit", font_color="white"))
        fig.update_xaxes(title_text="Session Time (Minutes)", row=2, col=1)
        
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error fetching weather: {e}", style={'color': 'red'})

@app.callback(
    Output("ms-output", "children"),
    Input("btn-fetch-ms", "n_clicks"),
    State("ms-year", "value"), State("ms-round", "value"),
    State("ms-driver1", "value"), State("ms-driver2", "value")
)
def fetch_microsector_map(n_clicks, year, round_num, d1, d2):
    if n_clicks == 0: return html.P("")
    if not d1 or not d2: return html.Div("Please enter two driver codes (e.g. VER, NOR).", style={'color': 'red'})
    try:
        ms_df = load_microsectors(year, round_num, d1.upper(), d2.upper(), num_sectors=30)
        if ms_df is None or ms_df.empty:
            return html.Div("Could not generate micro-sector map for these drivers/session.", style={'color': 'red'})
            
        colors = {d1.upper(): '#00d2be', d2.upper(): '#E10600'}
        fig = px.scatter(ms_df, x="X", y="Y", color="Fastest_Driver", color_discrete_map=colors, 
                         title=f"Micro-Sector Dominance ({d1.upper()} vs {d2.upper()})",
                         labels={"Fastest_Driver": "Fastest in Sector"})
                         
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), height=700, 
                          xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
                          
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error fetching microsectors: {e}", style={'color': 'red'})

@app.callback(
    Output("pit-output", "children"),
    Input("btn-fetch-pit", "n_clicks"),
    State("pit-year", "value"), State("pit-round", "value"), State("pit-driver", "value")
)
def fetch_pit_strategy(n_clicks, year, round_num, driver):
    if n_clicks == 0: return html.P("")
    try:
        race_data = load_race(year, round_num)
        laps = race_data.get('laps')
        if laps is None or laps.empty: return html.Div("No lap data available.", style={'color': 'red'})
        
        # Remove outlier laps (VSC/SC/In/Out)
        laps = laps.pick_quicklaps(1.5)
        laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
        
        if driver and str(driver).strip().upper() != '':
            drv_code = str(driver).strip().upper()
            laps = laps[laps['Driver'] == drv_code]
            if laps.empty:
                return html.Div(f"No valid racing lap data found for driver {drv_code}.", style={'color': 'red'})
            title = f"Tire Degradation - {drv_code}"
            hover_data = ["Compound", "Stint"]
        else:
            title = "Overall Race Tire Degradation (All Drivers)"
            hover_data = ["Driver", "Compound", "Stint"]
            
        compound_colors = {
            'SOFT': '#FF3333',
            'MEDIUM': '#FFE500',
            'HARD': '#FFFFFF',
            'INTERMEDIATE': '#39B54A',
            'WET': '#00AEEF',
            'TEST-UNKNOWN': '#808080'
        }
        
        fig = px.scatter(laps, x="LapNumber", y="LapTime_s", color="Compound", 
                         color_discrete_map=compound_colors,
                         hover_data=hover_data,
                         opacity=0.8, title=title)
        
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(0,0,0,0.6)')))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family="Outfit", color="#f0f0f5"), height=650,
                          yaxis=dict(title="Lap Time (s)", autorange="reversed"), 
                          xaxis=dict(title="Lap Number"))
                          
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error fetching strategy data: {e}", style={'color': 'red'})
