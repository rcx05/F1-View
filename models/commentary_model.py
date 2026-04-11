"""
AI Race Commentary Generator using Google Gemini API.
Produces engaging, F1-expert-level commentary from race lap/results data.
"""
import os
import pandas as pd


def _format_race_data_for_prompt(results_df: pd.DataFrame, laps_df: pd.DataFrame, weather_df: pd.DataFrame, year: int, round_num: int) -> str:
    """Builds a compact text summary of race data to feed into the LLM prompt."""
    lines = [f"F1 Grand Prix - Year: {year}, Round: {round_num}"]

    # Podium
    if not results_df.empty and 'Position' in results_df.columns:
        sorted_res = results_df.copy()
        sorted_res['Position'] = pd.to_numeric(sorted_res['Position'], errors='coerce')
        sorted_res = sorted_res.dropna(subset=['Position']).sort_values('Position')
        podium = sorted_res.head(5)
        lines.append("\nTop 5 Finishers:")
        for _, row in podium.iterrows():
            pos = int(row['Position'])
            abbr = row.get('Abbreviation', '???')
            team = row.get('TeamName', 'Unknown')
            pts = row.get('Points', 0)
            status = row.get('Status', 'Finished')
            lines.append(f"  P{pos}: {abbr} ({team}) | {pts} pts | Status: {status}")

        # DNFs
        dnfs = sorted_res[~sorted_res['Status'].str.contains('Finished|Lap', na=False, regex=True)]
        if not dnfs.empty:
            lines.append(f"\nDNFs: {', '.join(dnfs['Abbreviation'].tolist())}")

    # Fastest lap
    if not laps_df.empty and 'LapTime' in laps_df.columns:
        laps_df['LapTime_s'] = laps_df['LapTime'].dt.total_seconds()
        valid_laps = laps_df.dropna(subset=['LapTime_s'])
        if not valid_laps.empty:
            fastest_row = valid_laps.loc[valid_laps['LapTime_s'].idxmin()]
            lines.append(f"\nFastest Lap: {fastest_row.get('Driver','?')} — {fastest_row['LapTime_s']:.3f}s on lap {int(fastest_row.get('LapNumber', 0))}")

    # Lap count
    if not laps_df.empty and 'LapNumber' in laps_df.columns:
        lines.append(f"Total Race Laps: {int(laps_df['LapNumber'].max())}")

    # Weather summary
    if not weather_df.empty:
        cols = {}
        if 'AirTemp' in weather_df.columns:
            cols['Air Temp'] = f"{weather_df['AirTemp'].mean():.1f}°C"
        if 'TrackTemp' in weather_df.columns:
            cols['Track Temp'] = f"{weather_df['TrackTemp'].mean():.1f}°C"
        if 'Rainfall' in weather_df.columns:
            rain = weather_df['Rainfall'].any()
            cols['Rain'] = 'Yes' if rain else 'No'
        if 'WindSpeed' in weather_df.columns:
            cols['Wind Speed'] = f"{weather_df['WindSpeed'].mean():.1f} m/s"
        if cols:
            lines.append("\nWeather: " + " | ".join(f"{k}: {v}" for k, v in cols.items()))

    return "\n".join(lines)


def generate_commentary(results_df: pd.DataFrame, laps_df: pd.DataFrame, weather_df: pd.DataFrame,
                        year: int, round_num: int, api_key: str) -> str:
    """
    Calls Google Gemini to generate race commentary.
    Returns markdown-formatted commentary string.
    """
    if not api_key or not api_key.strip():
        return "⚠️ No Gemini API key provided. Enter your key above to enable AI commentary."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel('gemini-2.0-flash')
    except ImportError:
        return "❌ `google-generativeai` not installed. Run: `pip install google-generativeai`"
    except Exception as e:
        return f"❌ Failed to initialize Gemini: {e}"

    race_summary = _format_race_data_for_prompt(results_df, laps_df, weather_df, year, round_num)

    prompt = f"""You are an elite Formula 1 race analyst and commentator with decades of experience.
Using the following race data, write a vivid, expert-level race review (3-4 paragraphs).

Include:
- Narrative of how the race unfolded
- Key performances and battles
- Strategic moments or unusual events (DNFs, weather)
- Insight on what the result means for the championship

Keep the tone passionate, precise, and insightful — like a Sky Sports commentary piece.
Use driver surnames, not codes, where possible.

Race Data:
{race_summary}

Write the commentary now:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API error: {e}"


def analyze_head_to_head(driver1: str, d1_data: dict, driver2: str, d2_data: dict) -> dict:
    """
    Pure analytical scoring of two drivers' fastest lap telemetry.
    Divides the lap into thirds and compares multiple metrics.
    Returns a structured result dict.
    """
    import numpy as np

    results = {
        'driver1': driver1, 'driver2': driver2,
        'sectors': [],    # per-sector winner
        'metrics': [],    # top-level metric comparisons
        'overall_winner': None,
        'margin_text': '',
        'd1_score': 0, 'd2_score': 0
    }

    def get_arrays(data, key):
        return list(data.get(key, []))

    d1_speed = get_arrays(d1_data, 'Speed')
    d2_speed = get_arrays(d2_data, 'Speed')
    d1_dist  = get_arrays(d1_data, 'Distance')
    d2_dist  = get_arrays(d2_data, 'Distance')
    d1_brake = get_arrays(d1_data, 'Brake')
    d2_brake = get_arrays(d2_data, 'Brake')
    d1_thrott = get_arrays(d1_data, 'Throttle')
    d2_thrott = get_arrays(d2_data, 'Throttle')
    d1_laptime = d1_data.get('LapTime', 'N/A')
    d2_laptime = d2_data.get('LapTime', 'N/A')

    d1_score = 0
    d2_score = 0

    # --- Top-level metrics ---
    def compare(label, v1, v2, unit, higher_better=True, fmt=".1f"):
        nonlocal d1_score, d2_score
        if v1 is None or v2 is None:
            return
        winner = driver1 if (v1 > v2) == higher_better else driver2
        if abs(v1 - v2) < 1e-6:
            winner = 'Tied'
        else:
            if winner == driver1: d1_score += 1
            else: d2_score += 1
        results['metrics'].append({
            'label': label,
            'd1_val': f"{v1:{fmt}}{unit}",
            'd2_val': f"{v2:{fmt}}{unit}",
            'winner': winner
        })

    if d1_speed and d2_speed:
        compare("Top Speed", max(d1_speed), max(d2_speed), " km/h")
        compare("Avg Speed", np.mean(d1_speed), np.mean(d2_speed), " km/h")

    if d1_brake and d2_brake:
        d1_brake_pct = sum(1 for b in d1_brake if b) / len(d1_brake) * 100
        d2_brake_pct = sum(1 for b in d2_brake if b) / len(d2_brake) * 100
        compare("Braking Zone %", d1_brake_pct, d2_brake_pct, "%", higher_better=False)

    if d1_thrott and d2_thrott:
        d1_full_throttle = sum(1 for t in d1_thrott if t >= 95) / len(d1_thrott) * 100
        d2_full_throttle = sum(1 for t in d2_thrott if t >= 95) / len(d2_thrott) * 100
        compare("Full-Throttle %", d1_full_throttle, d2_full_throttle, "%")

    # --- Sector analysis (divide lap into 3 equal thirds by distance) ---
    if d1_dist and d2_dist and d1_speed and d2_speed:
        d1_max_dist = max(d1_dist)
        d2_max_dist = max(d2_dist)

        for s_idx in range(3):
            s_lo = s_idx / 3
            s_hi = (s_idx + 1) / 3
            label = f"Sector {s_idx+1}"

            def sector_avg_speed(dist, speed, lo_frac, hi_frac, total_dist):
                lo = total_dist * lo_frac
                hi = total_dist * hi_frac
                pts = [sp for d, sp in zip(dist, speed) if lo <= d <= hi]
                return np.mean(pts) if pts else None

            v1 = sector_avg_speed(d1_dist, d1_speed, s_lo, s_hi, d1_max_dist)
            v2 = sector_avg_speed(d2_dist, d2_speed, s_lo, s_hi, d2_max_dist)

            winner = 'N/A'
            if v1 is not None and v2 is not None:
                if v1 > v2 + 0.5:
                    winner = driver1
                    d1_score += 1
                elif v2 > v1 + 0.5:
                    winner = driver2
                    d2_score += 1
                else:
                    winner = 'Tied'

            results['sectors'].append({
                'label': label,
                'd1_speed': f"{v1:.1f} km/h" if v1 else 'N/A',
                'd2_speed': f"{v2:.1f} km/h" if v2 else 'N/A',
                'winner': winner
            })

    results['d1_score'] = d1_score
    results['d2_score'] = d2_score
    total = d1_score + d2_score
    if total == 0:
        results['overall_winner'] = 'Tied'
        results['margin_text'] = "Not enough data to score."
    elif d1_score > d2_score:
        results['overall_winner'] = driver1
        results['margin_text'] = f"{driver1} dominated {d1_score}/{total} categories vs {driver2}"
    elif d2_score > d1_score:
        results['overall_winner'] = driver2
        results['margin_text'] = f"{driver2} dominated {d2_score}/{total} categories vs {driver1}"
    else:
        results['overall_winner'] = 'Tied'
        results['margin_text'] = f"Dead heat — {d1_score}/{total} each"

    results['d1_laptime'] = d1_laptime
    results['d2_laptime'] = d2_laptime

    return results
