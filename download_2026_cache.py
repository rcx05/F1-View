import datetime
import fastf1
from data.fastf1_loader import load_lap_replay

def main():
    print("Fetching 2026 season schedule...")
    sched = fastf1.get_event_schedule(2026)
    
    # Filter for valid rounds (pre-season testing is usually 0)
    sched = sched[sched['RoundNumber'] > 0]
    
    now = datetime.datetime.now()
    # FastF1 EventDate is timezone aware sometimes, let's assure type compatibility
    past_events = []
    for _, row in sched.iterrows():
        try:
            event_date = pd.to_datetime(row['EventDate'])
            if event_date.timestamp() < now.timestamp():
                past_events.append(row['RoundNumber'])
        except:
            if hasattr(row['EventDate'], 'timestamp') and row['EventDate'].timestamp() < now.timestamp():
                past_events.append(row['RoundNumber'])
    
    if not past_events:
        print("No past events found in 2026 yet (or date filtering failed). Let's try downloading the first few rounds unconditionally.")
        past_events = [1, 2, 3, 4]

    for round_num in past_events:
        print(f"Caching replay data for 2026 round {round_num}...")
        try:
            # We fetch lap 1 as a representation to trigger fastf1 caching the session
            # Actually downloading the whole race's telemetry would be triggered by this
            res = load_lap_replay(2026, round_num, lap_number=1)
            if res is not None:
                print(f"Successfully loaded data for 2026 round {round_num}")
        except Exception as e:
            print(f"Could not load data for 2026 round {round_num}: {e}")

if __name__ == '__main__':
    main()
