import os
import json
import asyncio
from fastf1.livetiming.client import SignalRClient

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
LIVE_STREAM_FILE = os.path.join(CACHE_DIR, 'live_stream.txt')

def start_live_client():
    """Start the background SignalR client to listen to FastF1 live timing."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    async def run_client():
        try:
            client = SignalRClient(LIVE_STREAM_FILE, append=False)
            await client.start()
        except Exception as e:
            print(f"Live timing connection error: {e}")
            
    asyncio.run(run_client())

def get_latest_snapshot():
    """Parse the live_stream.txt for the latest data."""
    if not os.path.exists(LIVE_STREAM_FILE):
        return None
        
    drivers = {}
    session_info = {'track_temp': 25.0, 'safety_car_active': 0, 'track_status': '1'}
    
    try:
        with open(LIVE_STREAM_FILE, 'r') as f:
            lines = f.readlines()
            
        for line in lines[-50:]:  # parse last 50 lines to find latest state
            try:
                # SignalR lines might have prefixes, try to parse JSON
                # usually it's plain json or prefixed
                json_str = line.strip()
                if json_str.startswith('\x1e'): json_str = json_str[1:]
                
                msg = json.loads(json_str)
                if 'M' in msg:
                    for m in msg['M']:
                        if m['H'] == 'Streaming' and m['M'] == 'feed':
                            data = m['A'][0]
                            if 'TimingData' in data:
                                t_data = data['TimingData']
                                if 'Lines' in t_data:
                                    for num, info in t_data['Lines'].items():
                                        if num not in drivers: drivers[num] = {}
                                        if 'Position' in info: drivers[num]['Position'] = int(info['Position'])
            except Exception:
                pass
    except Exception as e:
        print(f"Error reading stream: {e}")

    return {
        'drivers': drivers,
        'session_info': session_info
    }

if __name__ == '__main__':
    print(f"Starting Live Timing Stream to {LIVE_STREAM_FILE}...")
    start_live_client()
