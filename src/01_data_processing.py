import re
import sys
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from google.transit import gtfs_realtime_pb2

root = Path.cwd().parent
sys.path.append(str(root))
plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"}) 
from utils.time import (
    timestamp_to_hhmmss,
    timestamp_to_yyyymmdd_hhmmss,
    seconds_to_hhmmss,
    hhmmss_to_seconds,
    timestamp_to_seconds,
)



def calculate_delays(record_path:Path, delay_path:Path, available_trip_ids:set, stop_times:pd.DataFrame)->None:


    realtime = pd.read_csv(record_path, low_memory=False)
    realtime = realtime[['route_id','trip_id','current_stop_sequence','timestamp']]
    realtime['time'] = realtime['timestamp'].apply(timestamp_to_seconds)
    incoming_trip_ids = set(realtime['trip_id'].unique())

    # Determine known trips
    unknown_trip_ids = incoming_trip_ids - available_trip_ids
    known_trip_ids = incoming_trip_ids - unknown_trip_ids

    # Observe only the known trips
    realtime = realtime[realtime['trip_id'].isin(known_trip_ids)]
    static = stop_times[stop_times['trip_id'].isin(known_trip_ids)].copy()
    static_grouped = static.groupby('trip_id')

    rows = []
    for route_id, group1 in tqdm(realtime.groupby('route_id'), desc="Routes"):
        for trip_id, realtime_trip in tqdm(group1.groupby('trip_id'), desc="Trips",leave=False):
            
            static_trip = static_grouped.get_group(trip_id)

            first_stop = static_trip.loc[static_trip['stop_sequence'].idxmin()]
            last_stop = static_trip.loc[static_trip['stop_sequence'].idxmax()]
            stop_seq_sp, time_sch_sp = first_stop['stop_sequence'], first_stop['arrival_time']
            stop_seq_ep, time_sch_ep = last_stop['stop_sequence'], last_stop['arrival_time']
            
            
            tmp = realtime_trip[realtime_trip['current_stop_sequence'].isin([stop_seq_ep])]
            msg = f'{record_path.name} - '

            if tmp.empty:
                msg += f"A {trip_id} útra nincs meg az infó, hogy mikor ért a végállomásra"
                pass
            else:
                # The time which is closest
                time_ep = tmp['time'].iloc[(tmp['time'] - time_sch_ep).abs().argmin()] # time_ep = tmp['time'].min()
                delay = int(time_ep - time_sch_ep)
                if delay < -23.5*3600: 
                    # close to midnight
                    delay += 24 * 3600
                
                if abs(delay) > 900:
                    # If we have more than 15min delay (assumed to be an outlier)
                    continue

                msg += f"A {trip_id} út "
                msg += f'{seconds_to_hhmmss(delay)} késéssel' if delay > 0 else f'{seconds_to_hhmmss(-delay)} gyorsabban'
                msg += ' lett teljesítve'

                rows.append([route_id, trip_id, delay, time_sch_sp])
            
                # print(msg)

    pd.DataFrame(
        rows, columns=['route_id','trip_id','delay', 'trip_start']
    ).to_csv(delay_path, index=False)    
    


if __name__ == '__main__':

    root = Path('/mnt/c/Users/rdsup/Desktop/vitmma19-pw-delay-detection')
    
    static_path  = root / 'data' / 'static_gtfs'
    dynamic_path = root / 'data' / 'dynamic_gtfs'
    
    
    trips = pd.read_csv(static_path / 'trips.txt', sep=',', low_memory=False)
    available_trip_ids = set(trips['trip_id'].unique())

    
    stop_times = pd.read_csv(static_path / 'stop_times.txt', sep=',', low_memory=False)
    stop_times = stop_times[['trip_id','stop_sequence','arrival_time']].copy()
    stop_times['arrival_time'] = stop_times['arrival_time'].apply(hhmmss_to_seconds)

    
    for record_path in (dynamic_path / 'records').iterdir():
        delay_path = Path(str(record_path).replace('records','delays'))

        if not delay_path.exists(): 
            print(f'Calculating delays of {record_path.name}...')
            calculate_delays(
                record_path,
                delay_path,
                available_trip_ids=available_trip_ids,
                stop_times=stop_times,
            )


