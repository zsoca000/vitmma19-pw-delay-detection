import re
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from google.transit import gtfs_realtime_pb2

from src.utils.time import (
    seconds_to_hhmmss,
    hhmmss_to_seconds,
    timestamp_to_seconds,
    name_to_day,
)


# -------------- STATIC GTFS DATA --------------

def static_sort_buses(static_path: Path):
    
    save_path = static_path.parent / f"{static_path.name}_bus"
    save_path.mkdir(exist_ok=True)

    # Load files
    print('Loading static GTFS data...')
    routes = pd.read_csv(static_path / 'routes.txt', low_memory=False)
    trips = pd.read_csv(static_path / 'trips.txt', low_memory=False)
    stops = pd.read_csv(static_path / 'stops.txt', low_memory=False)
    shapes = pd.read_csv(static_path / 'shapes.txt', low_memory=False)
    stop_times = pd.read_csv(static_path / 'stop_times.txt', low_memory=False)

    stop_times['arrival_time'] = stop_times['arrival_time'].apply(hhmmss_to_seconds) % (60*60*24)
    stop_times['departure_time'] = stop_times['departure_time'].apply(hhmmss_to_seconds) % (60*60*24)

    # IDs to filter buses
    bus_route_ids = routes[routes['route_type'] == 3]['route_id'].unique()
    bus_trip_ids = trips[trips['route_id'].isin(bus_route_ids)]['trip_id'].unique()
    bus_shape_ids = trips[trips['route_id'].isin(bus_route_ids)]['shape_id'].unique()
    bus_stop_ids = stop_times[stop_times['trip_id'].isin(bus_trip_ids)]['stop_id'].unique()

    # Original lengths
    original_counts = {
        'routes': len(routes),
        'trips': len(trips),
        'shapes': len(shapes),
        'stops': len(stops),
        'stop_times': len(stop_times)
    }

    # Filtering
    print('Filtering static GTFS data...')
    routes = routes[routes['route_id'].isin(bus_route_ids)]
    trips = trips[trips['trip_id'].isin(bus_trip_ids)]
    shapes = shapes[shapes['shape_id'].isin(bus_shape_ids)]
    stops = stops[stops['stop_id'].isin(bus_stop_ids)]
    stop_times = stop_times[stop_times['trip_id'].isin(bus_trip_ids)]

    # Filtered lengths
    filtered_counts = {
        'routes': len(routes),
        'trips': len(trips),
        'shapes': len(shapes),
        'stops': len(stops),
        'stop_times': len(stop_times)
    }

    # Prepare table text
    lines = []
    header = f"{'Table':<12} {'Before':>8} {'After':>8} {'Diff':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for table in original_counts:
        before = original_counts[table]
        after = filtered_counts[table]
        diff = before - after
        lines.append(f"{table:<12} {before:>8} {after:>8} {diff:>8}")
    
    table_text = "\n".join(lines)
    print(table_text)

    # Save table to txt
    with open(save_path / 'bus_filtering_summary.txt', 'w') as f:
        f.write(table_text)

    # Save filtered files
    print('Saving filtered static GTFS data...')
    routes.to_csv(save_path / 'routes.txt', index=False)
    trips.to_csv(save_path / 'trips.txt', index=False)
    stops.to_csv(save_path / 'stops.txt', index=False)
    shapes.to_csv(save_path / 'shapes.txt', index=False)
    stop_times.to_csv(save_path / 'stop_times.txt', index=False)

    # === Save the bus IDs separately ===
    # Save trip_ids
    with open(save_path / 'bus_trip_ids.txt', 'w') as f:
        for tid in bus_trip_ids:
            f.write(f"{tid}\n")

    # Optional: save route_ids
    with open(save_path / 'bus_route_ids.txt', 'w') as f:
        for rid in bus_route_ids:
            f.write(f"{rid}\n")

def get_bus_trip_ids(static_path: Path):
    with open(static_path / 'bus_trip_ids.txt', 'r') as f:
        return  [line.strip() for line in f]

def get_bus_route_ids(static_path: Path):
    with open(static_path / 'bus_route_ids.txt', 'r') as f:
        return  [line.strip() for line in f]




# -------------- DYNAMIC GTFS DATA --------------

def organize_dynamic_per_days(
        trials_path:Path, save_path:Path, 
        bus_route_ids:list[str], bus_trip_ids:list[str]
    ):

    bus_trip_ids = set(bus_trip_ids)
    bus_route_ids = set(bus_route_ids)

    save_path.mkdir(exist_ok=True)
    
    rows = []
    prev_day_str = None
    pattern = re.compile(r"feed_(\d{8})_(\d{6})\.pb")

    for folder in trials_path.iterdir():
        if not folder.is_dir():
            continue
        
        for file in tqdm(folder.iterdir(), desc=f'{folder.name}'):
            match = pattern.match(file.name)
            if not match or file.suffix != '.pb':
                continue
            
            # Which day?
            date_str, time_str = match.groups()
            dt = pd.to_datetime(date_str + time_str,format="%Y%m%d%H%M%S")
            day_str = dt.date().strftime("%Y%m%d")
            if (save_path / (day_str + ".csv")).exists():
                continue

            # Save the amount of rows, if the day changed
            if prev_day_str and prev_day_str != day_str and rows:
                csv_out = save_path / (prev_day_str + ".csv")
                df = pd.DataFrame(rows)
                df.to_csv(csv_out, mode="w", header=not csv_out.exists(), index=False)
                rows = []
            prev_day_str = day_str

            # Load GTFS data
            feed = gtfs_realtime_pb2.FeedMessage()
            with open(file, "rb") as f:
                feed.ParseFromString(f.read())

            # Read and append GTFS data
            for vehicle in feed.entity:
                if vehicle.vehicle.trip.trip_id != '':
                    
                    route_id = vehicle.vehicle.trip.route_id
                    trip_id = vehicle.vehicle.trip.trip_id

                    known_trip = trip_id in bus_trip_ids
                    known_route = route_id in bus_route_ids

                    # Append just the bus routes
                    if known_trip and known_route:
                        rows.append({
                            'id' : vehicle.id,
                            'trip_id' : trip_id,
                            'scheduled' : vehicle.vehicle.trip.schedule_relationship,
                            'route_id' : route_id,
                            'latitude' : vehicle.vehicle.position.latitude,
                            'longitude' : vehicle.vehicle.position.longitude,
                            'bearing' : vehicle.vehicle.position.bearing,
                            'speed' : vehicle.vehicle.position.speed,
                            'current_stop_sequence' : vehicle.vehicle.current_stop_sequence,
                            'current_status' : vehicle.vehicle.current_status,
                            'timestamp' : vehicle.vehicle.timestamp,
                            'stop_id' : vehicle.vehicle.stop_id,
                            'vehicle_id' : vehicle.vehicle.vehicle.id,
                            'label' : vehicle.vehicle.vehicle.label,
                            'license_plate' : vehicle.vehicle.vehicle.license_plate
                        })
            
    # Save the last day's data      
    if prev_day_str and rows:
        csv_out = save_path / (prev_day_str + ".csv")
        df = pd.DataFrame(rows)
        df.to_csv(csv_out, mode="w", header=not csv_out.exists(), index=False)

def save_trip_occurance_json(records_path:Path):
    occurance = {}
    for csv_path in records_path.iterdir():
        trip_ids = pd.read_csv(csv_path, low_memory=False)['trip_id'].tolist()
        for trip_id in tqdm(trip_ids,desc=csv_path.name):
            if trip_id not in occurance.keys():
                occurance[trip_id] = 1
            else:
                occurance[trip_id] += 1
                # print(f'{trip_id} occured {occurance[trip_id]} times')

    with open(records_path.parent / 'trip_occurance.json', 'w') as f:
        json.dump(occurance, f)

def save_days_json(records_path:Path):
    days = {}
    for record_path in records_path.iterdir():
        realtime = pd.read_csv(record_path, low_memory=False)
        num_trips = len(realtime['trip_id'].unique())
        _, d = name_to_day(record_path.stem)
        if d in days.keys():
            days[d]['paths'].append(record_path._str)
            days[d]['num_trips'].append(num_trips)
        else:
            days[d] = {
                'paths': [record_path._str],
                'num_trips': [num_trips]
            }
        
    with open(records_path.parent / 'days.json', 'w') as f:
        json.dump(days, f)

def load_days_json(records_path: Path):
    with open(records_path.parent / 'days.json', 'r') as f:
        return json.load(f)


def train_test_split(records_path:Path):
    
    records = {'train' : [], 'test' : []}
    
    num_train_trips, num_test_trips = 0, 0
    for day, info in load_days_json(records_path).items():
        num_trips = info['num_trips']
        paths = info['paths']
        
        idx = num_trips.index(min(num_trips))
        
        num_train_trips  += sum(num_trips) - num_trips[idx]
        num_test_trips += num_trips[idx]
        
        for i in range(len(paths)):
            if i == idx:
                records['test'].append(Path(paths[i]).stem)
            else:
                records['train'].append(Path(paths[i]).stem)

    print(
        f'Split ratio: {num_test_trips / (num_train_trips + num_test_trips)*100:.2f} %'
    )

    with open(records_path.parent.parent / 'records.json', 'w') as f:
        json.dump(records, f)

def load_train_test_split(data_path:Path):
    with open(data_path / 'records.json', 'r') as f:
        return json.load(f)

def known_trips(record_path:Path, bus_trip_ids:list[str]):
    realtime = pd.read_csv(record_path, low_memory=False)
    realtime = realtime[['route_id','trip_id','current_stop_sequence','timestamp']]
    realtime['time'] = realtime['timestamp'].apply(timestamp_to_seconds)
    incoming_trip_ids = set(realtime['trip_id'].unique())
    available_trip_ids = set(bus_trip_ids)

    # Determine known trips
    unknown_trip_ids = incoming_trip_ids - available_trip_ids
    known_trip_ids = incoming_trip_ids - unknown_trip_ids

    return (
        len(realtime), len(known_trip_ids), len(unknown_trip_ids)
    )

def summarize_records(records_path:Path, bus_trip_ids:list[str]):
    
    for record_path in records_path.iterdir():
        
        n, n_kn, n_ukn = known_trips(
            record_path=record_path,
            bus_trip_ids=bus_trip_ids,
        )

        print(f'{record_path.stem}:')
        print(f'\t* Num of records: {n}')
        print(f'\t* Num of known trips: {n_kn}')
        print(f'\t* Num of unknown trips: {n_ukn}')




# -------------- DELAY CALCULATION --------------

def calculate_delays_per_day(record_path:Path, delay_path:Path, stop_times:pd.DataFrame)->None:

    realtime = pd.read_csv(record_path, low_memory=False)
    realtime = realtime[['route_id','trip_id','current_stop_sequence','timestamp']]
    realtime['time'] = realtime['timestamp'].apply(timestamp_to_seconds)
    incoming_trip_ids = set(realtime['trip_id'].unique())

    # Observe only incoming
    static = stop_times[stop_times['trip_id'].isin(incoming_trip_ids)].copy()
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
    
def calculate_delays(records_path:Path, stop_times_path:Path): 

    stop_times = pd.read_csv(stop_times_path, sep=',', low_memory=False)
    stop_times = stop_times[['trip_id','stop_sequence','arrival_time']].copy()
    stop_times['arrival_time'] = stop_times['arrival_time'].apply(hhmmss_to_seconds)

    
    for record_path in records_path.iterdir():
        delay_path = Path(record_path.parent) / 'delays'

        if not delay_path.exists(): 
            print(f'Calculating delays of {record_path.name}...')
            calculate_delays(
                record_path,
                delay_path,
                stop_times=stop_times,
            )

def load_delays(record_name, delay_path:Path):
    return pd.read_csv(delay_path / f'{record_name}.csv', low_memory=False).set_index('trip_id')


# ------------ TRAINING CASES ------------------

def find_cases_numpy(records_path: Path, train_path:Path):
    days = load_days_json(records_path)
    trip_day_list = []

    for day, record_paths in days.items():
        for record_path in record_paths:
            print(f'{day}/{record_path}')
            realtime = pd.read_csv(record_path, low_memory=False)
            for trip_id in realtime['trip_id'].unique():
                trip_day_list.append((day, trip_id))
    
    # Convert to NumPy array
    trip_day_array = np.array(trip_day_list, dtype='object')
    np.save(train_path / 'inputs.npy', trip_day_array)




# -------------- GRAPH CONSTRUCTION --------------

def graph_static_features(static_path:Path, graphs_path:Path):
    
    # Create folder
    graphs_path.mkdir(exist_ok=True)
    
    # Load necessary static data
    stops = pd.read_csv(static_path / 'stops.txt', low_memory=False)
    stop_times = pd.read_csv(static_path / 'stop_times.txt', low_memory=False)


    # Neceessary rows
    tmp = stop_times.copy()
    tmp = tmp[tmp['stop_headsign'].notna() & (tmp['stop_headsign'] != "")]
    tmp = stop_times[['trip_id','stop_id','arrival_time','departure_time','stop_sequence','shape_dist_traveled']].copy()
    tmp = tmp.sort_values(['trip_id', 'stop_sequence']).copy()

    # Next stop and arrival time
    tmp['next_stop'] = tmp.groupby('trip_id')['stop_id'].shift(-1)
    tmp['next_arrival_time'] = tmp.groupby('trip_id')['arrival_time'].shift(-1)
    tmp['next_shape_dist'] = tmp.groupby('trip_id')['shape_dist_traveled'].shift(-1)

    # Travel time and distance until the next stop
    tmp['dt'] = tmp['next_arrival_time'] - tmp['departure_time']
    tmp['ds'] = tmp['next_shape_dist'] - tmp['shape_dist_traveled']

    # Drop invalid rows, and rename the cols
    tmp = tmp.dropna(subset=['next_stop', 'dt'])[['stop_id', 'next_stop', 'dt', 'ds']]
    tmp = tmp.rename(columns={'stop_id': 'src', 'next_stop': 'dst'})

    # Edges: average travel time between nodes
    edges = tmp.groupby(['src', 'dst'], as_index=False).agg({
        'dt': 'mean',
        'ds': 'mean'
    })
    del tmp

    # Nodes: stops
    nodes = stops[['stop_id','stop_lat','stop_lon']]

    # Create ID - index mappings
    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)
    # nodes['node_idx'] = nodes.index
    # edges['edge_idx'] = edges.index
    # node_id_to_idx = dict(zip(nodes['stop_id'], nodes['node_idx']))

    # edges['src_idx'] = edges['src'].map(node_id_to_idx)
    # edges['dst_idx'] = edges['dst'].map(node_id_to_idx)

    # edge_id_to_idx = {f"{row['src']}_{row['dst']}": row['edge_idx'] for _, row in edges.iterrows()}

    # Save mappings
    # with open(graphs_path / "edge_idx_map.json", 'w') as f:
    #     json.dump(edge_id_to_idx, f)

    # with open(graphs_path / "node_idx_map.json", 'w') as f:
    #     json.dump(node_id_to_idx, f)

    # Check the filtering
    nodes_of_edges = set(pd.concat([edges['src'], edges['dst']]).unique())
    nodes_itself = set(nodes['stop_id'])
    

    print('Edge nodes, without node', nodes_of_edges - nodes_itself)
    print('Nodes, without edge', nodes_itself - nodes_of_edges)

    edges.to_csv(graphs_path / "edges.csv", index=False)
    nodes.to_csv(graphs_path / "nodes.csv", index=False)


def dynamic_node_features_by_bins(
        dt, # use dt=1/2 
        static_path:Path, records_path:Path, graphs_path:Path
    ):
    
    # Load graphs and their static features
    nodes = pd.read_csv(graphs_path / "nodes.csv").set_index('stop_id')
    edges = pd.read_csv(graphs_path / "edges.csv")

    # Calculate edge features
    stop_id_to_idx = {
        stop_id: i for i, stop_id in enumerate(nodes.index.to_list())
    }

    with open(graphs_path.parent / 'stop_id_to_idx.json','w') as f:
        json.dump(stop_id_to_idx, f)

    src_idx = edges['src'].map(stop_id_to_idx).to_numpy()
    dst_idx = edges['dst'].map(stop_id_to_idx).to_numpy()
    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
    edge_attr = torch.tensor(edges[['dt', 'ds']].values, dtype=torch.float)

    # Calc bins
    bins = np.arange(0,24+dt,dt)
    bins *= 60*60
    # Get the saved data

    # record_paths = load_days_json(dynamic_path / 'records')[day]

    # Load static GTFS data
    print("Load static GTFS data...")
    stop_times = pd.read_csv(static_path / 'stop_times.txt', sep=',', low_memory=False)
    stop_times = stop_times[['trip_id','stop_id','stop_sequence','arrival_time']] # 'departure_time','shape_dist_traveled'
    stop_times = stop_times.set_index(['trip_id','stop_id','stop_sequence'])

    for record_path in records_path.iterdir():
        print(f"Processing record: {record_path.name}")

        # Load and prefilter realtime data of the day
        realtime = pd.read_csv(record_path, low_memory=False)
        realtime['time'] = realtime['timestamp'].apply(timestamp_to_seconds)
        relevant_cols = ['trip_id','stop_id','current_stop_sequence','time']
        realtime = realtime[relevant_cols].dropna(subset=relevant_cols)
        realtime.rename(columns={'current_stop_sequence': 'stop_sequence'}, inplace=True)
        
        # Remove duplicants and set idx for merge
        realtime = realtime.groupby(['trip_id','stop_id','stop_sequence'], as_index=True).agg({'time':'mean'})
       
        # Merge static
        realtime['time_scheduled'] = realtime.index.map(stop_times['arrival_time'])
        
        # Calc delays for each ('trip_id','stop_id','stop_sequence')
        realtime['delay'] = realtime['time_scheduled'] - realtime['time']
        realtime = realtime[np.abs(realtime['delay']) < 900] # exclude outliers
        
        # Split into intervals
        realtime['bin'] = pd.cut(realtime['time'], bins=bins, right=False)
        realtime['bin_code'] = realtime['bin'].cat.codes

        # Groupby node and bin
        realtime = realtime.groupby(
            [realtime.index.get_level_values('stop_id'), 'bin_code'])['delay'].agg(
            ['count','mean', 'std', 'max', 'min']
        ) # level='stop_id'

        groups = realtime.reset_index(level='bin_code').groupby('bin_code')
        print(f"  Number of bins: {len(groups)}")

        for bin_code, group in groups:
            x = nodes.join(
                group.drop(columns='bin_code'), 
                how='left',
            ).reindex(nodes.index).fillna(0)
            x = torch.tensor(x.values, dtype=torch.float)
            print(f"    Number of nodes with features at {bin_code} bin: {len(group)}")
            # Save PyG object
            data = Data(
                x=x,
                edge_index=edge_index,      
                edge_attr=edge_attr,        
            )
            
            save_path = graphs_path / record_path.stem / f'graph_bin_{bin_code}.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, save_path)
        print(f"  Saved all bins for {record_path.stem}")


def load_graph(bin_code, record_name, graphs_path:Path):
    # bin_code = int((seconds % (24 * 3600)) // (dt * 3600))
    load_path = graphs_path / record_name / f"graph_bin_{bin_code}.pt"

    return torch.load(load_path, weights_only=False)

def available_bin_codes(record_name:str, graphs_path:Path):
    return set([
        int(f.stem.replace("graph_bin_", ""))
        for f in (graphs_path / record_name).iterdir()
        if f.is_file() and f.name.startswith("graph_bin_") and f.suffix == ".pt"
    ])

def load_stop_id_to_idx(graphs_path:Path):
    with open(graphs_path.parent / 'stop_id_to_idx.json','r') as f:
        return json.load(f)


def save_trip_idx_list(static_path:Path, graphs_path:Path):
    
    stop_id_to_idx = load_stop_id_to_idx(graphs_path)
    
    stop_times = pd.read_csv(static_path / 'stop_times.txt', low_memory=False)
    stop_times['node_idx'] = stop_times['stop_id'].map(stop_id_to_idx)

    trip_idx_list = {}

    for trip_id, group in stop_times.groupby('trip_id'):
        trip_idx_list[trip_id] = [
            int(x) for x in group['node_idx'].unique()
        ]

    with open(graphs_path.parent / 'trip_idx_list.json', 'w') as f:
        json.dump(trip_idx_list, f)


def load_trip_idx_list(graphs_path:Path): 
    with open(graphs_path.parent / 'trip_idx_list.json', 'r') as f:
        return json.load(f)



if __name__ == '__main__':

    root = Path('/mnt/c/Users/rdsup/Desktop/vitmma19-pw-delay-detection')
    
    static_path  = root / 'data' / 'static_gtfs_bus' # 'static_gtfs' -> old
    dynamic_path = root / 'data' / 'dynamic_gtfs'
    train_path  = root / 'data' / 'training'
    graphs_path  = root / 'data' / 'graphs'

    trips_path = static_path / 'trips.txt'
    records_path = dynamic_path / 'records'
    trials_path = dynamic_path / 'trials'
    

    dynamic_node_features_by_bins(
        dt=1/2,
        static_path=static_path,
        records_path=records_path,
        graphs_path=graphs_path,
    )

    # graph_static_features(static_path, graphs_path)
    # find_cases_numpy(records_path, train_path)

    # G = StaticGraphs(path=graphs_path, dt=1/2)



