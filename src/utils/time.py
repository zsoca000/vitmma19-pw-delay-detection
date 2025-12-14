import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def timestamp_to_hhmmss(ts): 
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")

def timestamp_to_yyyymmdd_hhmmss(ts): 
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def seconds_to_hhmmss(sec):
    return str(datetime.timedelta(seconds=sec))

def hhmmss_to_seconds(hms):
    # If input is a string "HH:MM:SS"
    if isinstance(hms, str):
        h, m, s = map(int, hms.split(':'))
        return h*3600 + m*60 + s
    # If input is a datetime.timedelta
    elif isinstance(hms, datetime.timedelta):
        return int(hms.total_seconds())
    else:
        raise ValueError("Input must be a string 'HH:MM:SS' or datetime.timedelta")

def timestamp_to_seconds(ts):
    dt = datetime.datetime.fromtimestamp(int(ts), tz=ZoneInfo("Europe/Budapest"))
    return dt.hour*3600 + dt.minute*60 + dt.second

def timestamp_to_day(ts):
    dt = datetime.datetime.fromtimestamp(int(ts))
    return dt.isoweekday()


def name_to_day(date_str: str) -> tuple[int,str]:
    """
    Returns: (0, 'Monday')
    """
    d = datetime.datetime.strptime(date_str, "%Y%m%d")
    return d.weekday(), d.strftime("%A")

if __name__ == '__main__':
    root = Path('/mnt/c/Users/rdsup/Desktop/vitmma19-pw-delay-detection')
    static_path  = root / 'data' / 'static_gtfs'
    dynamic_path = root / 'data' / 'dynamic_gtfs'
    records_path = dynamic_path / 'records'

    for file in records_path.iterdir():
        print(file.stem, name_to_day(file.stem))