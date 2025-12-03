import datetime

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
    dt = datetime.datetime.fromtimestamp(int(ts))
    return dt.hour*3600 + dt.minute*60 + dt.second