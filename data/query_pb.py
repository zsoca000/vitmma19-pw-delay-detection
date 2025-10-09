import requests
from datetime import datetime, timedelta
import os
import time
import logging

# --------------------------
# Configuration
# --------------------------
API_KEY = "55bc10e5-ce8e-4a18-ba6f-fdee71636724"
URL = f"https://go.bkk.hu/api/query/v1/ws/gtfs-rt/full/VehiclePositions.pb?key={API_KEY}"
SAVE_DIR = "dynamic_gtfs"
INTERVAL_SECONDS = 60      # Run every 1 minute
RETENTION_DAYS = 14        # Keep 2 weeks of files
MAX_RETRIES = 3             # Retries if request fails
RETRY_DELAY = 10           # Seconds between retries

# --------------------------
# Setup logging
# --------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(SAVE_DIR, "downloader.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.getLogger().addHandler(logging.StreamHandler())

# --------------------------
# Helper functions
# --------------------------
def save_gtfs_feed(content: bytes):
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"feed_{timestamp_str}.pb"
    filepath = os.path.join(SAVE_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(content)
        logging.info(f"Saved GTFS feed: {filename}")
    except Exception as e:
        logging.error(f"Failed to save GTFS feed: {e}")
    return filepath

def cleanup_old_files():
    """Delete files older than RETENTION_DAYS."""
    cutoff_time = datetime.now() - timedelta(days=RETENTION_DAYS)
    for fname in os.listdir(SAVE_DIR):
        if fname.endswith(".pb"):
            fpath = os.path.join(SAVE_DIR, fname)
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(fpath))
                if file_time < cutoff_time:
                    os.remove(fpath)
                    logging.info(f"Deleted old feed file: {fname}")
            except Exception as e:
                logging.error(f"Error checking/deleting file {fname}: {e}")

def fetch_gtfs():
    """Fetch GTFS feed with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(URL, timeout=10)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"All {MAX_RETRIES} attempts failed.")
                return None

def get_cpu_temp():
    """Read CPU temperature in Celsius."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_milli = int(f.read().strip())
        return temp_milli / 1000.0
    except Exception as e:
        logging.error(f"Failed to read CPU temperature: {e}")
        return None

# --------------------------
# Main loop
# --------------------------
if __name__ == "__main__":
    logging.info("Starting GTFS downloader.")
    while True:
        feed_content = fetch_gtfs()
        cpu_temp = get_cpu_temp()
        if feed_content:
            filepath = save_gtfs_feed(feed_content)
            cleanup_old_files()
        if cpu_temp is not None:
            logging.info(f"CPU Temperature: {cpu_temp:.1f}C")
        time.sleep(INTERVAL_SECONDS)
