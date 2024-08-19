import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta


# Function to download the URL, called asynchronously by several child processes
def download_url(url, download_path):
    target_file_path = os.path.join(download_path, os.path.basename(url)) 
    if os.path.exists(target_file_path):
        # print(f"File already exists: {url}")
        return
    
    response = requests.get(url)
    if response.status_code == 404:
        # print(f"File not exist: {url}")
        pass
    else:

        # create the entire path if it doesn't exist
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

        with open(target_file_path, "wb") as f:
            f.write(response.content)
        # print(f"Downloaded: {url} to {target_file_path}")


def download_binance_monthly_data(
    cm_or_um, symbols, intervals, years, months, download_path
):
    # Verify if CM_OR_UM is correct, if not, exit
    if cm_or_um not in ["cm", "um"]:
        print("CM_OR_UM can be only cm or um")
        return
    base_url = f"https://data.binance.vision/data/futures/{cm_or_um}/monthly/klines"

    # Main loop to iterate over all the arrays and launch child processes
    with ThreadPoolExecutor() as executor:
        for symbol in symbols:
            for interval in intervals:
                for year in years:
                    for month in months:
                        url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month}.zip"
                        executor.submit(download_url, url, download_path)


def download_binance_daily_data(
    cm_or_um, symbols, intervals, year, month, download_path
):
    if cm_or_um not in ["cm", "um"]:
        print("CM_OR_UM can be only cm or um")
        return
    base_url = f"https://data.binance.vision/data/futures/{cm_or_um}/daily/klines"

    with ThreadPoolExecutor() as executor:
        for symbol in symbols:
            for interval in intervals:
                for day in range(1, 32):  # Assuming days range from 1 to 31
                    url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
                    executor.submit(download_url, url, download_path)

def download_binance_daily_spot_data(
    symbols, intervals, download_path
):    
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    
    # Convert start_date and end_date to datetime objects
    # today = datetime.today().strftime('%Y-%m-%d')
    today = datetime.today()
    end_date = today - timedelta(days=1)
    start_date = today -timedelta(days=4)
    
    with ThreadPoolExecutor() as executor:
        for symbol in symbols:
            for interval in intervals:
                current_date = start_date
                while current_date <= end_date:
                    url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{current_date.strftime('%Y-%m-%d')}.zip"
                    executor.submit(download_url, url, download_path)
                    current_date += timedelta(days=1)

def download_binance_daily_batch_data(
    cm_or_um, symbols, intervals, day, download_path
):    
    if cm_or_um not in ["cm", "um"]:
        print("CM_OR_UM can be only cm or um")
        return
    base_url = f"https://data.binance.vision/data/futures/{cm_or_um}/daily/klines"
    
    # Convert start_date and end_date to datetime objects
    today = datetime.today()
    end_date = today - timedelta(days=1)
    start_date = today - timedelta(days=day)
    
    with ThreadPoolExecutor() as executor:
        for symbol in symbols:
            for interval in intervals:
                current_date = start_date
                while current_date <= end_date:
                    url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{current_date.strftime('%Y-%m-%d')}.zip"
                    executor.submit(download_url, url, download_path)
                    current_date += timedelta(days=1)