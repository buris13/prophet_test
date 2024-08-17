import os
from zipfile import ZipFile
from datetime import datetime
import pandas as pd
from prophet import Prophet
from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path


binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
eth_training_price_data_path = os.path.join(data_base_path, "eth_price_data.csv")
btc_training_price_data_path = os.path.join(data_base_path, "btc_price_data.csv")
bnb_training_price_data_path = os.path.join(data_base_path, "bnb_price_data.csv")
sol_training_price_data_path = os.path.join(data_base_path, "sol_price_data.csv")
arb_training_price_data_path = os.path.join(data_base_path, "arb_price_data.csv")


def download_data():
    cm_or_um = "um"
    symbols = ["ETHUSDT", "BTCUSDT", "BNBUSDT", "SOLUSDT", "ARBUSDT"]
    intervals = ["1d"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    download_path = binance_data_path
    download_binance_monthly_data(
        cm_or_um, symbols, intervals, years, months, download_path
    )
    print(f"Downloaded monthly data to {download_path}.")
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    download_binance_daily_data(
        cm_or_um, symbols, intervals, current_year, current_month, download_path
    )
    print(f"Downloaded daily data to {download_path}.")


def format_data():
    eth_price_df = pd.DataFrame()
    btc_price_df = pd.DataFrame()
    bnb_price_df = pd.DataFrame()
    sol_price_df = pd.DataFrame()
    arb_price_df = pd.DataFrame()

    files = sorted([x for x in os.listdir(binance_data_path)])

    # No files to process
    if len(files) == 0:
        return

    for file in files:
        zip_file_path = os.path.join(binance_data_path, file)

        if not zip_file_path.endswith(".zip"):
            continue

        myzip = ZipFile(zip_file_path)
        with myzip.open(myzip.filelist[0]) as f:
            line = f.readline()
            header = 0 if line.decode("utf-8").startswith("open_time") else None
        df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
        df.columns = [
            "start_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "end_time",
            "volume_usd",
            "n_trades",
            "taker_volume",
            "taker_volume_usd",
        ]
        df.index = [pd.Timestamp(x + 1, unit="ms") for x in df["end_time"]]
        df.index.name = "date"
        if "ETHUSDT" in file:
            eth_price_df = pd.concat([eth_price_df, df])
        elif "BTCUSDT" in file:
            btc_price_df = pd.concat([btc_price_df, df])
        elif "BNBUSDT" in file:
            bnb_price_df = pd.concat([bnb_price_df, df])
        elif "SOLUSDT" in file:
            sol_price_df = pd.concat([sol_price_df, df])
        elif "ARBUSDT" in file:
            arb_price_df = pd.concat([arb_price_df, df])

    eth_price_df.sort_index().to_csv(os.path.join(data_base_path, "eth_price_data.csv"))
    btc_price_df.sort_index().to_csv(os.path.join(data_base_path, "btc_price_data.csv"))
    bnb_price_df.sort_index().to_csv(os.path.join(data_base_path, "bnb_price_data.csv"))
    sol_price_df.sort_index().to_csv(os.path.join(data_base_path, "sol_price_data.csv"))
    arb_price_df.sort_index().to_csv(os.path.join(data_base_path, "arb_price_data.csv"))

def train_eth_model():
    # Load the eth price data
    price_data = pd.read_csv(eth_training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["ds"] = pd.to_datetime(price_data["date"])
    df["y"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    #Train Prophet Model
    model = Prophet()
    model.fit(df)

    # Return the trained model
    return model


def train_btc_model():
    # Load the eth price data
    price_data = pd.read_csv(btc_training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["ds"] = pd.to_datetime(price_data["date"])
    df["y"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    #Train Prophet Model
    model = Prophet()
    model.fit(df)

    # Return the trained model
    return model


def train_bnb_model():
    # Load the eth price data
    price_data = pd.read_csv(bnb_training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["ds"] = pd.to_datetime(price_data["date"])
    df["y"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    #Train Prophet Model
    model = Prophet()
    model.fit(df)

    # Return the trained model
    return model


def train_sol_model():
    # Load the eth price data
    price_data = pd.read_csv(sol_training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["ds"] = pd.to_datetime(price_data["date"])
    df["y"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    #Train Prophet Model
    model = Prophet()
    model.fit(df)

    # Return the trained model
    return model


def train_arb_model():
    # Load the eth price data
    price_data = pd.read_csv(arb_training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["ds"] = pd.to_datetime(price_data["date"])
    df["y"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    #Train Prophet Model
    model = Prophet()
    model.fit(df)

    # Return the trained model
    return model
