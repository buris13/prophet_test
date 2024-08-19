import os
from zipfile import ZipFile
import pandas as pd
from prophet import Prophet
from updater import download_binance_monthly_data, download_binance_daily_data, download_binance_daily_spot_data, download_binance_daily_batch_data
from config import data_base_path
from datetime import datetime, timedelta
import json
from flask import Flask, jsonify, Response

app = Flask(__name__)

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
eth_training_price_data_path = os.path.join(data_base_path, "eth_price_data.csv")
btc_training_price_data_path = os.path.join(data_base_path, "btc_price_data.csv")
bnb_training_price_data_path = os.path.join(data_base_path, "bnb_price_data.csv")
sol_training_price_data_path = os.path.join(data_base_path, "sol_price_data.csv")
arb_training_price_data_path = os.path.join(data_base_path, "arb_price_data.csv")


def download_data():
    cm_or_um = "um"
    symbols = ["ETHUSDT", "BTCUSDT", "BNBUSDT", "SOLUSDT", "ARBUSDT"]
    intervals = ["1h"]
    years = ["2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    day = 600
    download_path = binance_data_path
    # download_binance_monthly_data(
    #     cm_or_um, symbols, intervals, years, months, download_path
    # )
    # print(f"Downloaded monthly data to {download_path}.")
    # current_datetime = datetime.now()
    # current_year = current_datetime.year
    # current_month = current_datetime.month
    # download_binance_daily_data(
    #     cm_or_um, symbols, intervals, current_year, current_month, download_path
    # )
    # print(f"Downloaded daily data to {download_path}.")

    # download_binance_daily_spot_data(symbols, intervals, download_path)
    download_binance_daily_batch_data(
    cm_or_um, symbols, intervals, day, download_path
)


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
    print('data generated')

    eth_df = pd.read_csv(eth_training_price_data_path)
    eth_df = eth_df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    eth_df.to_csv(eth_training_price_data_path, index=False)

    btc_df = pd.read_csv(btc_training_price_data_path)
    btc_df = btc_df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    btc_df.to_csv(btc_training_price_data_path, index=False)

    bnb_df = pd.read_csv(bnb_training_price_data_path)
    bnb_df = bnb_df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    bnb_df.to_csv(bnb_training_price_data_path, index=False)

    sol_df = pd.read_csv(sol_training_price_data_path)
    sol_df = sol_df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    sol_df.to_csv(sol_training_price_data_path, index=False)

    arb_df = pd.read_csv(arb_training_price_data_path)
    arb_df = arb_df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    arb_df.to_csv(arb_training_price_data_path, index=False)
    print('data formated')


# def prophet_eth():
#     df = pd.read_csv(eth_training_price_data_path)
#     model = Prophet()
#     model.add_seasonality(name='daily', period=1440, fourier_order=8)  # 1440 menit = 1 hari
#     model.fit(df)

#     future = model.make_future_dataframe(periods=24*60, freq='10T')
#     forecast = model.predict(future)

#     forecast_data = forecast[['ds', 'yhat']].tail(10)

#     print('ini adalah', forecast_data)

#     return forecast_data

def predict_eth(minutes_variable):
    df = pd.read_csv(eth_training_price_data_path)
    model = Prophet()

#     model = Prophet(
#     daily_seasonality=True,  # Menonaktifkan seasonality harian jika tidak diperlukan
#     yearly_seasonality=False,  # Mengaktifkan seasonality tahunan jika relevan
#     weekly_seasonality=False,   # Mengaktifkan seasonality mingguan jika relevan
#     seasonality_prior_scale=2.0
#     # changepoint_prior_scale=1.0
#     # seasonality_mode='multiplicative'
# )
    model.add_seasonality(name='daily', period=1440, fourier_order=8)  # 1440 menit = 1 hari
    # Menambahkan seasonality untuk 10 menit
    model.add_seasonality(name='10_minute', period=10, fourier_order=5)

    # Menambahkan seasonality untuk 20 menit
    model.add_seasonality(name='20_minute', period=20, fourier_order=5)
    model.fit(df)


    now = datetime.now()
    future_time = now + timedelta(minutes=minutes_variable)
    future_df = pd.DataFrame({'ds': [future_time]})
    forecast = model.predict(future_df)
    # model.predict(future_df)
    forecast_data = forecast['yhat'].iloc[0]

    # forecast_data = forecast[['ds', 'yhat']].tail(10)
    # print('ini adalah', forecast_data)
    return forecast_data
    # return forecast_data

# from model import download_data, format_data, train_eth
def predict_btc(minutes_variable):
    df = pd.read_csv(btc_training_price_data_path)
    model = Prophet()

    model.add_seasonality(name='daily', period=1440, fourier_order=8)  # 1440 menit = 1 hari
    model.add_seasonality(name='10_minute', period=10, fourier_order=5)
    model.add_seasonality(name='20_minute', period=20, fourier_order=5)
    model.fit(df)

    now = datetime.now()
    future_time = now + timedelta(minutes=minutes_variable)
    future_df = pd.DataFrame({'ds': [future_time]})
    forecast = model.predict(future_df)
    forecast_data = forecast['yhat'].iloc[0]

    return forecast_data

def predict_bnb(minutes_variable):
    df = pd.read_csv(bnb_training_price_data_path)
    model = Prophet()

    model.add_seasonality(name='daily', period=1440, fourier_order=8)  # 1440 menit = 1 hari
    model.add_seasonality(name='10_minute', period=10, fourier_order=5)
    model.add_seasonality(name='20_minute', period=20, fourier_order=5)
    model.fit(df)

    now = datetime.now()
    future_time = now + timedelta(minutes=minutes_variable)
    future_df = pd.DataFrame({'ds': [future_time]})
    forecast = model.predict(future_df)
    forecast_data = forecast['yhat'].iloc[0]

    return forecast_data

def predict_sol(minutes_variable):
    df = pd.read_csv(sol_training_price_data_path)
    model = Prophet()

    model.add_seasonality(name='daily', period=1440, fourier_order=8)  # 1440 menit = 1 hari
    model.add_seasonality(name='10_minute', period=10, fourier_order=5)
    model.add_seasonality(name='20_minute', period=20, fourier_order=5)
    model.fit(df)

    now = datetime.now()
    future_time = now + timedelta(minutes=minutes_variable)
    future_df = pd.DataFrame({'ds': [future_time]})
    forecast = model.predict(future_df)
    forecast_data = forecast['yhat'].iloc[0]

    return forecast_data

def predict_arb(minutes_variable):
    df = pd.read_csv(arb_training_price_data_path)
    model = Prophet()

    model.add_seasonality(name='daily', period=1440, fourier_order=8)  # 1440 menit = 1 hari
    model.add_seasonality(name='10_minute', period=10, fourier_order=5)
    model.add_seasonality(name='20_minute', period=20, fourier_order=5)
    model.fit(df)

    now = datetime.now()
    future_time = now + timedelta(minutes=minutes_variable)
    future_df = pd.DataFrame({'ds': [future_time]})
    forecast = model.predict(future_df)
    forecast_data = forecast['yhat'].iloc[0]

    return forecast_data

##Flask
@app.route("/inference/<string:token>/<int:topicId_minutes>")
def generate_inference(token, topicId_minutes):
    """Generate inference for given token."""
    """Generate inference for given token."""
    try:
        if token == "ETH":
            inference = predict_eth(topicId_minutes)
        elif token == "BTC":
            inference = predict_btc(topicId_minutes)
        elif token == "BNB":
            inference = predict_bnb(topicId_minutes)
        elif token == "SOL":
            inference = predict_sol(topicId_minutes)
        elif token == "ARB":
            inference = predict_arb(topicId_minutes)
        else:
            return Response(json.dumps({"error": "Token not supported"}), status=400, mimetype='application/json')
        
        return Response(str(inference), status=200)

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

def update_data():
    """Download price data, format data and train model."""
    download_data()
    format_data()

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)