from flask import Flask, jsonify, Response
import json
from config import eth_csv_path, btc_csv_path, bnb_csv_path, sol_csv_path, arb_csv_path
from model import download_data, format_data, predict_eth, predict_btc, predict_bnb, predict_sol, predict_arb
import pandas as pd
from datetime import datetime, UTC

app = Flask(__name__)

def update_data():
    """Download price data, format data and generate csv predict price."""
    download_data()
    format_data()
    predict_eth()
    predict_btc()
    predict_bnb()
    predict_sol()
    predict_arb()
    

def get_eth_price(Id_minutes):
    timestamp_now = int(datetime.now(UTC).timestamp())
    df = pd.read_csv(eth_csv_path)
    timestamp_future = int(timestamp_now) + Id_minutes * 60
    prediction = df[df['unix_time'] == timestamp_future]
    if not prediction.empty:
        result = prediction[['yhat']].to_dict(orient='records')[0]
        return str(result['yhat'])
    else:
        return "Prediction not available"
    
def get_btc_price(Id_minutes):
    timestamp_now = int(datetime.now(UTC).timestamp())
    df = pd.read_csv(btc_csv_path)
    timestamp_future = int(timestamp_now) + Id_minutes * 60
    prediction = df[df['unix_time'] == timestamp_future]
    if not prediction.empty:
        result = prediction[['yhat']].to_dict(orient='records')[0]
        return str(result['yhat'])
    else:
        return "Prediction not available"
    
def get_bnb_price(Id_minutes):
    timestamp_now = int(datetime.now(UTC).timestamp())
    df = pd.read_csv(bnb_csv_path)
    timestamp_future = int(timestamp_now) + Id_minutes * 60
    prediction = df[df['unix_time'] == timestamp_future]
    if not prediction.empty:
        result = prediction[['yhat']].to_dict(orient='records')[0]
        return str(result['yhat'])
    else:
        return "Prediction not available"
    
def get_sol_price(Id_minutes):
    timestamp_now = int(datetime.now(UTC).timestamp())
    df = pd.read_csv(sol_csv_path)
    timestamp_future = int(timestamp_now) + Id_minutes * 60
    prediction = df[df['unix_time'] == timestamp_future]
    if not prediction.empty:
        result = prediction[['yhat']].to_dict(orient='records')[0]
        return str(result['yhat'])
    else:
        return "Prediction not available"

def get_arb_price(Id_minutes):
    timestamp_now = int(datetime.now(UTC).timestamp())
    df = pd.read_csv(arb_csv_path)
    timestamp_future = int(timestamp_now) + Id_minutes * 60
    prediction = df[df['unix_time'] == timestamp_future]
    if not prediction.empty:
        result = prediction[['yhat']].to_dict(orient='records')[0]
        return str(result['yhat'])
    else:
        return "Prediction not available"
    

@app.route("/inference/<string:token>/<int:Id_minutes>")
def generate_inference(token, Id_minutes):
    """Generate inference for given token."""
    try:
        if token == "ETH":
            inference = get_eth_price(Id_minutes)
        elif token == "BTC":
            inference = get_btc_price(Id_minutes)
        elif token == "BNB":
            inference = get_bnb_price(Id_minutes)
        elif token == "SOL":
            inference = get_sol_price(Id_minutes)
        elif token == "ARB":
            inference = get_arb_price(Id_minutes)
        else:
            return Response(json.dumps({"error": "Token not supported"}), status=400, mimetype='application/json')
        
        return Response(str(inference), status=200)

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception as e:
        print(f"Error occurred: {e}")
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)