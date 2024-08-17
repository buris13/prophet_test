import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, Response
from model import download_data, format_data, train_eth_model, train_btc_model, train_bnb_model, train_sol_model, train_arb_model
# from config import eth_file_path, btc_file_path, bnb_file_path, sol_file_path, arb_file_path

app = Flask(__name__)

eth_model = None
btc_model = None
bnb_model = None
sol_model = None
arb_model = None

def update_data():
    """Download price data, format data and train model."""
    download_data()
    format_data()
    global eth_model
    eth_model = train_eth_model()
    
    global btc_model
    btc_model = train_btc_model()

    global bnb_model
    bnb_model = train_bnb_model()

    global sol_model
    sol_model = train_sol_model()

    global arb_model
    sol_model = train_arb_model()


def get_eth_inference():
    """Generate prediction using Prophet model."""
    if eth_model is None:
        raise RuntimeError("Model is not trained or loaded.")

    now = pd.Timestamp(datetime.now())
    future = pd.DataFrame({'ds': [now]})

    forecast = eth_model.predict(future)
    current_price_pred = forecast['yhat'].values[0]

    return current_price_pred

def get_btc_inference():
    """Generate prediction using Prophet model."""
    if btc_model is None:
        raise RuntimeError("Model is not trained or loaded.")

    now = pd.Timestamp(datetime.now())
    future = pd.DataFrame({'ds': [now]})

    forecast = btc_model.predict(future)
    current_price_pred = forecast['yhat'].values[0]

    return current_price_pred

def get_bnb_inference():
    """Generate prediction using Prophet model."""
    if bnb_model is None:
        raise RuntimeError("Model is not trained or loaded.")

    now = pd.Timestamp(datetime.now())
    future = pd.DataFrame({'ds': [now]})

    forecast = bnb_model.predict(future)
    current_price_pred = forecast['yhat'].values[0]

    return current_price_pred

def get_sol_inference():
    """Generate prediction using Prophet model."""
    if sol_model is None:
        raise RuntimeError("Model is not trained or loaded.")

    now = pd.Timestamp(datetime.now())
    future = pd.DataFrame({'ds': [now]})

    forecast = sol_model.predict(future)
    current_price_pred = forecast['yhat'].values[0]

    return current_price_pred
def get_arb_inference():
    """Generate prediction using Prophet model."""
    if arb_model is None:
        raise RuntimeError("Model is not trained or loaded.")

    now = pd.Timestamp(datetime.now())
    future = pd.DataFrame({'ds': [now]})

    forecast = arb_model.predict(future)
    current_price_pred = forecast['yhat'].values[0]

    return current_price_pred

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    """Generate inference for given token."""
    try:
        if token == "ETH":
            inference = get_eth_inference()
        elif token == "BTC":
            inference = get_btc_inference()
        elif token == "BNB":
            inference = get_bnb_inference()
        elif token == "SOL":
            inference = get_sol_inference()
        elif token == "ARB":
            inference = get_arb_inference()
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
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
