import os

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")

#csv path
eth_csv_path = os.path.join(data_base_path, "eth_predict.csv")
btc_csv_path = os.path.join(data_base_path, "btc_predict.csv")
bnb_csv_path = os.path.join(data_base_path, "bnb_predict.csv")
sol_csv_path = os.path.join(data_base_path, "sol_predict.csv")
arb_csv_path = os.path.join(data_base_path, "arb_predict.csv")

