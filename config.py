import os

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")

#model path
# eth_file_path = os.path.join(data_base_path, "eth_model.pkl")
# btc_file_path = os.path.join(data_base_path, "btc_model.pkl")
# bnb_file_path = os.path.join(data_base_path, "bnb_model.pkl")
# sol_file_path = os.path.join(data_base_path, "sol_model.pkl")
# arb_file_path = os.path.join(data_base_path, "arb_model.pkl")