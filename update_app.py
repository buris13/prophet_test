from app import download_data, format_data


def update_data():
    """Download price data, format data and train model."""
    download_data()
    format_data()