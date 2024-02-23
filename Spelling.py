
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()  # This will use your Kaggle API credentials from ~/.kaggle/kaggle.json

# Download dataset
api.dataset_download_files('bittlingmayer/spelling', path=r'C:\Users\noodle\Documents\.kaggle')