from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()  # This will use your Kaggle API credentials from ~/.kaggle/kaggle.json

# Download dataset
api.dataset_download_files('ronikdedhia/next-word-prediction', path=r'C:\Users\noodle\Documents\.kaggle')