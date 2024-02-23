from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()  # This will use your Kaggle API credentials from ~/.kaggle/kaggle.json

# List datasets
datasets = api.dataset_list()
print(datasets)