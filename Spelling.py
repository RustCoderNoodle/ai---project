from zipfile import ZipFile
file_name = "spelling.zip"
with ZipFile(file_name, 'r') as zip:
    zip.printdir()
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()  # This will use your Kaggle API credentials from ~/.kaggle/kaggle.json

# Download dataset
api.dataset_download_files('bittlingmayer/spelling', path=r'C:\Users\noodle\Documents\.kaggle')
