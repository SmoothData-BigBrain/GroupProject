import gdown
import zipfile
import os
import glob

def fetch_data(file_id, download_path, extract_to):
    """
    Fetches and downloads the data from Google Drive, unzips it to the specified directory.
    
    Parameters:
    - file_id: str, Google Drive file ID for the file to download
    - download_path: str, path to save the downloaded zip file
    - extract_to: str, path to extract the zip file content
    """
    # Ensure the extraction folder exists
    os.makedirs(extract_to, exist_ok=True)

    # Download the file from Google Drive to the specified path
    print("Downloading data...")
    gdown.download(f'https://drive.google.com/uc?id={file_id}', download_path, quiet=False)

    # Unzip the downloaded file to the specified directory
    print("Extracting data...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"File extracted to: {extract_to}")

def check_and_fetch_data(file_id, data_dir):
    """
    Checks if there are any CSV files in the specified directory.
    If not, it triggers the fetch_data function to download and extract the data.
    
    Parameters:
    - data_dir: str, directory to check for CSV files and download the data
    """

    # # Ensure the extraction folder exists
    os.makedirs(data_dir+'archive/raw/', exist_ok=True) # maybe remove the +...

    # Define paths
    download_path = os.path.join(data_dir, 'airlines.zip')  # Path to save the downloaded zip file

    # Check if any .CSV files exist in the directory
    csv_files = [f for f in os.listdir(data_dir+'archive/raw/') if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {data_dir}. Fetching data...")
        # Fetch the data if no CSV files are present
        fetch_data(file_id, download_path, data_dir)
    else:
        print(f"CSV files already exist in {data_dir}. Skipping data download.")