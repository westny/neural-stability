import os
import torch
import requests
import zipfile
import pandas as pd

from typing import List
from dataclasses import dataclass

url = "https://vehsys.gitlab-pages.liu.se/diagnostic_competition/competition/training_data/trainingdata.zip"

signals_csv = {"Intercooler_pressure": "y_p_ic",
               "intercooler_temperature": "y_T_ic",
               "intake_manifold_pressure": "y_p_im",
               "air_mass_flow": "y_W_af",
               "engine_speed": "y_omega_e",
               "throttle_position": "y_alpha_th",
               "wastegate_position": "y_u_wg",
               "injected_fuel_mass": "y_wfc",
               "ambient_temperature": "y_T_amb",
               "ambient_pressure": "y_p_amb"}


@dataclass
class Config:
    measurements: List[str]
    sensor: str


def process_data(file="wltp_NF.csv", root="./data/engine", download=True, keep=-1):
    # Check if data exists
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(f'{root}/' + file):
        if download:
            print(f'Downloading {file}...')
            download_file(url, f'{root}/data.zip')
            extract_specific_file(f'{root}/data.zip', file, root)
            os.remove(f'{root}/data.zip')
        else:
            raise FileNotFoundError(f'{root}/' + file)

    y = {}

    # Load data from engine
    df = pd.read_csv(f'{root}/' + file)

    # index by signal_csv keys
    df = df.rename(columns=signals_csv)

    # store values in dict
    for key in signals_csv.values():
        z = torch.from_numpy(df[key].values).float()
        y[key] = z[:keep] if keep > 0 else z

    t = None

    # basic configuration
    config = Config(measurements=['y_W_af', 'y_alpha_th', 'y_omega_e', 'y_p_amb', 'y_p_im', 'y_u_wg', 'y_wfc'],
                    sensor='y_p_ic')

    return y, t, config


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        total_size_in_gb = total_size_in_bytes / (1024 * 1024 * 1024)  # Convert to GB
        block_size = 1 * 1024 * 1024  # 10 MB
        progress = 0

        with open(filename, 'wb') as file:
            for data in r.iter_content(block_size):
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
                progress += len(data)
                downloaded_in_gb = progress / (1024 * 1024 * 1024)  # Convert to GB
                done = int(50 * progress / total_size_in_bytes)
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_in_gb:.2f}/{total_size_in_gb:.2f} GB", end='')
        print()


def extract_specific_file(zip_filename, target_filename, extract_to_folder='.'):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # List all file names in the zip
        file_names = zip_ref.namelist()

        # Construct the full path of the file in the zip
        full_path_in_zip = os.path.join("trainingdata", target_filename)

        if full_path_in_zip in file_names:
            # Extract the file data
            file_data = zip_ref.read(full_path_in_zip)

            # Define the full path for the extracted file
            extracted_file_path = os.path.join(extract_to_folder, target_filename)

            # Write the extracted file
            with open(extracted_file_path, 'wb') as f:
                f.write(file_data)

            print(f"Extracted {target_filename} to {extracted_file_path}")
        else:
            print(f"{target_filename} not found in the zip file.")


if __name__ == '__main__':
    y, _, conf = process_data()
