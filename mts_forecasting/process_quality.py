# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
import zipfile
import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"


def process_quality_data(file: str = "AirQualityUCI.xlsx",
                         root: str = "./data/quality",
                         download: bool = True
                         ) -> pd.DataFrame:
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

    # Load data from engine
    df = pd.read_excel(f'{root}/' + file)

    # Drop column "NHMC(GT)" (contains only missing values)
    df = df.drop(columns=["NMHC(GT)"])
    df = interpolate_missing_values(df)

    return df


def download_file(url: str,
                  filename: str) -> None:
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print("Failed to download the file.")


def extract_specific_file(zip_filename: str,
                          target_filename: str,
                          extract_to_folder: str = '.'
                          ) -> None:
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # List all file names in the zip
        file_names = zip_ref.namelist()

        if target_filename in file_names:
            # Extract the file data
            file_data = zip_ref.read(target_filename)

            # Define the full path for the extracted file
            extracted_file_path = os.path.join(extract_to_folder, target_filename)

            # Write the extracted file
            with open(extracted_file_path, 'wb') as f:
                f.write(file_data)

            print(f"Extracted {target_filename} to {extracted_file_path}")
        else:
            print(f"{target_filename} not found in the zip file.")


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates missing values (marked as -200) in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame with missing values marked as -200.

    Returns:
    pd.DataFrame: DataFrame with missing values interpolated.
    """

    # Drop "Date" and "Time" columns
    df = df.drop(columns=["Date", "Time"])

    # Replace -200 with NaN to mark as missing
    df_replaced = df.replace(-200, np.nan)

    # Perform linear interpolation
    df_interpolated = df_replaced.interpolate(method='linear', axis=0)

    # Fill the NaNs at the start or end of the columns, if any
    df_interpolated.bfill(axis=0, inplace=True)  # Backward fill
    df_interpolated.ffill(axis=0, inplace=True)  # Forward fill

    return df_interpolated


if __name__ == '__main__':
    df = process_quality_data()
    print(df.head())
