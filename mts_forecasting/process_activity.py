import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.datasets.utils import download_url


def process_activity_data(root, url=None, tag_ids=None, label_dict=None, output_file="data.csv"):
    """
    Process the activity data from the given URL and save it to a CSV file.

    :param root: Root directory to save the data to.
    :param url: URL to download the data from.
    :param tag_ids: List of tag IDs.
    :param label_dict: Dictionary to map labels.
    :param output_file: Name of the output CSV file.
    """
    if url is None:
        # https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt'

    if tag_ids is None:
        tag_ids = [
            "010-000-024-033",  # "ANKLE_LEFT",
            "010-000-030-096",  # "ANKLE_RIGHT",
            "020-000-033-111",  # "CHEST",
            "020-000-032-221"  # "BELT"
        ]

    if label_dict is None:
        label_dict = {
            "walking": 0,
            "falling": 1,
            "lying": 2,
            "lying down": 2,
            "sitting": 3,
            "sitting down": 3,
            "standing up from lying": 4,
            "standing up from sitting": 4,
            "standing up from sitting on the ground": 4,
            "on all fours": 5,
            "sitting on the ground": 6
        }

    # Download the data
    download_url(url, root, "data.txt", md5=None)

    # Initialize arrays for each tag
    arrays = {tag_id: {"x": [], "y": [], "z": [], "t": [], "label": []} for tag_id in tag_ids}
    record_id = None
    dfs = []

    txt_path = os.path.join(root, "data.txt")

    with open(txt_path) as f:
        for line in tqdm(f):
            cur_id, tag_id, time, date, x, y, z, label = line.split(",")

            if record_id is None:
                record_id = cur_id

            if record_id != cur_id:
                data_dict = {
                    "record_id": record_id,
                    "x1": [], "y1": [], "z1": [],
                    "x2": [], "y2": [], "z2": [],
                    "x3": [], "y3": [], "z3": [],
                    "x4": [], "y4": [], "z4": [],
                    "label": []
                }

                # Find the index of the longest array
                idx = np.argmax([len(arrays[tag_id]["x"]) for tag_id in tag_ids])
                new_arrays = {tag_id: {"x": [], "y": [], "z": []} for tag_id in tag_ids}

                # Process arrays
                for i in range(len(arrays[tag_ids[idx]]["x"])):
                    t = arrays[tag_ids[idx]]["t"][i]
                    for tag_id in tag_ids:
                        j = np.argmin(np.abs(np.array(arrays[tag_id]["t"]) - t))
                        new_arrays[tag_id]["x"].append(arrays[tag_id]["x"][j])
                        new_arrays[tag_id]["y"].append(arrays[tag_id]["y"][j])
                        new_arrays[tag_id]["z"].append(arrays[tag_id]["z"][j])

                for i, tag_id in enumerate(tag_ids):
                    data_dict[f"x{i + 1}"] = np.array(new_arrays[tag_id]["x"])
                    data_dict[f"y{i + 1}"] = np.array(new_arrays[tag_id]["y"])
                    data_dict[f"z{i + 1}"] = np.array(new_arrays[tag_id]["z"])

                data_dict["label"] = arrays[tag_ids[idx]]["label"]
                dfs.append(pd.DataFrame(data_dict))

                arrays = {tag_id: {"x": [], "y": [], "z": [], "t": [], "label": []} for tag_id in tag_ids}
                record_id = cur_id

            arrays[tag_id]["x"].append(float(x))
            arrays[tag_id]["y"].append(float(y))
            arrays[tag_id]["z"].append(float(z))
            arrays[tag_id]["t"].append(float(time))
            arrays[tag_id]["label"].append(label_dict[label.strip()])

    # Concatenate all data frames
    df = pd.concat(dfs)
    path = os.path.join(root, output_file)
    df.to_csv(path, index=False)
