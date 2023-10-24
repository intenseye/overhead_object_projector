import json
import os
import random
from typing import Optional
from datetime import datetime
from argparse import ArgumentParser


def split_dataset(folder_path: str, validation_ratio: float, test_ratio: float,
                  train_ratio: Optional[float] = None) -> None:
    """
    Split dataset into training, validation and test subset

    Parameters
    ----------
    folder_path: str
        Path to data folder
    validation_ratio: float
        Validation set split ratio
    test_ratio: float
        Test set split ratio
    train_ratio: Optional[float]
    """

    split_folder_name = 'split'
    time_stamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    split_folder_path = os.path.join(folder_path, split_folder_name, time_stamp)
    os.makedirs(split_folder_path, exist_ok=True)

    json_file_name = "coordinates.json"
    input_json_path = os.path.join(folder_path, json_file_name)

    data_val = {}
    data_test = {}
    data_train = {}
    with open(input_json_path, 'r') as f:
        data = f.read()
        file_content = json.loads(data)

        input_size = len(file_content)
        val_input_size = int(input_size * validation_ratio)
        test_input_size = int(input_size * test_ratio)
        train_input_size = min(input_size - (val_input_size + test_input_size), int(input_size * train_ratio))

        random_indices = random.sample(range(input_size), input_size)
        val_indices = random_indices[0:val_input_size]
        test_indices = random_indices[val_input_size:test_input_size + val_input_size]
        train_indices = random_indices[-train_input_size:]

        index_count = 0
        for key in file_content:
            current_frame_info = file_content[key]
            if current_frame_info["any_labelled"] is True:
                if index_count in val_indices:
                    data_val[key] = file_content[key]
                elif index_count in test_indices:
                    data_test[key] = file_content[key]
                elif index_count in train_indices:
                    data_train[key] = file_content[key]
                index_count += 1
            else:
                print('Unlabelled data: ' + current_frame_info['image_path'])

        with open(os.path.join(split_folder_path, "coordinates_val.json"), "w") as convert_file:
            convert_file.write(json.dumps(data_val, sort_keys=False, indent=4, separators=(",", ": ")))
        with open(os.path.join(split_folder_path, "coordinates_test.json"), "w") as convert_file:
            convert_file.write(json.dumps(data_test, sort_keys=False, indent=4, separators=(",", ": ")))
        with open(os.path.join(split_folder_path, "coordinates_train.json"), "w") as convert_file:
            convert_file.write(json.dumps(data_train, sort_keys=False, indent=4, separators=(",", ": ")))


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to delete unlabelled content")
    parser.add_argument("--data_folder_path", help="Path to the json file including annotations.",
                        default=r"./datasets/CraneIntenseye/Set01")
    parser.add_argument("--val_ratio", help="Validation split data ratio.", default='0.15')
    parser.add_argument("--test_ratio", help="Test split data ratio.", default='0.15')
    parser.add_argument("--train_ratio", help="Test split data ratio.", default=None)

    args = parser.parse_args()
    data_folder_path = args.data_folder_path
    val_ratio_ = float(args.val_ratio)
    test_ratio_ = float(args.test_ratio)
    train_ratio_ = float(args.train_ratio)

    for i in range(100):
        if train_ratio_ is None:
            split_dataset(data_folder_path, val_ratio_, test_ratio_)
        else:
            split_dataset(data_folder_path, val_ratio_, test_ratio_, train_ratio_)
