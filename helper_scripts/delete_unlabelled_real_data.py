import json
import os
from argparse import ArgumentParser


def delete_unlabelled_data(folder_path: str) -> None:
    """
    Deletes unlabelled data and json fields

    Parameters
    ----------
    folder_path: str
        Path to data folder
    """

    json_file_name = "coordinates.json"
    input_json_path = os.path.join(folder_path, json_file_name)
    data_filtered = {}
    with open(input_json_path, 'r') as f:
        data = f.read()
        file_content = json.loads(data)
        for key in file_content:
            current_frame_info = file_content[key]
            if current_frame_info["any_labelled"] is True:
                data_filtered[key] = file_content[key]
            else:
                image_relative_path = current_frame_info['image_path']
                to_be_deleted_image_path = os.path.join(folder_path, image_relative_path)
                if os.path.isfile(to_be_deleted_image_path):
                    os.remove(to_be_deleted_image_path)
                else:
                    print('File not exists: ' + to_be_deleted_image_path)
        with open(input_json_path, "w") as convert_file:
            convert_file.write(json.dumps(data_filtered, sort_keys=False, indent=4, separators=(",", ": ")))


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to delete unlabelled content")
    parser.add_argument("--data_folder_path", help="Path to the json file including annotations.",
                        default=r"/home/poyraz/intenseye/input_outputs/overhead_object_projector/datasets/segezha")

    args = parser.parse_args()
    data_folder_path = args.data_folder_path
    delete_unlabelled_data(data_folder_path)
