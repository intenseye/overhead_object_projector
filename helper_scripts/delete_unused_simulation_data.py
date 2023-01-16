import json
import os
from argparse import ArgumentParser


def delete_unlabelled_data(main_folder_path: str) -> None:
    """
    Deletes unlabelled data and json fields

    Parameters
    ----------
    main_folder_path: str
        Path to data folder
    """
    dev_folder_paths = [os.path.join(main_folder_path, name) for name in os.listdir(main_folder_path)
                        if os.path.isdir(os.path.join(main_folder_path, name))]
    input_json_paths = []
    relative_json_paths = ["split/coordinates_test.json", "split/coordinates_train.json", "split/coordinates_val.json"]
    for relative_json_path in relative_json_paths:
        input_json_paths.append(os.path.join(dev_folder_paths[0], relative_json_path))

    to_be_kept_image_paths = []
    for input_json_path in input_json_paths:
        with open(input_json_path, 'r') as f:
            data = f.read()
            file_content = json.loads(data)
            for key in file_content:
                current_frame_info = file_content[key]
                if current_frame_info["any_labelled"] is True:
                    image_relative_path = os.path.basename(current_frame_info['image_path'])
                    to_be_kept_image_paths.append(image_relative_path)

    for dev_folder_path in dev_folder_paths:
        image_list = [name for name in os.listdir(os.path.join(dev_folder_path, 'images'))]
        for image_name in image_list:
            if not image_name in to_be_kept_image_paths:
                to_be_deleted_image_path = os.path.join(dev_folder_path, 'images', image_name)
                if os.path.isfile(to_be_deleted_image_path):
                    os.remove(to_be_deleted_image_path)
                else:
                    print('File not exists: ' + to_be_deleted_image_path)


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to delete unlabelled content")
    parser.add_argument("--data_folder_path", help="Path to the json file including annotations.",
                        default=r"/home/poyraz/intenseye/input_outputs/overhead_object_projector/datasets/OverheadSimIntenseye_2/Set01")

    args = parser.parse_args()
    data_folder_path = args.data_folder_path
    delete_unlabelled_data(data_folder_path)
