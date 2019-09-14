import os
import cv2
import csv
from .csv_path import CSV_FILE_NAME


IMG_RELATIVE_PATH = "imgs"


def create_csv():
    cur_path = os.getcwd()
    imgs_path = os.path.join(cur_path, IMG_RELATIVE_PATH)
    imgs_type_paths = {
        garbage_type: retrieve_all_paths(img_type_path, os.listdir(img_type_path)) for img_type_path, garbage_type in
        [(os.path.join(imgs_path, garbage_type), garbage_type) for garbage_type in os.listdir(imgs_path)]
    }
    imgs_metadata = [
        retrieve_metadata_from_files(imgs_type_paths[garbage_type], garbage_type) for garbage_type in imgs_type_paths
    ]
    imgs_metadata = [
        metadata for list_of_metadata in imgs_metadata for metadata in list_of_metadata
    ]
    with open(CSV_FILE_NAME, "w") as csvfile:
        field_names = ["garbage_type", "path", "height", "width"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(imgs_metadata)
    print("Complete.")


def retrieve_all_paths(img_type_path, img_files):
    return [os.path.join(img_type_path, img_file) for img_file in img_files]


def retrieve_metadata_from_files(img_paths, garbage_type):
    return [retrieve_metadata_from_file(img_path, garbage_type) for img_path in img_paths]


def retrieve_metadata_from_file(img_path, garbage_type):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[0], img.shape[1]
    return {
        "garbage_type": garbage_type,
        "path": img_path,
        "height": height,
        "width": width
    }


if __name__ == "__main__":
    create_csv()
