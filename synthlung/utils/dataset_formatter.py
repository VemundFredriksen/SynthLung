import os
import shutil
import json
from abc import ABC, abstractmethod

NII_GZ_EXTENSION = '.nii.gz'
IMAGE_NII_GZ = 'image.nii.gz'
LABEL_NII_GZ = 'label.nii.gz'

class ImageSourceFormatter(ABC):

    @abstractmethod
    def format(self) -> None:
        pass

class JSONGenerator(ABC):

    @abstractmethod
    def generate_json(self) -> None:
        pass

class MSDImageSourceFormatter(ImageSourceFormatter, JSONGenerator):
    def __init__(self) -> None:
        self.target_directory = "./assets/source/msd/"
        self.source_directory = "./assets/Task06_Lung/"

    def format(self) -> None:
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

        self.__move_images__(self.source_directory + "/imagesTr/", "image")
        self.__move_images__(self.source_directory + "/labelsTr/", "label")
    
    def generate_json(self) -> None:
        self.__generate_json__()
    
    def __move_images__(self, images_directory: str, suffix: str) -> None:
        for filename in os.listdir(images_directory):
            if filename.endswith((NII_GZ_EXTENSION)):
                source_file_path = os.path.join(images_directory, filename)
                identity= filename[:filename.index(NII_GZ_EXTENSION)]

                new_filename = f"source_msd_{identity}_{suffix}{NII_GZ_EXTENSION}"
                target_file_path = os.path.join(self.target_directory, new_filename)

                shutil.copy(source_file_path, target_file_path)
    
    def __generate_json__(self) -> None:
        dataset_json = []
        for filename in os.listdir(self.target_directory):
            if filename.endswith((IMAGE_NII_GZ)):
                sample_data = {
                    "image": self.target_directory + filename,
                    "label": self.target_directory + filename[:filename.index(IMAGE_NII_GZ)] + LABEL_NII_GZ
                }
                dataset_json.append(sample_data)

        with open(self.target_directory + "/dataset.json", 'w') as json_file:
            json.dump(dataset_json, json_file, indent=4)

class MSDGenerateJSONFormatter(JSONGenerator):
    def __init__(self, path) -> None:
        self.path = path

    def generate_json(self) -> None:
        self.__generate_json__()

    def __generate_json__(self) -> None:
        dataset_json = []
        for filename in os.listdir(self.path):
            if filename.endswith((IMAGE_NII_GZ)):
                sample_data = {
                    "seed_image": self.path + filename,
                    "seed_label": self.path + filename[:filename.index(IMAGE_NII_GZ)] + LABEL_NII_GZ
                }
                dataset_json.append(sample_data)

        with open(self.path + "/dataset.json", 'w') as json_file:
            json.dump(dataset_json, json_file, indent=4)
