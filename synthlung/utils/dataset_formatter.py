import os
import shutil
from synthlung.utils.json_generator import JSONGenerator
from synthlung.utils.image_source_formatter import ImageSourceFormatter

NII_GZ_EXTENSION = '.nii.gz'
IMAGE_NII_GZ = 'image.nii.gz'
LABEL_NII_GZ = 'label.nii.gz'

class MSDImageSourceFormatter(ImageSourceFormatter, JSONGenerator):
    def __init__(self, source_directory: str = "./assets/images/Task06_Lung/", target_directory: str = "./assets/images/source/") -> None:
        self.target_directory = target_directory
        self.source_directory = source_directory

    def format(self) -> None:
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

        self._move_images(self.source_directory + "/imagesTr/", "image")
        self._move_images(self.source_directory + "/labelsTr/", "label")
    
    def generate_json(self) -> None:
        super().generate_json("image", "label", self.target_directory)
    
    def _move_images(self, images_directory: str, suffix: str) -> None:
        for filename in os.listdir(images_directory):
            if filename.endswith((NII_GZ_EXTENSION)):
                source_file_path = os.path.join(images_directory, filename)
                identity= filename[:filename.index(NII_GZ_EXTENSION)]

                new_filename = f"source_msd_{identity}_{suffix}{NII_GZ_EXTENSION}"
                target_file_path = os.path.join(self.target_directory, new_filename)

                shutil.copy(source_file_path, target_file_path)

class JsonSeedGenerator(JSONGenerator):
    def __init__(self, path) -> None:
        self.path = path

    def generate_json(self) -> None:
        super().generate_json("seed_image", "seed_label", self.path)

class JsonTrainingGenerator(JSONGenerator):
    def __init__(self, path) -> None:
        self.path = path

    def generate_json(self) -> None:
        super().generate_json("image", "label", self.path)
