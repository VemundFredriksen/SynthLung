import os
import shutil
import json

class ImageSourceFormatter():
    def format_directory(directory_path: str):
        pass

class MSDImageSourceFormatter(ImageSourceFormatter):
    def __init__(self) -> None:
        self.target_directory = "./assets/source_images/msd/"

    def format_directory(self, directory_path: str) -> None:
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

        self.__move_images__(directory_path + "/imagesTr/", "image")
        self.__move_images__(directory_path + "/labelsTr/", "label")
        self.__generate_json__()
    
    def __move_images__(self, images_directory: str, suffix: str) -> None:
        for filename in os.listdir(images_directory):
            if filename.endswith(('.nii.gz')):
                source_file_path = os.path.join(images_directory, filename)
                identity= filename[:filename.index('.nii.gz')]

                new_filename = f"source_msd_{identity}_{suffix}.nii.gz"
                target_file_path = os.path.join(self.target_directory, new_filename)

                shutil.copy(source_file_path, target_file_path)
    
    def __generate_json__(self) -> None:
        dataset_json = []
        for filename in os.listdir(self.target_directory):
            if filename.endswith(('image.nii.gz')):
                sample_data = {
                    "image": self.target_directory + filename,
                    "label": self.target_directory + filename[:filename.index('image.nii.gz')] + 'label.nii.gz'
                }
                dataset_json.append(sample_data)

        with open(self.target_directory + "/dataset.json", 'w') as json_file:
            json.dump(dataset_json, json_file, indent=4)




def main():
    formatter = MSDImageSourceFormatter()
    formatter.format_directory('./assets/Task06_Lung/')

main()