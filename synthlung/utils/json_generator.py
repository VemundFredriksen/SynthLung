import os, json

class JSONGenerator():

    NII_GZ_EXTENSION = '.nii.gz'
    IMAGE_NII_GZ = 'image.nii.gz'
    LABEL_NII_GZ = 'label.nii.gz'

    def generate_json(self, image_name, label_name, path) -> None:
        dataset_json = []
        for filename in os.listdir(path):
            if filename.endswith((self.IMAGE_NII_GZ)):
                sample_data = {
                    image_name: path + filename,
                    label_name: path + filename[:filename.index(self.IMAGE_NII_GZ)] + self.LABEL_NII_GZ
                }
                dataset_json.append(sample_data)

        with open(path + "/dataset.json", 'w') as json_file:
            json.dump(dataset_json, json_file, indent=4)